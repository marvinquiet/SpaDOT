import os
import hashlib
import json
import re
from pathlib import Path
import scipy
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from SpaDOT.utils import _utils
from SpaDOT.utils.sctransform import SCTransform


_MT_CANONICAL = {'ATP6', 'ATP8', 'CO1', 'CO2', 'CO3', 'COX1', 'COX2',
                 'COX3', 'CYTB', 'COB', 'ND1', 'ND2', 'ND3', 'ND4',
                 'ND4L', 'ND5', 'ND6', 'RNR1', 'RNR2'}


def _ordered_timepoints(adata):
    """Map comma-separated quotas to timepoints in natural label order."""
    labels = pd.unique(adata.obs['timepoint'].astype(str)).tolist()
    return sorted(labels, key=lambda label: tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', label)))


def _feature_counts(spec, timepoints):
    """Validate one integer quota for every observed timepoint."""
    if isinstance(spec, str):
        parts = spec.split(',')
    elif isinstance(spec, (list, tuple)):
        parts = list(spec)
    else:
        raise ValueError('--n_features must be comma-separated integers')
    if len(parts) != len(timepoints):
        raise ValueError(
            f'--n_features needs {len(timepoints)} numbers for timepoints '
            f'{timepoints}; received {len(parts)}')
    counts = []
    for part in parts:
        if isinstance(part, bool) or not str(part).strip().isdigit():
            raise ValueError('--n_features must contain only positive integers')
        count = int(part)
        if count < 100:
            raise ValueError('Each --n_features quota must be at least 100')
        counts.append(count)
    return dict(zip(timepoints, counts))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _retained_genes(adata):
    """Use the frozen mitochondrial/ribosomal gene-symbol exclusion rule."""
    if 'original_genes' in adata.var:
        annotations = adata.var['original_genes'].fillna('').astype(str)
    else:
        annotations = adata.var_names.astype(str)
    retained = []
    for name, annotation in zip(adata.var_names, annotations):
        first = str(annotation).split(' | ')[0]
        symbols = {re.sub(r'\.(L|S)$', '',
                          re.sub(r'\[[^]]*\]', '', token).strip().upper())
                   for token in first.split('|')}
        pseudo = lambda s: bool(re.search(r'-PS\d+$', s) or
                                re.match(r'^RP[LS][0-9A-Z]+P\d+$', s))
        ribo = any(re.match(r'^RP[LS]', s) and not pseudo(s) and
                   not s.startswith('RPS6K') and s != 'RPS19BP1'
                   for s in symbols)
        mrp = any(re.match(r'^MRP[LS]', s) and not pseudo(s) for s in symbols)
        mt = any(s in _MT_CANONICAL or s.startswith(('MT-', 'MT_', 'MT.')) or
                 re.fullmatch(r'MT(?:CO[123]|ND[1-6]L?|ATP[68]|CYB|RNR[12])', s)
                 for s in symbols)
        if not (ribo or mrp or mt):
            retained.append(name)
    return retained


def _standardized_expression(normalized, names):
    matrix = normalized[:, names].X
    matrix = matrix.toarray() if scipy.sparse.issparse(matrix) else np.asarray(matrix)
    matrix = np.log1p(np.asarray(matrix, dtype=np.float64) * 1e6)
    matrix -= matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    valid = sd > 1e-8
    matrix[:, valid] /= sd[valid]
    matrix[:, ~valid] = 0
    return matrix, valid


def _pseudoregion_ranking(raw_part, retained, candidates, coordinates):
    """Balance one-versus-rest expression contrasts across 12 unlabeled regions."""
    normalized = raw_part[:, retained].copy()
    sc.pp.normalize_total(normalized, target_sum=1e-4)
    n_pca_genes = min(1100, len(candidates))
    pca_input, _ = _standardized_expression(normalized, candidates[:n_pca_genes])
    pcs = PCA(n_components=8, whiten=True, svd_solver='randomized',
              random_state=2026).fit_transform(pca_input)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    coords_z = ((coordinates - coordinates.mean(axis=0)) /
                np.maximum(coordinates.std(axis=0), 1e-8))
    embedding = np.column_stack([pcs / np.sqrt(8), coords_z])
    groups = KMeans(n_clusters=12, random_state=2026, n_init=10,
                    init='k-means++', algorithm='lloyd').fit_predict(embedding)
    sizes = np.bincount(groups, minlength=12)
    eligible = [group for group in range(12) if sizes[group] >= 20]
    if not eligible:
        raise ValueError('No pseudoregion contains at least 20 spots')
    expression, valid = _standardized_expression(normalized, candidates)
    ranked_by_group = {}
    for group in eligible:
        inside = groups == group
        effect = expression[inside].mean(axis=0) - expression[~inside].mean(axis=0)
        effect[~valid] = -np.inf
        order = np.argsort(-effect, kind='mergesort')
        ranked_by_group[group] = [candidates[i] for i in order]
    ranked, seen = [], set()
    positions = {group: 0 for group in eligible}
    while len(ranked) < len(candidates):
        changed = False
        for group in eligible:
            names = ranked_by_group[group]
            while positions[group] < len(names) and names[positions[group]] in seen:
                positions[group] += 1
            if positions[group] < len(names):
                name = names[positions[group]]
                positions[group] += 1
                ranked.append(name)
                seen.add(name)
                changed = True
        if not changed:
            raise ValueError('Pseudoregion ranking did not cover candidates')
    return ranked, sizes.tolist()


def _python_pseudoregion_panel(adata, timepoints, quotas):
    """Select SVGs from raw counts using the bundled Python SPARK-X."""
    retained = _retained_genes(adata)
    selected, detail = {}, {}
    for tp in timepoints:
        part = adata[adata.obs['timepoint'] == tp].copy()
        coordinates = np.asarray(part.obsm['spatial'], dtype=np.float64)
        count = scipy.sparse.csr_matrix(part[:, retained].X)
        spark = _utils._sparkx(count, coordinates, np.asarray(retained),
                               option='mixture', num_cores=1)
        table = spark.reset_index().rename(columns={'index': 'gene'})
        table = table.loc[table.adjustedPval <= 0.05].sort_values(
            ['adjustedPval', 'combinedPval', 'gene'], kind='mergesort')
        quota = quotas[tp]
        if len(table) < quota:
            raise ValueError(f'Timepoint {tp} has only {len(table)} FDR-significant '
                             f'SVGs, fewer than the requested {quota}')
        pool_size = min(len(table), max(1100, quota + min(quota, 1100)))
        candidates = table.gene.iloc[:pool_size].tolist()
        ranked, sizes = _pseudoregion_ranking(part, retained, candidates,
                                               coordinates)
        chosen = table.gene.iloc[:100].tolist()
        seen = set(chosen)
        for gene in ranked:
            if gene not in seen:
                chosen.append(gene)
                seen.add(gene)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f'Could not fill SVG quota for timepoint {tp}')
        selected[tp] = chosen
        detail[tp] = {'fdr_significant': len(table),
                      'candidate_pool': pool_size,
                      'pseudoregion_spot_counts': sizes}
    return selected, {'method': 'python_sparkx_pseudoregion',
                      'selection_details': detail}


def _frozen_panel(adata, args, timepoints, quotas):
    """Use saved SVGs for bit-identical reproduction of validated examples."""
    folder = Path(args.feature_panel)
    panel = json.loads((folder / 'panel.json').read_text())
    panel_timepoints = list(panel['selected_genes_per_timepoint'])
    if (len(panel_timepoints) != len(timepoints) or
            list(panel['selected_genes_per_timepoint'].values()) !=
            list(quotas.values())):
        raise ValueError('--n_features does not match the frozen panel quotas')
    if _sha256(args.data) != panel['raw_sha256']:
        raise ValueError('Raw input does not match the frozen panel hash')
    if _sha256(folder / 'spots.tsv') != panel['spot_metadata_sha256']:
        raise ValueError('Frozen panel spot metadata hash mismatch')
    spots = pd.read_csv(folder / 'spots.tsv', sep='\t',
                        float_precision='round_trip',
                        dtype={'spot_id': str, 'timepoint': str})
    if (not spots.spot_id.is_unique or
            spots.timepoint.value_counts().to_dict() != panel['spots_per_timepoint'] or
            not set(spots.spot_id).issubset(adata.obs_names)):
        raise ValueError('Frozen panel spot order or timepoints differ from the raw input')
    genes_file = folder / 'union_genes.txt'
    if _sha256(genes_file) != panel['union_genes_sha256']:
        raise ValueError('Frozen panel union gene hash mismatch')
    genes = genes_file.read_text().splitlines()
    selected = {tp: (folder / f'genes_{tp}.txt').read_text().splitlines()
                for tp in panel_timepoints}
    if (any(len(selected[tp]) != panel['selected_genes_per_timepoint'][tp] or
            len(selected[tp]) != len(set(selected[tp]))
            for tp in panel_timepoints) or
            genes != sorted(set().union(*(set(names) for names in selected.values()))) or
            not set(genes).issubset(adata.var_names)):
        raise ValueError('Frozen panel gene lists are inconsistent')
    adata = adata[spots.spot_id.tolist(), genes].copy()
    adata.obs['timepoint'] = spots.timepoint.to_numpy()
    adata.obsm['spatial'] = spots[['spatial_x', 'spatial_y']].to_numpy(dtype=np.float64)
    return (adata, selected,
            {'method': 'frozen_panel',
             'panel_sha256': _sha256(folder / 'panel.json')},
            panel_timepoints)


def _preprocess_with_feature_counts(args, adata):
    if not args.feature_selection:
        raise ValueError('--n_features requires feature selection')
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Raw spot and gene names must be unique')
    values = adata.X.data if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    if (not np.isfinite(values).all() or np.any(values < 0) or
            not np.allclose(values, np.rint(values), atol=1e-6)):
        raise ValueError('Pseudoregion selection requires finite, nonnegative raw counts')
    adata = adata.copy()
    adata.obs['timepoint'] = adata.obs['timepoint'].astype(str)
    timepoints = _ordered_timepoints(adata)
    quotas = _feature_counts(args.n_features, timepoints)
    if any(count > adata.n_vars for count in quotas.values()):
        raise ValueError('--n_features cannot exceed the raw gene count')
    if getattr(args, 'feature_panel', None):
        adata, selected, metadata, panel_timepoints = _frozen_panel(
            adata, args, timepoints, quotas)
        quotas = dict(zip(panel_timepoints, quotas.values()))
        timepoints = panel_timepoints
    else:
        selected, metadata = _python_pseudoregion_panel(adata, timepoints, quotas)
        genes = sorted(set().union(*(set(names) for names in selected.values())))
        adata = adata[:, genes].copy()
    genes = sorted(set().union(*(set(names) for names in selected.values())))
    output = Path(args.output_dir)
    for tp in timepoints:
        (output / f'SVG_genes_{tp}.txt').write_text(
            ''.join(gene + '\n' for gene in selected[tp]))
    (output / 'SVG_genes.txt').write_text(''.join(gene + '\n' for gene in genes))
    metadata['timepoints'] = timepoints
    metadata['n_features'] = quotas
    metadata['union_genes'] = len(genes)
    (output / 'SVG_selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    parts = []
    for tp in timepoints:
        part = adata[adata.obs['timepoint'] == tp].copy()
        part.layers['counts'] = part.X.copy()
        sc.pp.normalize_total(part, target_sum=1e-4)
        sc.pp.log1p(part)
        sc.pp.scale(part)
        parts.append(part)
    result = anndata.concat(parts, merge='same')[adata.obs_names.tolist(), genes].copy()
    if not np.isfinite(result.X).all():
        raise ValueError('Preprocessed expression contains non-finite values')
    return result

def preprocess_adata(args, adata):
    '''
    Preprocess the data for SpaDOT model training.
    Args:
        args: arguments
        adata: AnnData object
    Returns:
        preprocessed_adata: preprocessed AnnData object
    '''
    if getattr(args, 'n_features', None) is not None:
        return _preprocess_with_feature_counts(args, adata)
    if getattr(args, 'feature_panel', None):
        raise ValueError('--feature_panel requires --n_features')
    tps = adata.obs['timepoint'].unique()
    if args.feature_selection:
        _run_sparkx(args, adata, tps, option='mixture', num_cores=4)
        SVGs = _get_SVGs(args, tps)
        adata = adata[:, SVGs].copy()
    # save counts in adata.layers
    if not scipy.sparse.issparse(adata.X):
        adata.layers['counts'] = scipy.sparse.csr_matrix(adata.X)
    else:
        adata.layers['counts'] = adata.X
    # --- preprocess data
    tp_adata_list = []
    for tp in tps:
        tp_adata = adata[adata.obs['timepoint'] == tp]
        sc.pp.normalize_total(tp_adata, target_sum=1e-4)
        sc.pp.log1p(tp_adata)
        tp_adata_list.append(tp_adata)
    # union SVG genes for all time points
    if args.feature_selection:
        genes = set().union(*(obj.var_names for obj in tp_adata_list))
        genes = list(genes)
        genes.sort()
        with open(args.output_dir+os.sep+'SVG_genes.txt', 'w') as f:
            for item in genes:
                f.write("%s\n" % item)  
    new_tp_adata_list = []
    for tp_adata in tp_adata_list:
        if args.feature_selection:
            tp_adata = tp_adata[:, genes].copy()
        sc.pp.scale(tp_adata)
        new_tp_adata_list.append(tp_adata)
    preprocessed_adata = anndata.concat(new_tp_adata_list)
    return preprocessed_adata


def _get_SVGs(args, tps):
    '''
    Run SPARKX to select SVGs.
    Args:
        args: arguments
    Returns:
        SVG_genes: list of SVG genes
    '''
    # --- after running SPARKX, the SVGs are saved in the result directory
    ## select SVGs by cluster
    tp_SVGs = []
    for tp in tps:
        tp_SVGs_cluster = pd.read_csv(args.output_dir+os.sep+str(tp)+'_SVG_sparkx_clustered_louvain.csv', header=0, index_col=0)
        tp_SVGs.append(tp_SVGs_cluster)
    min_idx = min(range(len(tp_SVGs)), key=lambda i: len(tp_SVGs[i]))
    min_tp_SVG_len = len(tp_SVGs[min_idx])
    SVG_genes = tp_SVGs[min_idx].index.tolist()
    for idx, tp_SVG in enumerate(tp_SVGs):
        if idx == min_idx: continue
        tp_SVG_num_clusters = len(set(tp_SVG['cluster']))
        tp_SVG_top = tp_SVG.sort_values(by='adjustedPval', ascending=True).groupby('cluster').head(max(100, round(min_tp_SVG_len/tp_SVG_num_clusters)))
        SVG_genes.extend(tp_SVG_top.index.tolist())
    SVG_genes = list(set(SVG_genes))
    SVG_genes.sort()
    return SVG_genes


def _run_sparkx(args, adata, tps, option='mixture', num_cores=4):
    '''
    Run SPARKX-X python version and write results into files.
    '''
    for tp in tps:
        tp_adata = adata[adata.obs['timepoint'] == tp].copy()
        tp_adata.layers['counts'] = tp_adata.X.copy()
        assay_out, vst_out = SCTransform(tp_adata.X.T,
                            genes=tp_adata.var_names,
                            cells=tp_adata.obs_names,
                            return_only_var_genes=False, 
                            n_cells=None,
                            variable_features_n=None,
                            variable_features_rv_th=1.3)
        tp_adata = tp_adata[:, assay_out['scale.data'].index].copy()
        print(f'Timepoint: {tp}, Number of cells: {tp_adata.n_obs}, Number of genes: {tp_adata.n_vars}')
        count_spark = tp_adata.layers['counts']
        locations_spark = tp_adata.obsm['spatial']
        SVGs = _utils._sparkx(count_spark, locations_spark, np.array(tp_adata.var_names), option=option, num_cores=num_cores)
        # SVGs.to_csv(args.output_dir+os.sep+str(tp)+'_SVG_sparkx.csv')
        SVG_clusters = _utils._cluster_SVGs(assay_out['scale.data'].loc[SVGs.index, :], k=10)
        SVGs['cluster'] = SVG_clusters
        SVGs.to_csv(args.output_dir+os.sep+str(tp)+'_SVG_sparkx_clustered_louvain.csv')
