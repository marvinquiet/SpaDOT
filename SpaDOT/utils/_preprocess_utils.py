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


def _selection_plan(path, timepoints, quotas):
    """Read numeric selection stages; no genes, ranks, or labels are loaded."""
    plan = json.loads(Path(path).read_text())
    if set(plan) != {'timepoints'} or set(plan['timepoints']) != set(timepoints):
        raise ValueError('--selection_plan needs exactly the observed timepoints')
    parsed = {}
    for tp in timepoints:
        spec = plan['timepoints'][tp]
        if set(spec) != {'pca_pool', 'stages'}:
            raise ValueError(f'{tp}: expected pca_pool and stages')
        pca_pool = spec['pca_pool']
        stages = spec['stages']
        if (type(pca_pool) is not int or pca_pool < 8 or
                not isinstance(stages, list) or not stages):
            raise ValueError(f'{tp}: invalid PCA pool or selection stages')
        previous = 0
        for stage in stages:
            if (not isinstance(stage, list) or len(stage) != 2 or
                    any(type(number) is not int for number in stage)):
                raise ValueError(f'{tp}: each stage needs [quota, candidate_pool]')
            count, pool = stage
            if (count < 100 or count <= previous or count > quotas[tp] or
                    pool < max(count, pca_pool)):
                raise ValueError(f'{tp}: invalid or nonascending selection stage')
            previous = count
        if previous != quotas[tp]:
            raise ValueError(f'{tp}: final stage must equal --n_features quota')
        parsed[tp] = {'pca_pool': pca_pool, 'stages': stages}
    return parsed


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


def _pseudoregion_ranking(raw_part, retained, candidates, coordinates,
                          pca_pool=1100):
    """Balance one-versus-rest expression contrasts across 12 unlabeled regions."""
    normalized = raw_part[:, retained].copy()
    sc.pp.normalize_total(normalized, target_sum=1e-4)
    n_pca_genes = min(pca_pool, len(candidates))
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


def _select_pseudoregion_genes(raw_part, retained, candidates, coordinates,
                               quota, pca_pool=1100, prefix=None):
    """Apply the historical top-100 and balanced pseudoregion rule."""
    if (len(candidates) < quota or len(candidates) != len(set(candidates)) or
            pca_pool < 8 or pca_pool > len(candidates)):
        raise ValueError('Invalid pseudoregion candidate pool')
    if prefix is not None and (len(prefix) > quota or
                               len(prefix) != len(set(prefix)) or
                               not set(prefix).issubset(candidates)):
        raise ValueError('Invalid pseudoregion prefix list')
    if prefix is not None and len(prefix) == quota:
        chosen, sizes = list(prefix), None
    else:
        ranked, sizes = _pseudoregion_ranking(
            raw_part, retained, candidates, coordinates, pca_pool)
        chosen = list(prefix) if prefix is not None else candidates[:100]
        seen = set(chosen)
        for gene in ranked:
            if gene not in seen:
                chosen.append(gene)
                seen.add(gene)
            if len(chosen) == quota:
                break
    if len(chosen) != quota or not set(candidates[:100]).issubset(chosen):
        raise ValueError('Pseudoregion quota or protected SVGs missing')
    return chosen, sizes


def _python_pseudoregion_panel(adata, timepoints, quotas, plan=None):
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
        if plan is None:
            pool_size = min(len(table), max(1100, quota + min(quota, 1100)))
            stages = [[quota, pool_size]]
            pca_pool = min(1100, pool_size)
        else:
            stages = plan[tp]['stages']
            pca_pool = plan[tp]['pca_pool']
        chosen = None
        stage_details = []
        for count, pool_size in stages:
            if len(table) < pool_size:
                raise ValueError(f'{tp}: only {len(table)} FDR-significant SVGs; '
                                 f'stage requires {pool_size}')
            candidates = table.gene.iloc[:pool_size].tolist()
            chosen, sizes = _select_pseudoregion_genes(
                part, retained, candidates, coordinates, count,
                pca_pool=pca_pool, prefix=chosen)
            stage_details.append({'quota': count, 'candidate_pool': pool_size,
                                  'pseudoregion_spot_counts': sizes})
        selected[tp] = chosen
        detail[tp] = {'fdr_significant': len(table),
                      'stages': stage_details}
    return selected, {'method': 'python_sparkx_pseudoregion',
                      'selection_details': detail}


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
    plan_file = getattr(args, 'selection_plan', None)
    plan = (_selection_plan(plan_file, timepoints, quotas)
            if plan_file else None)
    selected, metadata = _python_pseudoregion_panel(
        adata, timepoints, quotas, plan=plan)
    if plan_file:
        metadata['selection_plan_sha256'] = _sha256(plan_file)
    genes = sorted(set().union(*(set(names) for names in selected.values())))
    adata = adata[:, genes].copy()
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
    result = anndata.concat(parts, merge='same')[:, genes].copy()
    # Preserve an already-grouped input's spot order. For interleaved slices,
    # keep the concatenated timepoint blocks, preserving order within each
    # slice; the model samples inducing points by row index.
    labels = adata.obs['timepoint']
    if labels[labels.ne(labels.shift())].is_unique:
        result = result[adata.obs_names.tolist(), :].copy()
    if not np.isfinite(result.X).all():
        raise ValueError('Preprocessed expression contains non-finite values')
    return result

def preprocess_adata(args, adata):
    """Select a per-timepoint SVG panel, or preprocess all genes."""
    if getattr(args, 'n_features', None) is not None:
        return _preprocess_with_feature_counts(args, adata)
    if getattr(args, 'selection_plan', None):
        raise ValueError('--selection_plan requires --n_features')
    if args.feature_selection:
        raise ValueError(
            'Feature selection requires --n_features: provide one gene count '
            'per timepoint, or use --no-feature-selection to keep all genes.')
    adata = adata.copy()
    if not scipy.sparse.issparse(adata.X):
        adata.layers['counts'] = scipy.sparse.csr_matrix(adata.X)
    else:
        adata.layers['counts'] = adata.X.copy()
    parts = []
    for tp in adata.obs['timepoint'].unique():
        part = adata[adata.obs['timepoint'] == tp].copy()
        sc.pp.normalize_total(part, target_sum=1e-4)
        sc.pp.log1p(part)
        sc.pp.scale(part)
        parts.append(part)
    return anndata.concat(parts)
