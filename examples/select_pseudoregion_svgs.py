"""Recalculate the fixed-99 SVG panels without tissue labels or markers.

First export filtered raw counts for R SPARK-X, then rank its FDR-significant
genes by spatial/expression pseudoregion contrasts. See fixed99_panels/README.md.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from fixed99_example import _sha256


MT_CANONICAL = {'ATP6', 'ATP8', 'CO1', 'CO2', 'CO3', 'COX1', 'COX2',
                'COX3', 'CYTB', 'COB', 'ND1', 'ND2', 'ND3', 'ND4',
                'ND4L', 'ND5', 'ND6', 'RNR1', 'RNR2'}


def _symbols(annotation):
    first = str(annotation).split(' | ')[0]
    return sorted(set(re.sub(r'\.(L|S)$', '',
                             re.sub(r'\[[^]]*\]', '', token).strip().upper())
                      for token in first.split('|')))


def _pseudogene(symbol):
    return bool(re.search(r'-PS\d+$', symbol) or
                re.match(r'^RP[LS][0-9A-Z]+P\d+$', symbol))


def _remove_mt_ribo(annotation):
    names = _symbols(annotation)
    ribo = any(re.match(r'^RP[LS]', s) and not _pseudogene(s) and
               not s.startswith('RPS6K') and s != 'RPS19BP1' for s in names)
    mrp = any(re.match(r'^MRP[LS]', s) and not _pseudogene(s) for s in names)
    mt = any(s in MT_CANONICAL or s.startswith(('MT-', 'MT_', 'MT.')) or
             re.fullmatch(r'MT(?:CO[123]|ND[1-6]L?|ATP[68]|CYB|RNR[12])', s)
             for s in names)
    return bool(ribo or mrp or mt)


def retained_genes(raw):
    """Apply the frozen gene-symbol annotation rule before SPARK-X."""
    if 'original_genes' in raw.var:
        annotations = raw.var['original_genes'].fillna('').astype(str)
    else:
        annotations = raw.var_names.astype(str)
    return [name for name, annotation in zip(raw.var_names, annotations)
            if not _remove_mt_ribo(annotation)]


def _load_spots(panel_dir):
    panel = json.loads((panel_dir / 'panel.json').read_text())
    spots_path = panel_dir / 'spots.tsv'
    if _sha256(spots_path) != panel['spot_metadata_sha256']:
        raise ValueError('Spot metadata hash mismatch')
    spots = pd.read_csv(spots_path, sep='\t', float_precision='round_trip',
                        dtype={'spot_id': str, 'timepoint': str})
    if not spots.spot_id.is_unique:
        raise ValueError('Spot IDs must be unique')
    return panel, spots


def _standardized(full, names):
    x = full[:, names].X
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x = np.log1p(np.asarray(x, dtype=np.float64) * 1e6)
    x -= x.mean(axis=0)
    sd = x.std(axis=0)
    valid = sd > 1e-8
    x[:, valid] /= sd[valid]
    x[:, ~valid] = 0
    return x, valid


def _pseudoregion_ranking(raw_part, retained, candidates, coords, pca_pool):
    full = raw_part[:, retained].copy()
    sc.pp.normalize_total(full, target_sum=1e-4)
    x_pca, _ = _standardized(full, candidates[:pca_pool])
    pcs = PCA(n_components=8, whiten=True, svd_solver='randomized',
              random_state=2026).fit_transform(x_pca)
    coords_z = (coords - coords.mean(axis=0)) / np.maximum(coords.std(axis=0), 1e-8)
    embedding = np.column_stack([pcs / np.sqrt(8), coords_z])
    groups = KMeans(n_clusters=12, random_state=2026, n_init=10,
                    init='k-means++', algorithm='lloyd').fit_predict(embedding)
    counts = np.bincount(groups, minlength=12)
    eligible = [group for group in range(12) if counts[group] >= 20]
    if not eligible:
        raise ValueError('No pseudoregion has at least 20 spots')

    x, valid = _standardized(full, candidates)
    per_group = {}
    for group in eligible:
        inside = groups == group
        effect = x[inside].mean(axis=0) - x[~inside].mean(axis=0)
        effect[~valid] = -np.inf
        order = np.argsort(-effect, kind='mergesort')
        per_group[group] = [candidates[i] for i in order]
    ranked, seen = [], set()
    positions = {group: 0 for group in eligible}
    while len(ranked) < len(candidates):
        changed = False
        for group in eligible:
            names = per_group[group]
            while positions[group] < len(names) and names[positions[group]] in seen:
                positions[group] += 1
            if positions[group] < len(names):
                name = names[positions[group]]
                positions[group] += 1
                ranked.append(name)
                seen.add(name)
                changed = True
        if not changed:
            raise ValueError('Pseudoregion ranking did not cover the candidates')
    return ranked, counts.tolist()


def select_timepoint(raw_part, coords, retained, sparkx_csv, quota,
                     candidate_pool, pca_pool, prefix=None):
    """Choose one quota from SPARK-X significance and pseudoregion contrasts."""
    if pca_pool > candidate_pool or pca_pool < 8:
        raise ValueError('Invalid PCA candidate pool size')
    table = pd.read_csv(sparkx_csv).sort_values(
        ['adjustedPval', 'combinedPval', 'gene'], kind='mergesort')
    if (not table.gene.is_unique or
            not np.isfinite(table[['adjustedPval', 'combinedPval']]).all().all() or
            not set(table.gene).issubset(set(retained))):
        raise ValueError('Invalid SPARK-X gene table')
    significant = table.loc[table.adjustedPval <= 0.05]
    if len(significant) < candidate_pool or candidate_pool < quota:
        raise ValueError('Insufficient FDR-significant SVG candidates')
    candidates = significant.gene.iloc[:candidate_pool].tolist()
    if prefix is not None:
        if len(prefix) > quota or not set(prefix).issubset(candidates):
            raise ValueError('Invalid frozen-prefix list')
        if len(prefix) == quota:
            return prefix, {'candidate_count': candidate_pool, 'prefix_count': len(prefix)}
    ranked, counts = _pseudoregion_ranking(raw_part, retained, candidates,
                                           coords, pca_pool)
    chosen = list(prefix) if prefix is not None else candidates[:100]
    seen = set(chosen)
    for name in ranked:
        if name not in seen:
            chosen.append(name)
            seen.add(name)
        if len(chosen) == quota:
            break
    if len(chosen) != quota or not set(candidates[:100]).issubset(chosen):
        raise ValueError('Quota or protected SPARK-X genes missing')
    return chosen, {'candidate_count': candidate_pool,
                    'prefix_count': len(prefix) if prefix else 0,
                    'pseudoregion_spot_counts': counts}


def export_for_sparkx(raw, spots, retained, output_dir):
    """Write annotation-free count matrices and coordinates for the R script."""
    output_dir.mkdir(parents=True, exist_ok=False)
    for tp in spots.timepoint.unique():
        part_spots = spots.loc[spots.timepoint == tp]
        part = raw[part_spots.spot_id.tolist(), retained]
        folder = output_dir / tp
        folder.mkdir()
        mmwrite(folder / 'counts.mtx', sparse.csr_matrix(part.X).T)
        (folder / 'genes.txt').write_text(''.join(g + '\n' for g in retained))
        (folder / 'cells.txt').write_text(''.join(c + '\n' for c in part_spots.spot_id))
        part_spots[['spatial_x', 'spatial_y']].to_csv(
            folder / 'coordinates.tsv', sep='\t', index=False)


def select_panel(raw, spots, retained, panel, panel_dir, sparkx_dir, output_dir):
    """Rebuild the per-timepoint lists and sorted union into a fresh directory."""
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = {}
    params = panel['selection_parameters']
    for tp, quota in panel['selected_genes_per_timepoint'].items():
        part_spots = spots.loc[spots.timepoint == tp]
        part = raw[part_spots.spot_id.tolist()]
        coords = part_spots[['spatial_x', 'spatial_y']].to_numpy(dtype=np.float64)
        prefix_path = panel_dir / f'prefix_{tp}.txt'
        prefix = prefix_path.read_text().splitlines() if prefix_path.exists() else None
        table_path = sparkx_dir / tp / 'R_SPARKX_all_retained.csv'
        if _sha256(table_path) != params['sparkx_table_sha256'][tp]:
            raise ValueError(f'SPARK-X table hash differs for timepoint {tp}')
        genes, details = select_timepoint(
            part, coords, retained, table_path, quota,
            params['candidate_pool_per_timepoint'][tp],
            params['pca_pool_per_timepoint'][tp], prefix)
        selected[tp] = genes
        (output_dir / f'genes_{tp}.txt').write_text(''.join(g + '\n' for g in genes))
        print(tp, len(genes), details, flush=True)
    union = sorted(set().union(*(set(names) for names in selected.values())))
    (output_dir / 'union_genes.txt').write_text(''.join(g + '\n' for g in union))
    for name in [f'genes_{tp}.txt' for tp in selected] + ['union_genes.txt']:
        expected = (panel_dir / name).read_text()
        if (output_dir / name).read_text() != expected:
            raise ValueError(f'Regenerated {name} differs from the frozen panel')
    return union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--export-r-inputs', action='store_true')
    parser.add_argument('--sparkx-dir', type=Path)
    args = parser.parse_args()
    panel, spots = _load_spots(args.panel_dir)
    if _sha256(args.raw) != panel['raw_sha256']:
        raise ValueError('Raw data hash differs from the frozen source')
    raw = ad.read_h5ad(args.raw)
    if not spots.spot_id.is_unique or not set(spots.spot_id).issubset(raw.obs_names):
        raise ValueError('Spot IDs differ from raw data')
    retained = retained_genes(raw)
    retained_hash = hashlib.sha256(''.join(g + '\n' for g in retained).encode()).hexdigest()
    if (len(retained) != panel['retained_genes'] or
            retained_hash != panel['retained_genes_sha256']):
        raise ValueError('Mitochondrial/ribosomal filter differs from the frozen rule')
    if args.export_r_inputs:
        export_for_sparkx(raw, spots, retained, args.output_dir)
        return
    if args.sparkx_dir is None:
        parser.error('--sparkx-dir is required unless --export-r-inputs is set')
    select_panel(raw, spots, retained, panel, args.panel_dir,
                 args.sparkx_dir, args.output_dir)


if __name__ == '__main__':
    main()
