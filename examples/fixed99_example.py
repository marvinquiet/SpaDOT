"""Prepare one audited, dataset-specific fixed-99 example panel from raw counts."""

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_panel(raw_path, panel_dir, output_path, verify_raw=True):
    """Apply frozen SVG lists, then normalize, log and scale each timepoint.

    The panel assets record gene identities, spot order and spatial coordinates
    from the validated inputs. They contain no tissue annotations or markers.
    """
    raw_path, panel_dir, output_path = map(Path, (raw_path, panel_dir, output_path))
    panel = json.loads((panel_dir / 'panel.json').read_text())
    if verify_raw and _sha256(raw_path) != panel['raw_sha256']:
        raise ValueError('Raw data hash differs from the validated example')
    genes_path = panel_dir / 'union_genes.txt'
    if _sha256(genes_path) != panel['union_genes_sha256']:
        raise ValueError('Union gene list hash mismatch')
    genes = genes_path.read_text().splitlines()
    quotas = panel['selected_genes_per_timepoint']
    by_timepoint = {}
    for tp, quota in quotas.items():
        names = (panel_dir / f'genes_{tp}.txt').read_text().splitlines()
        if len(names) != quota or len(names) != len(set(names)):
            raise ValueError(f'Invalid gene quota for timepoint {tp}')
        by_timepoint[tp] = names
    if genes != sorted(set().union(*(set(names) for names in by_timepoint.values()))):
        raise ValueError('Union does not match the timepoint lists')

    spots_path = panel_dir / 'spots.tsv'
    if _sha256(spots_path) != panel['spot_metadata_sha256']:
        raise ValueError('Spot metadata hash mismatch')
    spots = pd.read_csv(spots_path, sep='\t', float_precision='round_trip',
                        dtype={'spot_id': str, 'timepoint': str})
    if (not spots.spot_id.is_unique or
            spots.timepoint.value_counts().to_dict() != panel['spots_per_timepoint']):
        raise ValueError('Spot order or timepoint counts differ from the panel')

    raw = ad.read_h5ad(raw_path)
    if not raw.obs_names.is_unique or not raw.var_names.is_unique:
        raise ValueError('Raw spot and gene names must be unique')
    if not set(spots.spot_id).issubset(raw.obs_names) or not set(genes).issubset(raw.var_names):
        raise ValueError('A selected spot or gene is absent from the raw counts')
    selected = raw[spots.spot_id.tolist(), genes].copy()
    selected.obs['timepoint'] = spots.timepoint.to_numpy()
    selected.obsm['spatial'] = spots[['spatial_x', 'spatial_y']].to_numpy(dtype=np.float64)

    parts = []
    for tp in quotas:
        part = selected[selected.obs.timepoint == tp].copy()
        part.layers['counts'] = part.X.copy()
        sc.pp.normalize_total(part, target_sum=1e-4)
        sc.pp.log1p(part)
        sc.pp.scale(part)
        parts.append(part)
    result = ad.concat(parts, merge='same')[spots.spot_id.tolist(), genes].copy()
    if not np.isfinite(result.X).all():
        raise ValueError('Preprocessed expression contains non-finite values')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path)
    return output_path
