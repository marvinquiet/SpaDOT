"""Freeze ranked SPARK-X candidates from the audited historical CSV tables.

This one-time conversion uses Python only. The resulting small text files are
shipped with the examples; preprocessing never needs the CSV tables or R.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def export(panel_dir, tables_dir):
    manifest = panel_dir / 'panel.json'
    panel = json.loads(manifest.read_text())
    params = panel['selection_parameters']
    rank_hashes = {}
    prefix_hashes = {}
    for tp, pool in params['candidate_pool_per_timepoint'].items():
        source = tables_dir / panel['dataset'] / tp / 'R_SPARKX_all_retained.csv'
        if sha256(source) != params['sparkx_table_sha256'][tp]:
            raise ValueError(f'Historical SPARK-X table hash mismatch: {tp}')
        table = pd.read_csv(source)
        if (not table.gene.is_unique or
                not np.isfinite(table[['adjustedPval', 'combinedPval']]).all().all()):
            raise ValueError(f'Invalid SPARK-X table: {tp}')
        ranked = table.sort_values(
            ['adjustedPval', 'combinedPval', 'gene'], kind='mergesort')
        candidates = ranked.loc[ranked.adjustedPval <= .05, 'gene'].head(pool)
        if len(candidates) != pool:
            raise ValueError(f'Insufficient FDR-significant candidates: {tp}')
        target = panel_dir / f'sparkx_candidates_{tp}.txt'
        target.write_text(''.join(gene + '\n' for gene in candidates))
        rank_hashes[tp] = sha256(target)
        prefix = panel_dir / f'prefix_{tp}.txt'
        if prefix.exists():
            prefix_hashes[tp] = sha256(prefix)
    params['sparkx_candidate_ranks_sha256'] = rank_hashes
    params['prefix_sha256'] = prefix_hashes
    manifest.write_text(json.dumps(panel, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels-dir', type=Path, required=True)
    parser.add_argument('--tables-dir', type=Path, required=True)
    args = parser.parse_args()
    for panel_dir in sorted(args.panels_dir.iterdir()):
        if (panel_dir / 'panel.json').exists():
            export(panel_dir, args.tables_dir)
            print(panel_dir.name)
