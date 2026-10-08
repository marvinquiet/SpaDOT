"""Input checks for per-timepoint Python SVG selection."""

import json
from argparse import Namespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from SpaDOT.utils._preprocess_utils import (
    _feature_counts, _ordered_timepoints, _preprocess_with_feature_counts, _sha256,
)


def test_feature_quotas_follow_natural_timepoint_order():
    raw = ad.AnnData(
        X=np.ones((3, 200)),
        obs=pd.DataFrame({'timepoint': ['t2', 't0', 't1']},
                         index=['a', 'b', 'c']),
    )
    timepoints = _ordered_timepoints(raw)
    assert timepoints == ['t0', 't1', 't2']
    assert _feature_counts('500,500,1000', timepoints) == {
        't0': 500, 't1': 500, 't2': 1000,
    }


@pytest.mark.parametrize('spec', ['500,1000', '500,500,1000,1000',
                                   '500,one,1000', '500,0,1000',
                                   '500,99,1000'])
def test_feature_quota_errors_are_explicit(spec):
    with pytest.raises(ValueError, match='--n_features'):
        _feature_counts(spec, ['t0', 't1', 't2'])


def test_oversized_quota_rejected_before_sparkx(tmp_path):
    raw = ad.AnnData(X=np.ones((3, 200), dtype=np.int64),
                     obs=pd.DataFrame({'timepoint': ['t2', 't0', 't1']},
                                      index=['a', 'b', 'c']))
    args = Namespace(feature_selection=True, feature_panel=None,
                     n_features='500,500,1000', output_dir=str(tmp_path))
    with pytest.raises(ValueError, match='raw gene count'):
        _preprocess_with_feature_counts(args, raw)


def test_frozen_panel_can_use_different_validated_timepoint_labels(tmp_path):
    names = [f'g{i:03d}' for i in range(100)]
    raw = ad.AnnData(
        X=(np.arange(400).reshape(4, 100) % 7 + 1).astype(np.int64),
        obs=pd.DataFrame({'timepoint': ['t0', 't0', 't1', 't1']},
                         index=['a', 'b', 'c', 'd']),
    )
    raw.var_names = names
    raw.obsm['spatial'] = np.arange(8, dtype=float).reshape(4, 2)
    raw_file = tmp_path / 'raw.h5ad'
    raw.write_h5ad(raw_file)
    panel_dir = tmp_path / 'panel'
    panel_dir.mkdir()
    spots = pd.DataFrame({
        'spot_id': ['b', 'a', 'd', 'c'], 'timepoint': ['0', '0', '1', '1'],
        'spatial_x': [2., 0., 6., 4.], 'spatial_y': [3., 1., 7., 5.],
    })
    spots_file = panel_dir / 'spots.tsv'
    spots.to_csv(spots_file, sep='\t', index=False)
    union = panel_dir / 'union_genes.txt'
    union.write_text(''.join(gene + '\n' for gene in names))
    for tp in ['0', '1']:
        (panel_dir / f'genes_{tp}.txt').write_bytes(union.read_bytes())
    (panel_dir / 'panel.json').write_text(json.dumps({
        'raw_sha256': _sha256(raw_file),
        'selected_genes_per_timepoint': {'0': 100, '1': 100},
        'spots_per_timepoint': {'0': 2, '1': 2},
        'spot_metadata_sha256': _sha256(spots_file),
        'union_genes_sha256': _sha256(union),
    }))
    args = Namespace(data=str(raw_file), feature_selection=True,
                     feature_panel=str(panel_dir), n_features='100,100',
                     output_dir=str(tmp_path))
    actual = _preprocess_with_feature_counts(args, raw)
    assert actual.obs_names.tolist() == ['b', 'a', 'd', 'c']
    assert actual.obs.timepoint.tolist() == ['0', '0', '1', '1']
    assert actual.var_names.tolist() == names
    assert np.isfinite(actual.X).all()
