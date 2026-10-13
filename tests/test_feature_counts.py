"""Input checks for per-timepoint Python SVG selection."""

import json
from argparse import Namespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from SpaDOT.utils import _preprocess_utils as prep
from SpaDOT.utils._preprocess_utils import (
    _feature_counts, _ordered_timepoints, _preprocess_with_feature_counts,
    _selection_plan,
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
    args = Namespace(feature_selection=True, selection_plan=None,
                     n_features='500,500,1000', output_dir=str(tmp_path))
    with pytest.raises(ValueError, match='raw gene count'):
        _preprocess_with_feature_counts(args, raw)


def test_numeric_plan_requires_matching_timepoints_and_final_quotas(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'timepoints': {
        't0': {'pca_pool': 1100, 'stages': [[100, 1200]]},
        't1': {'pca_pool': 1100, 'stages': [[100, 1200]]},
    }}))
    assert _selection_plan(path, ['t0', 't1'], {'t0': 100, 't1': 100})['t0'][
        'stages'] == [[100, 1200]]
    with pytest.raises(ValueError, match='observed timepoints'):
        _selection_plan(path, ['t0', 't2'], {'t0': 100, 't2': 100})
    with pytest.raises(ValueError, match='final stage'):
        _selection_plan(path, ['t0', 't1'], {'t0': 100, 't1': 200})


@pytest.mark.parametrize('labels,expected', [
    (['t0', 't1', 't0', 't1'], ['a', 'c', 'b', 'd']),
    (['t0', 't0', 't1', 't1'], ['a', 'b', 'c', 'd']),
])
def test_preprocess_preserves_within_slice_order(tmp_path, monkeypatch,
                                                 labels, expected):
    names = [f'g{i:03d}' for i in range(100)]
    raw = ad.AnnData(
        X=(np.arange(400).reshape(4, 100) % 7 + 1).astype(np.int64),
        obs=pd.DataFrame({'timepoint': labels}, index=['a', 'b', 'c', 'd']),
    )
    raw.var_names = names
    raw.obsm['spatial'] = np.arange(8, dtype=float).reshape(4, 2)
    monkeypatch.setattr(prep, '_python_pseudoregion_panel',
                        lambda *args, **kwargs: (
                            {'t0': names, 't1': names}, {'method': 'test'}))
    args = Namespace(feature_selection=True, selection_plan=None,
                     n_features='100,100', output_dir=str(tmp_path))
    actual = _preprocess_with_feature_counts(args, raw)
    assert actual.obs_names.tolist() == expected
    assert actual.var_names.tolist() == names
    np.testing.assert_array_equal(actual.layers['counts'],
                                  raw[expected, names].X)
    assert np.isfinite(actual.X).all()
