"""Regression checks for the explicit 30-epoch KL period."""
import numpy as np
import pytest

from SpaDOT.utils._train_utils import _beta_cycle_linear, _beta_cycle_linear_period, _epoch_plan


def test_first_100_weights_match_the_validated_300_epoch_schedule():
    actual = _beta_cycle_linear_period(100, 30, stop=1.0)
    reference = _beta_cycle_linear(300, stop=1.0)[:100]
    assert actual.shape == (100,)
    np.testing.assert_array_equal(actual, reference)
    assert actual[0] == actual[30] == actual[60] == actual[90] == 0.0
    assert actual[99] == reference[99]


def test_epochs_one_through_ninety_nine_keep_original_update_phases():
    plan = _epoch_plan({"epoch_start": 1, "epoch_stop_exclusive": 100})
    assert list(plan) == list(range(1, 100))
    assert [(epoch - plan.start) for epoch in plan] == list(range(99))
    np.testing.assert_array_equal(
        _beta_cycle_linear_period(len(plan), 30),
        _beta_cycle_linear(300)[:99],
    )


@pytest.mark.parametrize('bounds', [
    {"epoch_start": 1, "epoch_stop_exclusive": 1},
    {"epoch_start": 100, "epoch_stop_exclusive": 1},
    {"epoch_start": 1},
])
def test_invalid_epoch_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        _epoch_plan(bounds)


def test_period_is_independent_of_training_horizon():
    np.testing.assert_array_equal(
        _beta_cycle_linear_period(100, 30),
        _beta_cycle_linear_period(120, 30)[:100],
    )
    np.testing.assert_array_equal(
        _beta_cycle_linear_period(100, 10),
        _beta_cycle_linear(100),
    )


@pytest.mark.parametrize('n_iter,period', [(0, 30), (100, 0), (100, -1), (100, 3.5)])
def test_invalid_period_or_horizon(n_iter, period):
    with pytest.raises(ValueError):
        _beta_cycle_linear_period(n_iter, period)
