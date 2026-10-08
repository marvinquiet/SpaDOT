"""Protect the evaluated NumPy OT path after removing the bundled C backend."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from SpaDOT.utils.OT_loss.ot_solvers import compute_transport_map, default_config


def test_python_ot_matches_pre_removal_reference():
    # Reference is the old use_Py=True/use_C=False implementation on the same
    # deterministic small cost geometry, recorded before removing libot.so.
    source = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    target = np.array([[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]])
    expected = np.array([
        [1.2909709828378142, 7.014028617181969e-10, 3.4010818068330925e-13],
        [7.014028617181989e-10, 1.2909709828378138, 3.4010818068330915e-13],
        [0.03200735769399827, 1.7390051814533135e-11, 0.6658460713767859],
        [1.739005181453313e-11, 0.03200735769399824, 0.6658460713767859],
    ])
    assert "use_C" not in default_config
    assert "use_Py" not in default_config
    config = dict(default_config, growth_iters=1, max_iter=100000)
    with threadpool_limits(limits=1):
        actual = compute_transport_map(source, target, config)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_legacy_python_flags_remain_compatible():
    source = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    target = np.array([[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]])
    with threadpool_limits(limits=1):
        current = compute_transport_map(
            source, target, dict(default_config, growth_iters=1))
        legacy = compute_transport_map(
            source, target, dict(default_config, growth_iters=1,
                                 use_Py=True, use_C=False))
    np.testing.assert_array_equal(current, legacy)


@pytest.mark.parametrize("flags", [{"use_C": True}, {"use_Py": False}])
def test_removed_backends_have_clear_error(flags):
    source = np.array([[0., 0.], [1., 1.]])
    config = dict(default_config, growth_iters=1, **flags)
    with pytest.raises(ValueError, match="Only the Python OT backend"):
        compute_transport_map(source, source + 0.5, config)
