"""Numerical and convergence regressions for the NumPy OT solver."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

path = Path(__file__).resolve().parents[1] / 'SpaDOT/utils/OT_loss/ot_solvers.py'
spec = importlib.util.spec_from_file_location('ot_log_stability_under_test', path)
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def solve(cost, **kwargs):
    config = dict(solver.default_config, epsilon=.01, lambda1=.1, lambda2=5,
                  max_iter=10000)
    config.update(kwargs)
    with threadpool_limits(limits=1):
        return solver.optimal_transport_duality_gap(
            C=np.asarray(cost, dtype=float), G=np.ones(len(cost)), **config)


def test_dual_logs_above_float64_exponential_limit():
    log_a = np.array([0., 0.])
    log_b = np.array([732., 735., 730.])
    R = K = np.ones((2, 3))
    dx, dy = np.ones(2)/2, np.ones(3)/3
    with np.errstate(over='raise', invalid='raise'):
        actual = solver.dual(None, K, R, dx, dy, np.ones(2), np.ones(3),
                             log_a, log_b, .01, .1, 5, log_scalings=True)
    # Analytic KL-conjugate value for zero source potential and R=K.
    expected = 5 * np.mean(1-np.exp(-.01*log_b/5))
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_relative_change_handles_unrepresentable_scalings():
    with np.errstate(over='raise', invalid='raise'):
        actual = solver._relative_change_from_logs(
            np.log(np.array([1., 2.])) + 1000,
            np.log(np.array([.99, 1.98])) + 1000)
    np.testing.assert_allclose(actual, .01, rtol=1e-11)


def test_zero_cost_zero_objective_is_valid():
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        actual = solve(np.zeros((2, 3)))
    np.testing.assert_array_equal(actual, np.ones((2, 3))/3)


def test_constant_cost_matches_analytic_unbalanced_solution():
    cost = .75
    actual = solve(np.full((2, 3), cost))
    expected = np.exp(-cost/(.1+5+.01))/3
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_negative_gap_does_not_stop_early(monkeypatch):
    original = solver.dual
    calls = []
    def first_invalid_dual(*args, **kwargs):
        calls.append(1)
        if len(calls)==1:
            primal = solver.primal(*args)
            return primal + max(abs(primal), 1)*.001
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, 'dual', first_invalid_dual)
    result = solve(np.array([[.1, 1., 2.], [2., .2, .7]]))
    assert len(calls)>1 and np.isfinite(result).all()


def test_nonfinite_objective_is_rejected(monkeypatch):
    monkeypatch.setattr(solver, 'dual', lambda *a, **kw: np.nan)
    with pytest.raises(FloatingPointError, match='Non-finite OT primal or dual'):
        solve(np.full((2, 3), .75))


def test_iteration_limit_does_not_return_unverified_map():
    with pytest.raises(RuntimeError, match='max_iter'):
        solve(np.array([[.1, 1., 2.], [2., .2, .7]]), max_iter=1)


def test_analyze_routes_wot_through_fixed_solver(tmp_path, monkeypatch):
    ad = pytest.importorskip('anndata')
    import pandas as pd
    from types import SimpleNamespace
    from SpaDOT.utils import _analyze_utils as analysis
    real_factory = analysis.wot.ot.OTModel
    def forbid_old_solver(*a, **kw):
        raise AssertionError('Analysis fell back to the unfixed WOT solver')
    def tracked_factory(*a, **kw):
        assert kw["epsilon"] == .01
        model = real_factory(*a, **kw)
        model.solver = forbid_old_solver
        return model
    monkeypatch.setattr(analysis.wot.ot, 'OTModel', tracked_factory)
    data = ad.AnnData(
        X=np.array([[0.,0.], [.2,.1], [1.,1.], [1.2,.8],
                    [.1,.2], [.3,.2], [1.1,.9], [1.3,1.1]]),
        obs=pd.DataFrame({'timepoint':['t0']*4+['t1']*4,
                          'kmeans':['0','0','1','1']*2},
                         index=[str(i) for i in range(8)]))
    with threadpool_limits(limits=1):
        analysis.OT_analysis(SimpleNamespace(output_dir=str(tmp_path), prefix=''), data)
    result = ad.read_h5ad(tmp_path/'transition_table_0_1.h5ad')
    assert result.shape==(2,2) and np.isfinite(result.X).all() and (result.X>=0).all()
