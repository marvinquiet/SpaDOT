import time
from copy import deepcopy

import numpy as np
import sklearn.metrics

default_config = {
    "growth_iters": 3,
    "epsilon": 0.05,
    "lambda1": 1,
    "lambda2": 50,
    "epsilon0": 1,
    "tau": 1000,
    "tolerance": 1e-8,
    "max_iter": 1e7,
    "batch_size": 5,
    "profiling": False
}


def compute_transport_map(a, b, config, C=None, G=None):
    if not isinstance(a, np.ndarray):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()

    # Compute cost matrix
    if C is None:
        C = sklearn.metrics.pairwise.pairwise_distances(a, b, metric="sqeuclidean", n_jobs=1)
        C = C / np.median(C)

    config["C"] = C  # cost matrix
    if G is None:
        config["G"] = np.ones(C.shape[0])
    else:
        config["G"] = G
    growth_iters = config["growth_iters"]
    gammas = []
    for i in range(growth_iters):
        print("OT iter", i)
        if i == 0:
            row_sums = config["G"]
        else:
            row_sums = gamma.sum(axis=1)  # / tmap.shape[1]
        config["G"] = row_sums
        gamma = optimal_transport_duality_gap(**config)
        gammas.append(deepcopy(gamma))
    return gammas[0]


def fdiv(l, x, p, dx):
    return l * np.sum(dx * (x * (np.log(x / p)) - x + p))


def fdivstar(l, u, p, dx):
    return l * np.sum((p * dx) * (np.exp(u / l) - 1))


def primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide="ignore"):
        return (
            F1(np.dot(R, dy), dx)
            + F2(np.dot(R.T, dx), dy)
            + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) + np.sum(R * C)) / (I * J)
        )


def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2, *, log_scalings=False):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)

    log_a = a if log_scalings else np.log(a)
    log_b = b if log_scalings else np.log(b)
    t1 = -F1c(-epsilon * log_a, dx)
    t2 = -F2c(-epsilon * log_b, dy)
    t3 = -epsilon * np.sum(R - K) / (I * J)

    return t1 + t2 + t3


def _relative_change_from_logs(current, previous):
    """Compute ||x-y||/(1+||x||) without exponentiating large log scalings."""
    shift = max(0.0, float(np.max(current)), float(np.max(previous)))
    x = np.exp(current - shift)
    y = np.exp(previous - shift)
    return np.linalg.norm(x - y) / (np.exp(-shift) + np.linalg.norm(x))


# end @ Lénaïc Chizat


def optimal_transport_duality_gap(
    C,
    G,
    lambda1,
    lambda2,
    epsilon,
    batch_size,
    tolerance,
    tau,
    epsilon0,
    max_iter,
    profiling=False,
    **ignored,
):
    """Compute the stabilized OT map using the evaluated NumPy solver.

    Older configs with ``use_Py=True, use_C=False`` still work. A request
    for another backend fails before the solver starts.
    """
    if ignored.get("use_C", False) or not ignored.get("use_Py", True):
        raise ValueError("Only the Python OT backend is available")

    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(-np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J
    p = G
    q = np.ones(C.shape[1]) * np.average(G)
    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)

    dtype = np.float64
    C = np.ascontiguousarray(C, dtype=dtype)
    dx = np.ascontiguousarray(dx, dtype=dtype)
    dy = np.ascontiguousarray(dy, dtype=dtype)
    p = np.ascontiguousarray(p, dtype=dtype)
    q = np.ascontiguousarray(q, dtype=dtype)
    u = np.ascontiguousarray(u, dtype=dtype)
    v = np.ascontiguousarray(v, dtype=dtype)
    a = np.ascontiguousarray(a, dtype=dtype)
    b = np.ascontiguousarray(b, dtype=dtype)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    for e in range(epsilon_scalings + 1):
        if profiling:
            print('Compute for epsilon scaling {}'.format(e))

        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        a, b = np.ones(I), np.ones(J)
        epsilon_i = epsilon_i / scale_factor
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        old_a = a.copy()
        old_b = b.copy()
        threshold = tolerance if e == epsilon_scalings else 1e-6

        time_k0 = time.time()
        _K = np.exp(-C / epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        time_k5 = time.time()
        R = np.zeros(K.shape)
        if profiling:
            print('step0 {:3f} ms'.format(1000 * (time_k5 - time_k0)))

        while duality_gap > threshold:
            time_s1 = time.time()
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b

                if profiling:
                    time_h1 = time.time()
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                if profiling:
                    time_h2 = time.time()
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))
                if profiling:
                    time_h3 = time.time()
                    print('step1 {:3f} ms'.format(1000 * (time_h2 - time_h1)))
                    print('step1 {:3f} ms'.format(1000 * (time_h3 - time_h2)))

                # Preserve the evaluated NumPy stabilization step.
                if max(max(abs(a)), max(abs(b))) > tau:
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)

                if current_iter >= max_iter:
                    raise RuntimeError("OT did not converge before max_iter was reached")

            time_s2 = time.time()
            if profiling:
                print('step1 {:3f} ms'.format(1000 * (time_s2 - time_s1)))

            time_s3 = time.time()
            # Keep unscaled dual variables in log form, including diagnostics.
            log_a = np.log(a) + u / epsilon_i
            log_b = np.log(b) + v / epsilon_i

            # Before the final epsilon scale, use dual-variable evolution.
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                # The primal expression does not depend on its a/b arguments.
                pri = primal(C, _K, R, dx, dy, p, q, a, b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, log_a, log_b, epsilon_i, lambda1, lambda2, log_scalings=True)
                if not np.isfinite(pri) or not np.isfinite(dua):
                    raise FloatingPointError("Non-finite OT primal or dual objective")
                duality_gap = abs(pri - dua) / max(abs(pri), np.finfo(np.float64).tiny)
            else:
                duality_gap = max(
                    _relative_change_from_logs(log_a, np.log(old_a) + u / epsilon_i),
                    _relative_change_from_logs(log_b, np.log(old_b) + v / epsilon_i),
                )
            if not np.isfinite(duality_gap):
                raise FloatingPointError("Non-finite OT convergence diagnostic")

            time_s4 = time.time()
            if profiling:
                print('step2 {:3f} ms'.format(1000 * (time_s4 - time_s3)))

    if not np.isfinite(R).all() or np.any(R < 0):
        raise FloatingPointError("OT returned an invalid transport map")
    return R / C.shape[1]
