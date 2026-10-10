import logging
import time
from copy import deepcopy

import numpy as np
import sklearn.metrics

logger = logging.getLogger(__name__)

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


def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)

    t1 = -F1c(-epsilon * np.log(a), dx)
    t2 = -F2c(-epsilon * np.log(b), dy)
    t3 = -epsilon * np.sum(R - K) / (I * J)

    #print("py", I, J, lambda1, lambda2, t1, t2, t3)
    #return -F1c(-epsilon * np.log(a), dx) - F2c(-epsilon * np.log(b), dy) - epsilon * np.sum(R - K) / (I * J)

    return t1 + t2 + t3


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
    profiling,
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
                    logger.warning("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            time_s2 = time.time()
            if profiling:
                print('step1 {:3f} ms'.format(1000 * (time_s2 - time_s1)))

            time_s3 = time.time()
            # The real dual variables; a and b are only stabilized variables.
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)

            # Before the final epsilon scale, use dual-variable evolution.
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)),
                )

            time_s4 = time.time()
            if profiling:
                print('step2 {:3f} ms'.format(1000 * (time_s4 - time_s3)))

    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]
