import ot
import sklearn
import sklearn.metrics
import numpy as np
import torch
from copy import deepcopy
import time
import logging

logger = logging.getLogger(__name__)

log_step = False

debug = False
debug_p1 = False

default_config = {
    "growth_iters": 3,
    "epsilon": 0.05,
    "lambda1": 1,
    "lambda2": 50,
    "epsilon0": 1,
    "tau": 1000,
    "scaling_iter": 3000,
    "inner_iter_max": 50,
    "tolerance": 1e-8,
    "max_iter": 1e7,
    "batch_size": 5,
    "extra_iter": 1000,
    "numItermax": 1000000,
    "use_Py": True,
    "use_C": False,
    "profiling": False
}  # specified in tutorial


def solve_ot(feats, ot_solver, ot_config, gammas, days, g_est):
    for i in range(len(feats) - 1):
        print("Solving OT for days:", days[i], days[i + 1])
        delta_days = float(days[i + 1]) - float(days[i])
        g = np.power(g_est[i], delta_days)
        gamma = ot_solver(feats[i], feats[i + 1], ot_config, G=g)
        print("Transport matrix shape: ", gamma.shape)
        gammas[f"{i}_{i + 1}"] = gamma


def get_total_ot_loss(feats, indices, gammas):
    # feats (bsz, days, dim)
    days = feats.shape[1]
    alignment_loss = 0
    for i in range(days - 1):
        # for j in range(i + 1, len(feats)):
        feat1 = feats[:, i]
        feat2 = feats[:, i + 1]
        index1 = indices[:, i].numpy()
        index2 = indices[:, i + 1].numpy()
        gamma = gammas[f"{i}_{i+1}"]  # use this to compute alignment loss
        # sample gamma matrix with index here
        gamma = gamma[index1][:, index2]  # sample submatrix of gamma with rna index
        gamma = gamma / gamma.sum(axis=1, keepdims=True)  # normalize rows
        gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)  # prune invalid values
        gamma = torch.tensor(gamma).cuda().float()
        cost_matrix = torch.cdist(feat1, feat2, p=2)
        transport_cost = torch.mean(gamma * cost_matrix)
        alignment_loss += transport_cost
    alignment_loss /= days - 1
    return alignment_loss


def compute_transport_map_pot(a, b, config, C=None):
    print("Using OT ", config["method"])
    if not isinstance(a, np.ndarray):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
    if C is None:
        C = sklearn.metrics.pairwise.pairwise_distances(a, b, metric="sqeuclidean", n_jobs=-1)
    C = np.ascontiguousarray(C)
    if config["method"] == "emd":
        gamma = ot.emd(ot.unif(a.shape[0]), ot.unif(b.shape[0]), C, numItermax=config["numItermax"])
    elif config["method"] == "sinkhorn":
        gamma = ot.sinkhorn(ot.unif(a.shape[0]), ot.unif(b.shape[0]), C, reg=config["epsilon"])
    elif config["method"] == "unbalanced":
        gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(
            ot.unif(a.shape[0]),
            ot.unif(b.shape[0]),
            C,
            reg=config["epsilon"],
            reg_m=config["lambda"],
        )  # use wad-OT setting
    return gamma


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
    use_Py,
    use_C,
    profiling,
    **ignored,
):
    """Compute the stabilized OT map using the evaluated NumPy solver.

    The legacy flags remain accepted for config compatibility. The bundled
    package supports only ``use_Py=True, use_C=False``; requesting the removed
    C backend raises an explicit error before any solver work begins.
    """
    if use_C or not use_Py:
        raise ValueError("This SpaDOT build requires use_Py=True and use_C=False")

    cnt_step0 = 0
    cnt_step1 = 0
    cnt_step2 = 0
    time_i0 = time.time()

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
        cnt_step0 += time_k5 - time_k0

        while duality_gap > threshold:
            if debug:
                print('  {} - duality_gap {}'.format(current_iter, duality_gap))
            if debug_p1:
                print('pre', a.sum(), b.sum())
                print('a', a.sum())
                print('b', b.sum())

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

                if debug_p1:
                    print('a', a.sum())
                    print('b', b.sum())

                # Numerical stabilization is identical to the previous
                # use_Py=True, use_C=False branch.
                if max(max(abs(a)), max(abs(b))) > tau:
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)
                    if debug_p1:
                        print('stabilization')

                if current_iter >= max_iter:
                    logger.warning("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            time_s2 = time.time()
            if debug_p1:
                print('post', a.sum(), b.sum())
                print('old ab', old_a.sum(), old_b.sum())
                print('uv', u.sum(), v.sum())
            if profiling:
                print('step1 {:3f} ms'.format(1000 * (time_s2 - time_s1)))
            cnt_step1 += time_s2 - time_s1

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
            cnt_step2 += time_s4 - time_s3
            if debug_p1:
                print('last', a.sum(), b.sum())

    time_i1 = time.time()
    if log_step:
        print('---------------------------------------------------------')
        print('Step0 {:5f} ms'.format(1000 * cnt_step0))
        print('Step1 {:5f} ms'.format(1000 * cnt_step1))
        print('Step2 {:5f} ms'.format(1000 * cnt_step2))
        print('Total {:5f} ms'.format(1000 * (time_i1 - time_i0)))
        print('DualityGap', duality_gap, 'R.sum', R.sum())
        print('---------------------------------------------------------')

    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]


def transport_stablev2(
    C,
    lambda1,
    lambda2,
    epsilon,
    scaling_iter,
    G,
    tau,
    epsilon0,
    extra_iter,
    inner_iter_max,
    **ignored,
):
    """
    Compute the optimal transport with stabilized numerics.
    Args:
        C: cost matrix to transport cell i to cell j
        lambda1: regularization parameter for marginal constraint for p.
        lambda2: regularization parameter for marginal constraint for q.
        epsilon: entropy parameter
        scaling_iter: number of scaling iterations
        G: growth value for input cells
    """

    warm_start = tau is not None
    epsilon_final = epsilon

    def get_reg(n):  # exponential decreasing
        return (epsilon0 - epsilon_final) * np.exp(-n) + epsilon_final

    epsilon_i = epsilon0 if warm_start else epsilon
    dx = np.ones(C.shape[0]) / C.shape[0]
    dy = np.ones(C.shape[1]) / C.shape[1]

    p = G
    q = np.ones(C.shape[1]) * np.average(G)

    u = np.zeros(len(p))
    v = np.zeros(len(q))
    b = np.ones(len(q))
    K = np.exp(-C / epsilon_i)

    alpha1 = lambda1 / (lambda1 + epsilon_i)
    alpha2 = lambda2 / (lambda2 + epsilon_i)
    epsilon_index = 0
    iterations_since_epsilon_adjusted = 0
    eps = 1e-10
    for i in range(scaling_iter):
        # scaling iteration
        a = (p / (K.dot(np.multiply(b, dy)) + eps)) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
        b = (q / (K.T.dot(np.multiply(a, dx)) + eps)) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

        # stabilization
        iterations_since_epsilon_adjusted += 1
        if max(max(abs(a)), max(abs(b))) > tau:
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(p))
            b = np.ones(len(q))

        if warm_start and iterations_since_epsilon_adjusted == inner_iter_max:
            epsilon_index += 1
            iterations_since_epsilon_adjusted = 0
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            epsilon_i = get_reg(epsilon_index)
            alpha1 = lambda1 / (lambda1 + epsilon_i)
            alpha2 = lambda2 / (lambda2 + epsilon_i)
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(p))
            b = np.ones(len(q))

    for i in range(extra_iter):
        a = (p / (K.dot(np.multiply(b, dy)) + eps)) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
        b = (q / (K.T.dot(np.multiply(a, dx)) + eps)) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

    R = (K.T * a).T * b

    return R / C.shape[1]
