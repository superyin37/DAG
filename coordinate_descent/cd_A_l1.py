"""Coordinate descent for DAG learning with exact L1 penalty.

Mirror of `coordinate0.dag_coordinate_descent_l0` but with an L1 penalty
``lambda_l1 * |A_ij|`` on each off-diagonal entry. The one-coordinate
subproblem

    g(delta) = f(A + delta * E_ij) + lambda_l1 * |delta|

is solved exactly in closed form (path B in the design notes): for each
sign branch (delta > 0 / delta < 0) we solve a quadratic in delta, keep the
sign-consistent root, then pick the branch (or zero) that minimizes g.

Optional weak-faithfulness screening from `cd_A_weakfaith` is supported
via the same parameters (`faithfulness_tau`, `screening`, `combine`,
`glasso_alpha`, `sampling_mode`); set `faithfulness_tau=0` to disable.

Algebra reference
-----------------
Let alpha = A_inv[j, i], b = S[i, :] @ A[:, j], c = S[i, i]. Then

    f(A + delta * E_ij) - f(A) = -2 log(1 + delta*alpha) + 2*delta*b + delta^2*c.

The KKT condition g'(delta) = 0 with sigma = sign(delta) becomes

    c*alpha * delta^2 + (c + alpha*b_sigma) * delta + (b_sigma - alpha) = 0,

where b_sigma = b + sigma * lambda_l1 / 2. Each branch contributes at
most one sign-consistent interior root (g is convex on each sign half-line);
delta = 0 is the soft-threshold dead-zone candidate.
"""

import numpy as np

try:
    from .coordinate0 import (
        update_diagonal,
        f,
        weight_to_adjacency,
        is_DAG,
        _sm_update_A_inv,
        _graph_snapshot,
    )
    from .cd_A_weakfaith import _build_faithfulness_mask
except ImportError:
    from coordinate0 import (
        update_diagonal,
        f,
        weight_to_adjacency,
        is_DAG,
        _sm_update_A_inv,
        _graph_snapshot,
    )
    from cd_A_weakfaith import _build_faithfulness_mask


def _g_rel(delta, alpha, b, c, lambda_l1):
    """Value of g(delta) - g(0) (i.e. with the constant f(A) dropped).

    Returns ``+inf`` when the log-det domain ``1 + delta*alpha > 0`` is
    violated, so an out-of-domain candidate never wins the argmin.
    """
    arg = 1.0 + delta * alpha
    if arg <= 0.0:
        return np.inf
    return (-2.0 * np.log(arg)
            + 2.0 * delta * b
            + delta * delta * c
            + lambda_l1 * abs(delta))


def delta_star_l1(A, S, i, j, lambda_l1, A_inv=None, eps=1e-12):
    """Closed-form minimizer of ``f(A + delta * E_ij) + lambda_l1 * |delta|``.

    Parameters mirror `coordinate0.delta_star`. Returns 0.0 when the
    soft-threshold dead zone is optimal or when the problem is degenerate
    (no real sign-consistent root in either branch).
    """
    if A_inv is None:
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return 0.0

    c = float(S[i, i])
    if c <= eps:
        return 0.0
    b = float(S[i, :] @ A[:, j])
    alpha = float(A_inv[j, i])
    half = 0.5 * lambda_l1

    # Always include 0 as a candidate (the soft-threshold dead zone).
    best_delta = 0.0
    best_g = 0.0  # g_rel(0) = 0

    for sigma in (+1.0, -1.0):
        b_sigma = b + sigma * half

        # Linear degenerate case: alpha = 0 collapses the quadratic.
        if abs(alpha) < eps:
            delta = -b_sigma / c
            if delta * sigma > 0.0:
                g = _g_rel(delta, alpha, b, c, lambda_l1)
                if g < best_g:
                    best_g, best_delta = g, delta
            continue

        A_quad = c * alpha
        B_quad = c + alpha * b_sigma
        C_quad = b_sigma - alpha
        disc = B_quad * B_quad - 4.0 * A_quad * C_quad
        if disc < 0.0:
            continue
        sqrt_disc = np.sqrt(disc)

        # Both roots; sign-consistency + log-det domain filter.
        denom = 2.0 * A_quad
        for sign_root in (+1.0, -1.0):
            delta = (-B_quad + sign_root * sqrt_disc) / denom
            if delta * sigma <= 0.0:
                continue
            if 1.0 + delta * alpha <= eps:
                continue
            g = _g_rel(delta, alpha, b, c, lambda_l1)
            if g < best_g:
                best_g, best_delta = g, delta

    return best_delta


def update_off_diagonal_l1(A, S, i, j, lambda_l1=0.2, A_inv=None):
    """One off-diagonal coordinate update under the L1 penalty.

    Same protocol as `coordinate0.update_off_diagonal`: zero both A[i,j]
    and A[j,i], try each direction (subject to the DAG constraint),
    compute the L1-optimal delta in each, and accept the direction whose
    objective drop is largest. The "stay at zero" outcome is built into
    `delta_star_l1` (it returns 0 when the soft-threshold zone wins).
    """
    d = S.shape[0]
    old_aij = A[i, j]
    old_aji = A[j, i]
    A[i, j] = A[j, i] = 0.0

    if A_inv is not None:
        ok = True
        if old_aij != 0.0:
            ok = _sm_update_A_inv(A_inv, i, j, -old_aij)
        if ok and old_aji != 0.0:
            ok = _sm_update_A_inv(A_inv, j, i, -old_aji)
        if not ok:
            A_inv[:] = np.linalg.inv(A)

    Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
    Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]

    Δ_ij, Δ_ji = -np.inf, -np.inf
    δ_t, δ_bar = 0.0, 0.0

    if is_DAG(A + Eij):
        δ_t = delta_star_l1(A, S, i, j, lambda_l1, A_inv=A_inv)
        Δ_ij = f(A, S) - f(A + δ_t * Eij, S) - lambda_l1 * abs(δ_t)

    if is_DAG(A + Eji):
        δ_bar = delta_star_l1(A, S, j, i, lambda_l1, A_inv=A_inv)
        Δ_ji = f(A, S) - f(A + δ_bar * Eji, S) - lambda_l1 * abs(δ_bar)

    if Δ_ij == -np.inf and Δ_ji == -np.inf:
        return A

    # Both Delta values are >= 0 by construction (delta_star_l1 already
    # compares the optimum against staying at 0). Pick the direction with
    # the larger objective drop; ties favor i->j.
    if Δ_ij >= Δ_ji:
        if δ_t != 0.0:
            if A_inv is not None:
                if not _sm_update_A_inv(A_inv, i, j, δ_t):
                    A_inv[:] = np.linalg.inv(A + δ_t * Eij)
            A = A + δ_t * Eij
    else:
        if δ_bar != 0.0:
            if A_inv is not None:
                if not _sm_update_A_inv(A_inv, j, i, δ_bar):
                    A_inv[:] = np.linalg.inv(A + δ_bar * Eji)
            A = A + δ_bar * Eji
    return A


def f_l1(A, S, lambda_l1):
    """L1-penalized objective ``f(A) + lambda_l1 * sum(|A_ij|, i!=j)``."""
    val = f(A, S)
    if lambda_l1 > 0.0:
        off = np.abs(A).sum() - np.abs(np.diag(A)).sum()
        val += lambda_l1 * off
    return val


def dag_coordinate_descent_l1(
    S,
    T=100,
    seed=0,
    threshold=0.05,
    lambda_l1=0.2,
    return_history=False,
    return_graph_history=False,
    A_init=None,
    early_stop=False,
    check_every=None,
    tol=1e-4,
    patience=10,
    min_steps=None,
    # --- weak-faithfulness screening (set faithfulness_tau=0 to disable) ---
    faithfulness_tau=0.0,
    sampling_mode="preserve",
    screening="corr",
    glasso_alpha=0.01,
    combine="union",
):
    """Random coordinate descent for DAG learning with exact L1 penalty.

    Drop-in counterpart of `coordinate0.dag_coordinate_descent_l0` with
    ``lambda_l0`` replaced by an exact L1 penalty ``lambda_l1`` (path B).

    Weak-faithfulness screening is integrated identically to
    `cd_A_weakfaith.dag_coordinate_descent_l0_weakfaith`: setting
    ``faithfulness_tau > 0`` masks out coordinate pairs (i, j) whose
    selected screen statistics are below the threshold.

    Parameters
    ----------
    S, T, seed, threshold, return_history, return_graph_history, A_init,
    early_stop, check_every, tol, patience, min_steps :
        Same as `coordinate0.dag_coordinate_descent_l0`.
    lambda_l1 : float, default 0.2
        L1 penalty strength on each off-diagonal entry.
    faithfulness_tau : float, default 0.0
        Screening threshold; 0 disables (uniform sampling, original RNG sequence).
    sampling_mode : {"preserve", "pool"}, default "preserve"
        Allocation between diagonal and allowed off-diagonal coordinates;
        see the design doc.
    screening : str or sequence of str, default "corr"
        One or more of {"corr", "pcorr", "glasso"}.
    glasso_alpha : float, default 0.01
        L1 strength for the inner Graphical-Lasso when "glasso" is selected.
    combine : {"union", "intersect"}, default "union"
        How to merge forbidden masks across multiple screens.

    The history records the **L1-penalized objective**
    ``f(A) + lambda_l1 * sum(|A|_off-diag)`` after each iteration, so that
    monotone non-increase is visible.
    """
    if sampling_mode not in ("preserve", "pool"):
        raise ValueError(
            f"unknown sampling_mode {sampling_mode!r} "
            f"(expected 'preserve' or 'pool')"
        )

    np.random.seed(seed)
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)
    history = []
    graph_history_list = [] if return_graph_history else None

    A_inv = np.linalg.inv(A)

    allowed_offdiag, M = _build_faithfulness_mask(
        S,
        faithfulness_tau,
        screening,
        glasso_alpha=glasso_alpha,
        combine=combine,
    )

    if early_stop:
        if check_every is None:
            check_every = d * (d + 1) // 2
        if min_steps is None:
            min_steps = check_every * 10
        prev_check_f = f_l1(A, S, lambda_l1)
        no_improve_count = 0

    for t in range(T):
        if allowed_offdiag is None:
            i, j = np.random.choice(d, 2, replace=True)
        elif sampling_mode == "preserve":
            if np.random.rand() < 1.0 / d:
                i = j = int(np.random.randint(d))
            else:
                idx = int(np.random.randint(M))
                i = int(allowed_offdiag[idx, 0])
                j = int(allowed_offdiag[idx, 1])
        else:  # "pool"
            r = int(np.random.randint(d + M))
            if r < d:
                i = j = r
            else:
                i = int(allowed_offdiag[r - d, 0])
                j = int(allowed_offdiag[r - d, 1])

        if i == j:
            A = update_diagonal(A, S, i, A_inv=A_inv)
        else:
            A = update_off_diagonal_l1(A, S, i, j, lambda_l1, A_inv=A_inv)

        history.append(f_l1(A, S, lambda_l1))
        if graph_history_list is not None:
            graph_history_list.append(_graph_snapshot(A, threshold))

        if early_stop and (t + 1) % check_every == 0:
            curr_f = history[-1]
            rel_improve = (prev_check_f - curr_f) / max(1.0, abs(prev_check_f))
            prev_check_f = curr_f
            if t + 1 >= min_steps:
                if rel_improve < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break

    G = weight_to_adjacency(A, threshold)
    final_obj = history[-1] if len(history) > 0 else f_l1(A, S, lambda_l1)

    if graph_history_list is not None:
        graph_history = np.array(graph_history_list, dtype=np.uint8)
    else:
        graph_history = None

    if return_history and return_graph_history:
        return A, G, final_obj, history, graph_history
    if return_history:
        return A, G, final_obj, history
    if return_graph_history:
        return A, G, final_obj, graph_history
    return A, G, final_obj
