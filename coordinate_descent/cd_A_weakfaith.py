"""Random coordinate descent for DAG learning with weak-faithfulness screening.

This module implements `dag_coordinate_descent_l0_weakfaith`, a variant of
`coordinate0.dag_coordinate_descent_l0` (a.k.a. `cd_A_noepoch`) that prunes the
off-diagonal candidate pool using a one-edge faithfulness assumption: if the
marginal correlation (or partial correlation) between x_i and x_j is below a
threshold tau, the coordinate pair (i, j) is excluded from sampling.

See `docs/weak_faithfulness_cd_A_noepoch_zh.md` for the full design rationale.
"""

import numpy as np

try:
    from .coordinate0 import (
        update_diagonal,
        update_off_diagonal,
        f,
        weight_to_adjacency,
        _graph_snapshot,
    )
except ImportError:
    from coordinate0 import (
        update_diagonal,
        update_off_diagonal,
        f,
        weight_to_adjacency,
        _graph_snapshot,
    )


def _build_faithfulness_mask(S, tau, screening):
    """Build the forbidden/allowed masks for one-edge faithfulness screening.

    Parameters
    ----------
    S : (d, d) ndarray
        Sample covariance matrix (or Gram matrix divided by n). Must be symmetric
        with positive diagonal.
    tau : float
        Threshold below which |statistic| is treated as zero. tau <= 0 disables
        screening and returns (None, d*(d-1)).
    screening : {"corr", "pcorr"}
        Statistic used for the independence test.

    Returns
    -------
    allowed_offdiag : (M, 2) ndarray or None
        Row = index pair (i, j) with i != j that survives screening.
        None when tau <= 0.
    M : int
        Number of allowed off-diagonal pairs.
    """
    d = S.shape[0]
    if tau <= 0.0:
        return None, d * (d - 1)

    if screening == "corr":
        std = np.sqrt(np.diag(S))
        stat = S / np.outer(std, std)
    elif screening == "pcorr":
        try:
            Omega = np.linalg.inv(S + 1e-6 * np.eye(d))
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "S is singular even after ridge regularization; "
                "cannot compute pcorr screening."
            ) from e
        d_std = np.sqrt(np.diag(Omega))
        stat = -Omega / np.outer(d_std, d_std)
    else:
        raise ValueError(
            f"unknown screening {screening!r} (expected 'corr' or 'pcorr')"
        )

    forbidden = np.abs(stat) < tau
    np.fill_diagonal(forbidden, False)
    allowed = np.argwhere(~forbidden & ~np.eye(d, dtype=bool))
    M = len(allowed)
    if M == 0:
        raise ValueError(
            f"All off-diagonal pairs masked (tau={tau}, screening={screening!r}); "
            f"try a smaller tau."
        )
    return allowed, M


def dag_coordinate_descent_l0_weakfaith(
    S,
    T=100,
    seed=0,
    threshold=0.05,
    lambda_l0=0.2,
    return_history=False,
    return_graph_history=False,
    A_init=None,
    early_stop=False,
    check_every=None,
    tol=1e-4,
    patience=10,
    min_steps=None,
    faithfulness_tau=0.0,
    sampling_mode="preserve",
    screening="corr",
):
    """Random coordinate descent with one-edge faithfulness screening.

    Semantically equivalent to `coordinate0.dag_coordinate_descent_l0` when
    `faithfulness_tau <= 0`; in that regime the RNG call sequence is identical,
    so results match byte-for-byte under the same seed.

    Parameters
    ----------
    S, T, seed, threshold, lambda_l0, return_history, return_graph_history,
    A_init, early_stop, check_every, tol, patience, min_steps :
        Same as `coordinate0.dag_coordinate_descent_l0`.
    faithfulness_tau : float, default 0.0
        Screening threshold. 0 disables screening and reverts to the original
        uniform (i, j) sampling.
    sampling_mode : {"preserve", "pool"}, default "preserve"
        How to allocate sampling probability between diagonal and off-diagonal
        coordinates when screening is active.

        - "preserve": P(diag) = 1/d (matches original), off-diag uniform over
          the allowed pool.
        - "pool": uniform over {d diagonal coords} ∪ {M allowed off-diag coords}.
    screening : {"corr", "pcorr"}, default "corr"
        Statistic used to judge marginal (corr) or conditional-on-rest (pcorr)
        independence.

    Returns
    -------
    Same tuple shape as `coordinate0.dag_coordinate_descent_l0`.
    """
    if sampling_mode not in ("preserve", "pool"):
        raise ValueError(
            f"unknown sampling_mode {sampling_mode!r} "
            f"(expected 'preserve' or 'pool')"
        )
    if screening not in ("corr", "pcorr"):
        raise ValueError(
            f"unknown screening {screening!r} (expected 'corr' or 'pcorr')"
        )

    np.random.seed(seed)
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)
    history = []
    graph_history_list = [] if return_graph_history else None

    A_inv = np.linalg.inv(A)

    allowed_offdiag, M = _build_faithfulness_mask(S, faithfulness_tau, screening)

    if early_stop:
        if check_every is None:
            check_every = d * (d + 1) // 2
        if min_steps is None:
            min_steps = check_every * 10
        prev_check_f = f(A, S)
        no_improve_count = 0

    for t in range(T):
        if allowed_offdiag is None:
            # tau <= 0: original uniform sampling (preserves RNG sequence).
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
            A = update_off_diagonal(A, S, i, j, lambda_l0, A_inv=A_inv)

        history.append(f(A, S))
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
    final_obj = history[-1] if len(history) > 0 else f(A, S)

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
