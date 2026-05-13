"""Random coordinate descent for DAG learning with weak-faithfulness screening.

This module implements `dag_coordinate_descent_l0_weakfaith`, a variant of
`coordinate0.dag_coordinate_descent_l0` (a.k.a. `cd_A_noepoch`) that prunes the
off-diagonal candidate pool using a one-edge faithfulness assumption: if the
marginal correlation, partial correlation, or Graphical-Lasso precision entry
between x_i and x_j is below a threshold tau, the coordinate pair (i, j) is
excluded from sampling.

Multiple screens can be combined via the ``combine`` argument. An optional
initial-gain mask can also freeze out directed pairs whose first-round L0 gain
is not positive.

See `docs/weak_faithfulness_cd_A_noepoch_zh.md` for the full design rationale.
"""

from itertools import combinations

import numpy as np

try:
    from .coordinate0 import (
        update_diagonal,
        update_off_diagonal,
        delta_star,
        f,
        is_DAG,
        weight_to_adjacency,
        _sm_update_A_inv,
        _graph_snapshot,
    )
except ImportError:
    from coordinate0 import (
        update_diagonal,
        update_off_diagonal,
        delta_star,
        f,
        is_DAG,
        weight_to_adjacency,
        _sm_update_A_inv,
        _graph_snapshot,
    )


_VALID_SCREENS = ("corr", "pcorr", "glasso", "moral_graph")


def _normalize_screening(screening):
    """Accept str or sequence of str; return a tuple of valid screen names."""
    if isinstance(screening, str):
        screens = (screening,)
    else:
        screens = tuple(screening)
        if len(screens) == 0:
            raise ValueError("screening must contain at least one method")
    for s in screens:
        if s not in _VALID_SCREENS:
            raise ValueError(
                f"unknown screening {s!r} (expected any of {_VALID_SCREENS})"
            )
    if len(set(screens)) != len(screens):
        raise ValueError(f"duplicate screening entries: {screens}")
    return screens


def _screen_forbidden(S, tau, screen, glasso_alpha,
                      X=None, moral_graph_alpha=0.01):
    """Return a (d, d) bool forbidden mask for a single screening method.

    Parameters
    ----------
    S, tau, screen, glasso_alpha : as before
        For ``screen in {"corr", "pcorr", "glasso"}`` the forbidden mask is
        ``|stat| < tau``.
    X : (n, d) ndarray, optional
        Raw data matrix; required when ``screen == "moral_graph"``.
    moral_graph_alpha : float, default 0.01
        Significance level for IAMB's Fisher-Z CI tests. Only used when
        ``screen == "moral_graph"``. Matches CALM's default
        (``run_experiment.py`` passes alpha=0.01).
    """
    d = S.shape[0]
    if screen == "moral_graph":
        if X is None:
            raise ValueError(
                "screen='moral_graph' requires X (the raw n×d data matrix); "
                "pass it as the X= keyword to dag_coordinate_descent_l0_weakfaith."
            )
        try:
            from .iamb import iamb_markov_network
        except ImportError:
            from iamb import iamb_markov_network
        moral, _ = iamb_markov_network(X, alpha=moral_graph_alpha)
        # forbidden = absent in moral graph (i.e. moral_mask[i,j] == 0)
        forbidden = ~(moral.astype(bool))
        np.fill_diagonal(forbidden, False)
        return forbidden
    if screen == "corr":
        std = np.sqrt(np.diag(S))
        stat = S / np.outer(std, std)
    elif screen == "pcorr":
        try:
            Omega = np.linalg.inv(S + 1e-6 * np.eye(d))
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "S is singular even after ridge regularization; "
                "cannot compute pcorr screening."
            ) from e
        d_std = np.sqrt(np.diag(Omega))
        stat = -Omega / np.outer(d_std, d_std)
    elif screen == "glasso":
        try:
            from sklearn.covariance import graphical_lasso
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for glasso screening; "
                "install scikit-learn."
            ) from e

        # Standardize to correlation scale: the raw covariance is often
        # ill-conditioned at small alpha (triggers "Non SPD result"), while
        # the correlation matrix has unit diagonal and entries in [-1, 1],
        # which is also the natural scale for our tau threshold.
        std_S = np.sqrt(np.diag(S))
        std_S = np.where(std_S > 0, std_S, 1.0)
        S_corr = S / np.outer(std_S, std_S)

        attempts = (
            {"alpha": glasso_alpha, "ridge": 0.0},
            {"alpha": glasso_alpha, "ridge": 1e-4},
            {"alpha": max(glasso_alpha * 5.0, 0.05), "ridge": 1e-3},
        )
        Omega_gl = None
        last_err = None
        for a in attempts:
            emp = S_corr + a["ridge"] * np.eye(d) if a["ridge"] > 0 else S_corr
            try:
                _, Omega_gl = graphical_lasso(
                    emp, alpha=a["alpha"], max_iter=500, tol=1e-3
                )
                break
            except (FloatingPointError, np.linalg.LinAlgError) as e:
                last_err = e
        if Omega_gl is None:
            raise ValueError(
                "graphical_lasso failed even after standardization, ridge, "
                f"and alpha bump (last error: {last_err!r}). "
                "Try a larger glasso_alpha or drop 'glasso' from screening."
            )

        d_std = np.sqrt(np.diag(Omega_gl))
        d_std = np.where(d_std > 0, d_std, 1.0)
        stat = -Omega_gl / np.outer(d_std, d_std)
    else:
        raise ValueError(f"unknown screening {screen!r}")

    forbidden = np.abs(stat) < tau
    np.fill_diagonal(forbidden, False)
    return forbidden


def _combine_forbidden_masks(forbidden_list, combine):
    if combine == "union":
        forbidden = forbidden_list[0].copy()
        for fb in forbidden_list[1:]:
            forbidden &= fb
    elif combine == "intersect":
        forbidden = forbidden_list[0].copy()
        for fb in forbidden_list[1:]:
            forbidden |= fb
    else:
        raise ValueError(
            f"unknown combine {combine!r} (expected 'union' or 'intersect')"
        )
    np.fill_diagonal(forbidden, False)
    return forbidden


def _offdiag_count(mask):
    d = mask.shape[0]
    offdiag = ~np.eye(d, dtype=bool)
    return int(np.sum(mask & offdiag))


def faithfulness_mask_diagnostics(
    S, tau, screening, glasso_alpha=0.01, combine="union",
    X=None, moral_graph_alpha=0.01,
):
    """Return numeric diagnostics for weak-faithfulness screening masks.

    Counts are over directed off-diagonal coordinate pairs, so the denominator
    is ``d * (d - 1)``. ``mask_zero_count`` is the number of pairs removed by
    the combined mask used by the algorithm. Per-screen counts and pairwise
    overlaps describe how much each individual screen removes and how much the
    removed sets agree.

    ``X`` and ``moral_graph_alpha`` are only consulted when ``screening``
    includes ``"moral_graph"``.
    """
    d = S.shape[0]
    total = d * (d - 1)
    screens = _normalize_screening(screening)
    stats = {
        "mask_total_offdiag": int(total),
        "mask_tau": float(tau),
        "mask_n_screens": int(len(screens)),
    }

    tau_based = {"corr", "pcorr", "glasso"}
    has_moral = "moral_graph" in screens
    if tau <= 0.0 and not has_moral:
        stats.update(
            {
                "mask_allowed_count": int(total),
                "mask_zero_count": 0,
                "mask_keep_ratio": 1.0,
                "mask_zero_ratio": 0.0,
            }
        )
        return stats

    # Skip tau-based screens when tau<=0 in the diagnostics dict too.
    forbidden_by_screen = {
        s: _screen_forbidden(
            S, tau, s, glasso_alpha,
            X=X, moral_graph_alpha=moral_graph_alpha,
        )
        for s in screens
        if s == "moral_graph" or (s in tau_based and tau > 0.0)
    }
    # Iterate only over screens that actually contributed (others were dropped
    # because tau<=0 disabled them).
    active_screens = tuple(s for s in screens if s in forbidden_by_screen)
    forbidden_list = [forbidden_by_screen[s] for s in active_screens]
    combined = _combine_forbidden_masks(forbidden_list, combine)

    zero_count = _offdiag_count(combined)
    allowed_count = total - zero_count
    stats.update(
        {
            "mask_allowed_count": int(allowed_count),
            "mask_zero_count": int(zero_count),
            "mask_keep_ratio": float(allowed_count / total) if total else 1.0,
            "mask_zero_ratio": float(zero_count / total) if total else 0.0,
        }
    )

    for s, fb in forbidden_by_screen.items():
        screen_zero = _offdiag_count(fb)
        stats[f"mask_{s}_zero_count"] = int(screen_zero)
        stats[f"mask_{s}_allowed_count"] = int(total - screen_zero)
        stats[f"mask_{s}_zero_ratio"] = (
            float(screen_zero / total) if total else 0.0
        )
        stats[f"mask_{s}_keep_ratio"] = (
            float((total - screen_zero) / total) if total else 1.0
        )

    for a, b in combinations(active_screens, 2):
        fa = forbidden_by_screen[a]
        fb = forbidden_by_screen[b]
        inter = _offdiag_count(fa & fb)
        union = _offdiag_count(fa | fb)
        min_zero = min(_offdiag_count(fa), _offdiag_count(fb))
        prefix = f"mask_overlap_{a}_{b}"
        stats[f"{prefix}_count"] = int(inter)
        stats[f"{prefix}_union_count"] = int(union)
        stats[f"{prefix}_jaccard"] = float(inter / union) if union else 1.0
        stats[f"{prefix}_overlap_ratio"] = (
            float(inter / min_zero) if min_zero else 1.0
        )

    if len(active_screens) >= 2:
        all_inter = forbidden_list[0].copy()
        all_union = forbidden_list[0].copy()
        for fb in forbidden_list[1:]:
            all_inter &= fb
            all_union |= fb
        all_inter_count = _offdiag_count(all_inter)
        all_union_count = _offdiag_count(all_union)
        stats["mask_overlap_all_count"] = int(all_inter_count)
        stats["mask_overlap_all_union_count"] = int(all_union_count)
        stats["mask_overlap_all_jaccard"] = (
            float(all_inter_count / all_union_count) if all_union_count else 1.0
        )

    return stats


def _build_faithfulness_mask(
    S, tau, screening, glasso_alpha=0.01, combine="union",
    X=None, moral_graph_alpha=0.01,
):
    """Build the forbidden/allowed masks for one-edge faithfulness screening.

    Parameters
    ----------
    S : (d, d) ndarray
        Sample covariance matrix (or Gram matrix divided by n). Must be symmetric
        with positive diagonal.
    tau : float
        Threshold below which |statistic| is treated as zero. ``tau <= 0``
        disables every tau-based screen ({"corr", "pcorr", "glasso"}); a
        ``"moral_graph"`` screen is unaffected by ``tau`` (its threshold is
        ``moral_graph_alpha``).
    screening : str or sequence of str
        One or more of {"corr", "pcorr", "glasso", "moral_graph"}. A sequence
        activates all listed screens and combines them via ``combine``.
    glasso_alpha : float, default 0.01
        Regularization strength passed to ``sklearn.covariance.graphical_lasso``
        when "glasso" is selected.
    combine : {"union", "intersect"}, default "union"
        How to merge forbidden masks across multiple screens.

        - "union" (default, matches the rule "discard only if all screens say
          zero"): an edge is forbidden only when every selected screen marks it
          as zero. Allowed set = union of per-screen allowed sets.
        - "intersect": an edge is forbidden if any screen marks it as zero
          (stricter pruning). Allowed set = intersection of per-screen allowed
          sets.
    X : (n, d) ndarray, optional
        Raw data matrix. Required only when ``"moral_graph"`` is in
        ``screening``.
    moral_graph_alpha : float, default 0.01
        Fisher-Z significance level for IAMB; only used when ``"moral_graph"``
        is in ``screening``. Matches CALM's default.

    Returns
    -------
    allowed_offdiag : (M, 2) ndarray or None
        Row = index pair (i, j) with i != j that survives screening.
        None when no screen is active.
    M : int
        Number of allowed off-diagonal pairs.
    """
    d = S.shape[0]
    screens = _normalize_screening(screening)

    # Drop tau-based screens when tau <= 0 (they would block nothing); keep
    # 'moral_graph' regardless. If nothing remains, fall back to "no screening".
    tau_based = {"corr", "pcorr", "glasso"}
    active_screens = tuple(
        s for s in screens
        if s == "moral_graph" or (s in tau_based and tau > 0.0)
    )
    if len(active_screens) == 0:
        return None, d * (d - 1)

    forbidden_list = [
        _screen_forbidden(
            S, tau, s, glasso_alpha,
            X=X, moral_graph_alpha=moral_graph_alpha,
        )
        for s in active_screens
    ]

    forbidden = _combine_forbidden_masks(forbidden_list, combine)
    allowed = np.argwhere(~forbidden & ~np.eye(d, dtype=bool))
    M = len(allowed)
    if M == 0:
        raise ValueError(
            f"All off-diagonal pairs masked (tau={tau}, screening={screens!r}, "
            f"combine={combine!r}); try a smaller tau or a more lenient combine."
        )
    return allowed, M


def _zero_pair_in_place(A, A_inv, i, j):
    """Zero A[i, j] and A[j, i], keeping A_inv in sync when supplied."""
    old_aij = A[i, j]
    old_aji = A[j, i]
    A[i, j] = A[j, i] = 0.0

    if A_inv is None:
        return

    ok = True
    if old_aij != 0.0:
        ok = _sm_update_A_inv(A_inv, i, j, -old_aij)
    if ok and old_aji != 0.0:
        ok = _sm_update_A_inv(A_inv, j, i, -old_aji)
    if not ok:
        A_inv[:] = np.linalg.inv(A)


def _directed_l0_net_gain(A, S, i, j, lambda_l0, A_inv=None):
    """Return the best one-direction L0 net gain for i -> j from current A."""
    d = A.shape[0]
    Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
    if not is_DAG(A + Eij):
        return -np.inf

    delta = delta_star(A, S, i, j, A_inv=A_inv)
    if A_inv is None:
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return -np.inf
    alpha = float(A_inv[j, i])
    b = float(S[i, :] @ A[:, j])
    c = float(S[i, i])
    log_arg = 1.0 + delta * alpha
    if log_arg <= 0.0:
        return -np.inf

    score = 2.0 * np.log(log_arg) - 2.0 * delta * b - delta * delta * c
    return float(score - lambda_l0)


def _build_initial_gain_mask(A, S, lambda_l0, allowed_offdiag=None, eps=0.0):
    """Keep only directed pairs whose initial L0 update has positive net gain."""
    d = S.shape[0]
    allowed_direction = np.zeros((d, d), dtype=bool)

    if allowed_offdiag is None:
        candidates = np.argwhere(~np.eye(d, dtype=bool))
    else:
        candidates = allowed_offdiag

    for i_raw, j_raw in candidates:
        i = int(i_raw)
        j = int(j_raw)
        A_pair = A.copy()
        try:
            A_pair_inv = np.linalg.inv(A_pair)
        except np.linalg.LinAlgError:
            continue
        try:
            _zero_pair_in_place(A_pair, A_pair_inv, i, j)
            gain = _directed_l0_net_gain(
                A_pair, S, i, j, lambda_l0, A_inv=A_pair_inv
            )
        except np.linalg.LinAlgError:
            continue
        if gain > eps:
            allowed_direction[i, j] = True

    allowed = np.argwhere(allowed_direction)
    return allowed, len(allowed), allowed_direction


def _update_off_diagonal_with_direction_mask(
    A, S, i, j, lambda_l0=0.2, A_inv=None, direction_mask=None
):
    """Pair update that never evaluates directions forbidden by direction_mask."""
    _zero_pair_in_place(A, A_inv, i, j)

    gain_ij = -np.inf
    gain_ji = -np.inf
    delta_ij = 0.0
    delta_ji = 0.0

    if direction_mask is None or direction_mask[i, j]:
        gain_ij = _directed_l0_net_gain(A, S, i, j, lambda_l0, A_inv=A_inv)
        if gain_ij > 0.0:
            delta_ij = delta_star(A, S, i, j, A_inv=A_inv)

    if direction_mask is None or direction_mask[j, i]:
        gain_ji = _directed_l0_net_gain(A, S, j, i, lambda_l0, A_inv=A_inv)
        if gain_ji > 0.0:
            delta_ji = delta_star(A, S, j, i, A_inv=A_inv)

    if gain_ij == -np.inf and gain_ji == -np.inf:
        return A
    if gain_ij < 0.0 and gain_ji < 0.0:
        return A

    if gain_ij > gain_ji:
        if A_inv is not None:
            if not _sm_update_A_inv(A_inv, i, j, delta_ij):
                A_new = A.copy()
                A_new[i, j] += delta_ij
                A_inv[:] = np.linalg.inv(A_new)
        A[i, j] += delta_ij
    else:
        if A_inv is not None:
            if not _sm_update_A_inv(A_inv, j, i, delta_ji):
                A_new = A.copy()
                A_new[j, i] += delta_ji
                A_inv[:] = np.linalg.inv(A_new)
        A[j, i] += delta_ji
    return A


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
    glasso_alpha=0.01,
    combine="union",
    initial_gain_mask=False,
    initial_gain_eps=0.0,
    # ★ added 2026-05: scoring + moral-graph screen
    score="l0",
    n_samples=None,
    penalty_discount=1.0,
    X=None,
    moral_graph_alpha=0.01,
):
    """Random coordinate descent with one-edge faithfulness screening.

    Semantically equivalent to `coordinate0.dag_coordinate_descent_l0` when
    `faithfulness_tau <= 0`, `initial_gain_mask=False`, `score="l0"`, and
    ``"moral_graph"`` is not in `screening`; in that regime the RNG call
    sequence is identical, so results match byte-for-byte under the same seed.

    Parameters
    ----------
    S, T, seed, threshold, lambda_l0, return_history, return_graph_history,
    A_init, early_stop, check_every, tol, patience, min_steps :
        Same as `coordinate0.dag_coordinate_descent_l0`.
    faithfulness_tau : float, default 0.0
        Weak-faithfulness screening threshold for tau-based screens
        ({"corr", "pcorr", "glasso"}). 0 disables them. ``"moral_graph"`` does
        not use this threshold.
    sampling_mode : {"preserve", "pool"}, default "preserve"
        How to allocate sampling probability between diagonal and off-diagonal
        coordinates when screening is active.

        - "preserve": P(diag) = 1/d (matches original), off-diag uniform over
          the allowed pool.
        - "pool": uniform over {d diagonal coords} ∪ {M allowed off-diag coords}.
    screening : str or sequence of str, default "corr"
        One or more statistics used to judge "x_i and x_j are approximately
        independent". Valid names:

        - "corr"        : marginal correlation
        - "pcorr"       : partial correlation, conditioning on all other variables
        - "glasso"      : Graphical-Lasso precision entry, normalized to pcorr scale
        - "moral_graph" : IAMB Markov-network estimate (Fisher-Z CI tests),
          identical to the moral-graph stage in CALM
          (kaifeng-jin/CALM, ``iamb.iamb_markov_network``).

        Pass a sequence (e.g. ``["pcorr", "moral_graph"]``) to enable multiple
        screens; they are merged via ``combine``.
    glasso_alpha : float, default 0.01
        L1 regularization strength for Graphical Lasso; only used when
        "glasso" is among the selected screens.
    combine : {"union", "intersect"}, default "union"
        How to merge forbidden masks when multiple screens are selected.

        - "union": an edge is forbidden only when every screen marks it as
          zero (allowed set = union of per-screen allowed sets).
        - "intersect": an edge is forbidden if any screen marks it as zero
          (stricter pruning, allowed set = intersection).

        Ignored when only one screen is selected.
    initial_gain_mask : bool, default False
        If True, run one initial scan over directed off-diagonal pairs and keep
        only pairs whose one-direction L0 net gain is positive at the initial A.
    initial_gain_eps : float, default 0.0
        Numerical tolerance for `initial_gain_mask`.
    score : {"l0", "bic"}, default "l0"
        Per-edge penalty in the L0-style sub-objective evaluated by
        ``update_off_diagonal``.

        - "l0"  : per-edge penalty = ``lambda_l0`` (original behavior).
        - "bic" : per-edge penalty = ``penalty_discount * log(n_samples) /
          n_samples``. This makes the algorithm minimize a Gaussian SEM BIC
          (since cd_A's per-sample loss ``f(A,S) = -2 log det(A) + tr(A^T S A)``
          equals ``-2/n × log L`` up to a constant for equal-variance Gaussian
          SEM, so adding ``log(n)/n × #edges`` per sample = ``log(n) × #edges``
          total = the BIC penalty).
    n_samples : int, optional
        Sample size; required when ``score="bic"``.
    penalty_discount : float, default 1.0
        Multiplier on the BIC penalty (matches FGES's ``penalty_discount``).
        Larger -> sparser. Only used when ``score="bic"``.
    X : (n_samples, d) ndarray, optional
        Raw data matrix; required when ``"moral_graph"`` is in ``screening``.
    moral_graph_alpha : float, default 0.01
        Significance level for IAMB's Fisher-Z CI tests (matches CALM).

    Returns
    -------
    Same tuple shape as `coordinate0.dag_coordinate_descent_l0`.
    """
    if sampling_mode not in ("preserve", "pool"):
        raise ValueError(
            f"unknown sampling_mode {sampling_mode!r} "
            f"(expected 'preserve' or 'pool')"
        )

    # Resolve effective per-edge penalty from `score`.
    if score == "l0":
        lambda_eff = float(lambda_l0)
    elif score == "bic":
        if n_samples is None:
            if X is not None:
                n_samples = int(X.shape[0])
            else:
                raise ValueError(
                    "score='bic' requires n_samples (or X, from which n_samples "
                    "is inferred)."
                )
        if n_samples <= 1:
            raise ValueError(f"n_samples must be > 1 for BIC, got {n_samples}")
        lambda_eff = float(penalty_discount) * np.log(n_samples) / n_samples
    else:
        raise ValueError(f"unknown score {score!r}; expected 'l0' or 'bic'")

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
        X=X,
        moral_graph_alpha=moral_graph_alpha,
    )
    initial_direction_mask = None
    if initial_gain_mask:
        allowed_offdiag, M, initial_direction_mask = _build_initial_gain_mask(
            A,
            S,
            lambda_eff,
            allowed_offdiag=allowed_offdiag,
            eps=initial_gain_eps,
        )

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
        elif M == 0:
            i = j = int(np.random.randint(d))
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
        elif initial_direction_mask is not None:
            A = _update_off_diagonal_with_direction_mask(
                A,
                S,
                i,
                j,
                lambda_eff,
                A_inv=A_inv,
                direction_mask=initial_direction_mask,
            )
        else:
            A = update_off_diagonal(A, S, i, j, lambda_eff, A_inv=A_inv)

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
