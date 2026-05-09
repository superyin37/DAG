import numpy as np
from typing import List, Optional

try:
    from .coordinate0 import f, delta_star, weight_to_adjacency
except ImportError:
    from coordinate0 import f, delta_star, weight_to_adjacency


def _build_adjacency(A: np.ndarray, threshold: float = 1e-8) -> List[set]:
    """Build an adjacency list from nonzero off-diagonal entries of A."""
    d = A.shape[0]
    adj = [set() for _ in range(d)]
    rows, cols = np.where((np.abs(A) > threshold) & ~np.eye(d, dtype=bool))
    for i, j in zip(rows.tolist(), cols.tolist()):
        adj[i].add(j)
    return adj


def _is_dag_kahn(adj: List[set]) -> bool:
    """Check whether the adjacency list represents a DAG."""
    d = len(adj)
    in_deg = [0] * d
    for u in range(d):
        for v in adj[u]:
            in_deg[v] += 1

    queue = [u for u in range(d) if in_deg[u] == 0]
    count = 0
    while queue:
        u = queue.pop()
        count += 1
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    return count == d


def _can_add_edge(adj: List[set], i: int, j: int) -> bool:
    """Check whether adding i -> j preserves acyclicity."""
    if i == j:
        return False
    adj[i].add(j)
    result = _is_dag_kahn(adj)
    adj[i].discard(j)
    return result


def _can_flip_edge(adj: List[set], i: int, j: int) -> bool:
    """Check whether replacing j -> i with i -> j preserves acyclicity.

    Temporarily removes j -> i (which is assumed to currently be present) and
    adds i -> j, runs Kahn's check, then restores the original adjacency.
    """
    if i == j:
        return False
    adj[j].discard(i)
    adj[i].add(j)
    result = _is_dag_kahn(adj)
    adj[i].discard(j)
    adj[j].add(i)
    return result


def _rank1_update(
    A: np.ndarray,
    A_inv: np.ndarray,
    G: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    delta: float,
    f_state: Optional[list] = None,
    adj: Optional[List[set]] = None,
    adj_threshold: float = 1e-8,
) -> bool:
    """
    Apply A[i, j] += delta and update cached A_inv, G, f_state, and adj in place.

    Returns False only when the Sherman-Morrison denominator is too small and the
    caller should fall back to an exact recomputation.
    """
    if abs(delta) < 1e-30:
        return True

    alpha = float(A_inv[j, i])
    denom = 1.0 + delta * alpha
    if abs(denom) < 1e-15:
        return False

    u = A_inv[:, i].copy()
    v = A_inv[j, :].copy()
    c = -delta / denom

    if f_state is not None:
        sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
        f_state[0] += (
            -2.0 * np.log1p(delta * alpha)
            + 2.0 * delta * sa_ij
            + delta ** 2 * float(S[i, i])
        )

    A_inv += c * np.outer(u, v)
    G += (-2.0 * c) * np.outer(v, u)
    G[:, j] += 2.0 * delta * S[:, i]
    A[i, j] += delta

    if adj is not None and i != j:
        if abs(A[i, j]) > adj_threshold:
            adj[i].add(j)
        else:
            adj[i].discard(j)

    return True


def _recompute_all(
    A: np.ndarray,
    S: np.ndarray,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
) -> None:
    """Recompute cached inverse, gradient, and objective exactly from A."""
    A_inv[:] = np.linalg.inv(A)
    G[:] = -2.0 * A_inv.T + 2.0 * S @ A
    sign, logabsdet = np.linalg.slogdet(A)
    f_state[0] = (
        -2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))
        if sign > 0
        else np.inf
    )


def _incremental_df(alpha: float, sa_ij: float, s_ii: float, delta: float) -> float:
    """Closed-form change in f(A, S) after A[i, j] += delta."""
    return (
        -2.0 * np.log1p(delta * alpha)
        + 2.0 * delta * sa_ij
        + delta ** 2 * s_ii
    )


def _pair_scores(G: np.ndarray):
    """Return upper-triangle pair scores max(|G[i,j]|, |G[j,i]|)."""
    score_matrix = np.maximum(np.abs(G), np.abs(G.T))
    ii, jj = np.triu_indices(G.shape[0], k=1)
    return ii, jj, score_matrix[ii, jj]


def _candidate_iter(scores: np.ndarray, mode: str):
    """Yield candidate indices in (descending) score order.

    Two strategies are supported:

    * ``"sorted"`` — one ``argsort`` up front (``O(N log N)``), then iterate.
      The original ``scores`` array is not mutated. Stops at the first
      non-positive entry, so the caller does not need its own break check.
    * ``"argmax"`` — repeated ``argmax`` (``O(m N)`` for m probes) on a local
      copy of ``scores`` whose visited entries are marked ``-1``.

    See `dag_greedy_A` for the cost trade-off between the two strategies.
    """
    if mode == "sorted":
        order = np.argsort(scores)[::-1]
        for idx in order.tolist():
            if scores[idx] <= 0.0:
                return
            yield int(idx)
    elif mode == "argmax":
        work = scores.copy()
        while True:
            idx = int(np.argmax(work))
            if work[idx] <= 0.0:
                return
            yield idx
            work[idx] = -1.0
    else:
        raise ValueError(
            f"unknown selection mode: {mode!r} (expected 'sorted' or 'argmax')"
        )


def _directed_coordinate_gain(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float,
    A_inv: np.ndarray,
    G: np.ndarray,
    adj: List[set],
) -> tuple[float, list]:
    """
    Return (gain, plan) for the directed coordinate A[i, j].

    plan is a list of (row, col, delta) tuples to apply in order. There are
    three cases depending on the current state of (i, j) in the DAG:

      1. i -> j already exists: re-optimize A[i, j] in place.
      2. j -> i exists but i -> j does not: flip the edge — first zero out
         A[j, i], then take the 1-D optimum at A[i, j] from the post-zero
         state. The combined two-step objective change is reported as gain.
      3. neither direction exists: add the new edge i -> j (with L0 penalty)
         provided the DAG constraint allows it.

    gain returns the net objective decrease after the relevant L0 accounting.
    Empty plan / -inf gain means the candidate is infeasible.
    """
    if i == j:
        return -np.inf, []

    edge_exists = j in adj[i]
    reverse_exists = i in adj[j]

    if edge_exists:
        # Case 1: re-optimize an existing directed edge.
        delta = delta_star(A, S, i, j, A_inv=A_inv)
        alpha = float(A_inv[j, i])
        sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
        df = _incremental_df(alpha, sa_ij, float(S[i, i]), delta)
        return -df, [(i, j, delta)]

    if reverse_exists:
        # Case 2: try to flip j -> i into i -> j.
        if not _can_flip_edge(adj, i, j):
            return -np.inf, []

        # Step 1: zero A[j, i].
        delta1 = -float(A[j, i])
        alpha1 = float(A_inv[i, j])
        denom = 1.0 + delta1 * alpha1
        if abs(denom) < 1e-15:
            return -np.inf, []
        sa_ji = (float(G[j, i]) + 2.0 * alpha1) / 2.0
        df1 = _incremental_df(alpha1, sa_ji, float(S[j, j]), delta1)

        # Step 2: optimal delta for A[i, j] from the post-zero state.
        # (S A')_{ij} == (S A)_{ij} because zeroing A[j, i] only touches the
        # i-th column of A, not the j-th, so sa_ij is unchanged.
        sa_ij = (float(G[i, j]) + 2.0 * float(A_inv[j, i])) / 2.0
        # alpha_2 = (A')^{-1}[j, i] via Sherman-Morrison on the rank-1 update
        # A' = A + delta1 * e_j e_i^T.
        c_sm = delta1 / denom
        alpha2 = (
            float(A_inv[j, i])
            - c_sm * float(A_inv[j, j]) * float(A_inv[i, i])
        )
        s_ii = float(S[i, i])

        if abs(alpha2) < 1e-12:
            delta2 = -sa_ij / s_ii
        else:
            D = (s_ii + alpha2 * sa_ij) ** 2 - 4.0 * alpha2 * s_ii * (sa_ij - alpha2)
            D = max(D, 0.0)
            delta2 = 2.0 * (sa_ij - alpha2) / (
                -(s_ii + alpha2 * sa_ij) - np.sqrt(D)
            )

        df2 = _incremental_df(alpha2, sa_ij, s_ii, delta2)
        # Edge count is unchanged (one removed, one added) so L0 cancels out.
        return -(df1 + df2), [(j, i, delta1), (i, j, delta2)]

    # Case 3: add a brand-new edge i -> j.
    if not _can_add_edge(adj, i, j):
        return -np.inf, []

    delta = delta_star(A, S, i, j, A_inv=A_inv)
    alpha = float(A_inv[j, i])
    sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
    df = _incremental_df(alpha, sa_ij, float(S[i, i]), delta)
    return -df - lambda_l0, [(i, j, delta)]


def _diagonal_coordinate_gain(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    A_inv: np.ndarray,
    G: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return (gain, reset_delta, final_delta) for the diagonal coordinate A[i, i].

    The diagonal update mirrors coordinate0.update_diagonal:
    reset to 0.3, then take the exact 1-D optimum from that reset state.
    """
    reset_delta = 0.3 - float(A[i, i])
    alpha = float(A_inv[i, i])
    sa_ii = (float(G[i, i]) + 2.0 * alpha) / 2.0
    s_ii = float(S[i, i])

    df_reset = _incremental_df(alpha, sa_ii, s_ii, reset_delta)

    reset_alpha = alpha / (1.0 + reset_delta * alpha)
    reset_sa_ii = sa_ii + reset_delta * s_ii

    if abs(reset_alpha) < 1e-12:
        final_delta = -reset_sa_ii / s_ii
    else:
        b = reset_sa_ii
        D = (s_ii + reset_alpha * b) ** 2 - 4.0 * reset_alpha * s_ii * (b - reset_alpha)
        D = max(D, 0.0)
        final_delta = 2.0 * (b - reset_alpha) / (
            -(s_ii + reset_alpha * b) - np.sqrt(D)
        )

    df_final = _incremental_df(reset_alpha, reset_sa_ii, s_ii, final_delta)
    return -(df_reset + df_final), reset_delta, final_delta


def _apply_cached_update(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    delta: float,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
    adj: Optional[List[set]] = None,
) -> None:
    """Apply a cached rank-1 update, falling back to exact recomputation if needed."""
    ok = _rank1_update(A, A_inv, G, S, i, j, delta, f_state, adj)
    if ok:
        return

    A[i, j] += delta
    if adj is not None and i != j:
        if abs(A[i, j]) > 1e-8:
            adj[i].add(j)
        else:
            adj[i].discard(j)
    _recompute_all(A, S, A_inv, G, f_state)


def update_off_diagonal_greedy(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
    adj: List[set],
) -> None:
    """
    Pair-based off-diagonal update.

    This keeps the historical epoch behavior: zero both directions, then add back
    the better feasible direction if it improves the objective.
    """
    if i == j:
        return

    def _zero(r: int, c: int) -> None:
        old = float(A[r, c])
        if old == 0.0:
            return
        ok = _rank1_update(A, A_inv, G, S, r, c, -old, f_state, adj)
        if not ok:
            A[r, c] = 0.0
            adj[r].discard(c)
            _recompute_all(A, S, A_inv, G, f_state)

    _zero(i, j)
    _zero(j, i)

    gain_ij = gain_ji = -np.inf
    delta_ij = delta_ji = 0.0

    if _can_add_edge(adj, i, j):
        delta_ij = delta_star(A, S, i, j, A_inv=A_inv)
        alpha = float(A_inv[j, i])
        sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
        gain_ij = -_incremental_df(alpha, sa_ij, float(S[i, i]), delta_ij) - lambda_l0

    if _can_add_edge(adj, j, i):
        delta_ji = delta_star(A, S, j, i, A_inv=A_inv)
        alpha = float(A_inv[i, j])
        sa_ji = (float(G[j, i]) + 2.0 * alpha) / 2.0
        gain_ji = -_incremental_df(alpha, sa_ji, float(S[j, j]), delta_ji) - lambda_l0

    if gain_ij <= 0.0 and gain_ji <= 0.0:
        return

    if gain_ij > gain_ji:
        _apply_cached_update(A, S, i, j, delta_ij, A_inv, G, f_state, adj)
    else:
        _apply_cached_update(A, S, j, i, delta_ji, A_inv, G, f_state, adj)


def update_directed_coordinate(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
    adj: List[set],
) -> bool:
    """Apply one directed off-diagonal update if it has positive gain.

    The plan returned by _directed_coordinate_gain may contain one step
    (re-optimize / add) or two steps (flip: zero the reverse edge, then
    activate the forward edge).
    """
    gain, plan = _directed_coordinate_gain(A, S, i, j, lambda_l0, A_inv, G, adj)
    if gain <= 0.0:
        return False
    for r, c, d in plan:
        _apply_cached_update(A, S, r, c, d, A_inv, G, f_state, adj)
    return True


def update_diagonal_greedy(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
) -> None:
    """Apply one diagonal update if it has positive net gain."""
    gain, reset_delta, final_delta = _diagonal_coordinate_gain(A, S, i, A_inv, G)
    if gain <= 0.0:
        return

    _apply_cached_update(A, S, i, i, reset_delta, A_inv, G, f_state, adj=None)
    _apply_cached_update(A, S, i, i, final_delta, A_inv, G, f_state, adj=None)


def _init_state(S: np.ndarray, A_init: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, List[set]]:
    """Initialize A and all cached state."""
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)
    A_inv = np.linalg.inv(A)
    G = -2.0 * A_inv.T + 2.0 * S @ A
    sign, logabsdet = np.linalg.slogdet(A)
    f_state = [-2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))]
    if sign <= 0:
        f_state[0] = np.inf
    adj = _build_adjacency(A)
    return A, A_inv, G, f_state, adj


def _dag_greedy_A_impl(
    S: np.ndarray,
    T: int,
    seed: int,
    threshold: float,
    lambda_l0: float,
    return_history: bool,
    A_init: Optional[np.ndarray],
    include_diagonal: bool,
    selection: str = "sorted",
    early_stop: bool = False,
    tol: float = 1e-4,
    patience: int = 10,
    check_every: Optional[int] = None,
    min_steps: Optional[int] = None,
    return_graph_history: bool = False,
):
    """Shared implementation for the T-step greedy variants.

    Candidate selection inside each outer step is delegated to
    `_candidate_iter`. ``selection='sorted'`` (default) does one upfront
    ``argsort``; ``selection='argmax'`` does the historical repeated
    ``argmax`` with in-place rejection. See `dag_greedy_A` for guidance.

    Two early-stopping mechanisms coexist:

    * Implicit (always on): break as soon as no candidate yields a positive
      objective decrease — this is the KKT-style "nothing left to improve"
      stop and is independent of `early_stop`.
    * Explicit (opt in via `early_stop=True`): every `check_every` iterations,
      compute the relative improvement against the previous checkpoint. If it
      stays below `tol` for `patience` consecutive checks (and we have already
      run at least `min_steps` iterations), break. This catches the
      "improving but tiny" regime where the implicit stop would never fire.

    Defaults follow `coordinate0.dag_coordinate_descent_l0`:
    `check_every = d*(d+1)//2`, `min_steps = check_every * 10`.
    """
    np.random.seed(seed)
    d = S.shape[0]
    A, A_inv, G, f_state, adj = _init_state(S, A_init)

    ii_off, jj_off = np.where(~np.eye(d, dtype=bool))
    history = []
    graph_history_list = [] if return_graph_history else None

    if early_stop:
        if check_every is None:
            check_every = d * (d + 1) // 2
        if min_steps is None:
            min_steps = check_every * 10
        prev_check_f = f_state[0]
        no_improve_count = 0

    for t in range(T):
        offdiag_scores = np.abs(G[ii_off, jj_off])
        if include_diagonal:
            diag_scores = np.abs(np.diag(G))
            scores = np.concatenate([diag_scores, offdiag_scores])
        else:
            scores = offdiag_scores

        selected = None
        for idx in _candidate_iter(scores, selection):
            if include_diagonal and idx < d:
                gain, _, _ = _diagonal_coordinate_gain(A, S, idx, A_inv, G)
                if gain > 0.0:
                    selected = ("diag", int(idx))
                    break
                continue

            off_idx = idx - d if include_diagonal else idx
            i = int(ii_off[off_idx])
            j = int(jj_off[off_idx])
            gain, _ = _directed_coordinate_gain(A, S, i, j, lambda_l0, A_inv, G, adj)
            if gain > 0.0:
                selected = ("offdiag", i, j)
                break

        if selected is None:
            break

        if selected[0] == "diag":
            update_diagonal_greedy(A, S, selected[1], A_inv, G, f_state)
        else:
            update_directed_coordinate(
                A, S, selected[1], selected[2], lambda_l0, A_inv, G, f_state, adj
            )
        history.append(f_state[0])
        if graph_history_list is not None:
            graph_history_list.append(weight_to_adjacency(A, threshold).astype(np.uint8))

        if early_stop and (t + 1) % check_every == 0:
            curr_f = f_state[0]
            rel_improve = (prev_check_f - curr_f) / max(1.0, abs(prev_check_f))
            prev_check_f = curr_f
            if t + 1 >= min_steps:
                if rel_improve < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    break

    G_binary = weight_to_adjacency(A, threshold)
    graph_history = (
        np.array(graph_history_list, dtype=np.uint8) if graph_history_list is not None else None
    )
    if return_history and return_graph_history:
        return A, G_binary, f_state[0], history, graph_history
    if return_history:
        return A, G_binary, f_state[0], history
    if return_graph_history:
        return A, G_binary, f_state[0], graph_history
    return A, G_binary, f_state[0]


def dag_greedy_A(
    S: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    return_history: bool = False,
    A_init: Optional[np.ndarray] = None,
    selection: str = "sorted",
    early_stop: bool = False,
    tol: float = 1e-4,
    patience: int = 10,
    check_every: Optional[int] = None,
    min_steps: Optional[int] = None,
    return_graph_history: bool = False,
):
    """
    Main T-step greedy solver.

    Uses directed coordinate selection, includes diagonal updates, and stops early
    when no coordinate yields positive objective decrease (always-on implicit
    stop). Pass `early_stop=True` to additionally enable a relative-improvement
    early stop with `tol`/`patience`/`check_every`/`min_steps`.

    `selection` controls how the inner candidate scan is ordered:

    * ``"sorted"`` (default): one ``argsort`` per outer step, ``O(N log N)``
      preprocessing then iterate. Asymptotically better when many candidates
      get rejected per step (typical near convergence).
    * ``"argmax"``: repeated ``argmax`` with in-place rejection,
      ``O(m N)`` for m probes. Slightly faster when m is very small.
    """
    return _dag_greedy_A_impl(
        S=S,
        T=T,
        seed=seed,
        threshold=threshold,
        lambda_l0=lambda_l0,
        return_history=return_history,
        A_init=A_init,
        include_diagonal=True,
        selection=selection,
        early_stop=early_stop,
        tol=tol,
        patience=patience,
        check_every=check_every,
        min_steps=min_steps,
        return_graph_history=return_graph_history,
    )


def dag_greedy_A_directed(
    S: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    return_history: bool = False,
    A_init: Optional[np.ndarray] = None,
    selection: str = "sorted",
    early_stop: bool = False,
    tol: float = 1e-4,
    patience: int = 10,
    check_every: Optional[int] = None,
    min_steps: Optional[int] = None,
):
    """
    Directed off-diagonal-only T-step greedy solver.

    This is retained as a baseline for experiments that want the old
    directed-without-diagonal behavior. Supports the same `selection` and
    explicit early-stop parameters as `dag_greedy_A`.
    """
    return _dag_greedy_A_impl(
        S=S,
        T=T,
        seed=seed,
        threshold=threshold,
        lambda_l0=lambda_l0,
        return_history=return_history,
        A_init=A_init,
        include_diagonal=False,
        selection=selection,
        early_stop=early_stop,
        tol=tol,
        patience=patience,
        check_every=check_every,
        min_steps=min_steps,
    )


def dag_greedy_A_epoch(
    S: np.ndarray,
    n_epochs: int = 500,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    tol: float = 1e-4,
    patience: int = 10,
    min_epochs: int = 50,
    verbose: bool = False,
    A_init: Optional[np.ndarray] = None,
):
    """
    Epoch-based greedy coordinate descent.

    Each epoch runs a pair-based structure block followed by a diagonal block.
    """
    np.random.seed(seed)
    A = A_init.copy() if A_init is not None else np.eye(S.shape[0])

    history = []
    no_improve = 0
    prev_val = f(A, S)
    curr_val = prev_val

    for epoch in range(1, n_epochs + 1):
        A_inv = np.linalg.inv(A)
        G = -2.0 * A_inv.T + 2.0 * S @ A
        sign, logabsdet = np.linalg.slogdet(A)
        f_state = [-2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))]
        if sign <= 0:
            f_state[0] = np.inf
        adj = _build_adjacency(A)

        ii, jj, scores = _pair_scores(G)
        sorted_idx = np.argsort(scores)[::-1]
        for idx in sorted_idx.tolist():
            update_off_diagonal_greedy(
                A, S, int(ii[idx]), int(jj[idx]), lambda_l0, A_inv, G, f_state, adj
            )

        for i in range(S.shape[0]):
            update_diagonal_greedy(A, S, i, A_inv, G, f_state)

        curr_val = f_state[0]
        history.append(curr_val)
        rel_improve = (prev_val - curr_val) / max(1.0, abs(prev_val))

        if verbose:
            print(
                f"[Epoch {epoch:03d}] f = {curr_val:.6f}, "
                f"rel_improve = {rel_improve:.3e}"
            )

        if epoch >= min_epochs:
            if rel_improve < tol:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                break

        prev_val = curr_val

    G_binary = weight_to_adjacency(A, threshold)
    return A, G_binary, curr_val, history
