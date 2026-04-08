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


def _directed_coordinate_gain(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float,
    A_inv: np.ndarray,
    G: np.ndarray,
    adj: List[set],
) -> tuple[float, float]:
    """
    Return (gain, delta_star) for the directed coordinate A[i, j].

    gain = current objective decrease after any L0 penalty.
    """
    if i == j:
        return -np.inf, 0.0

    edge_exists = j in adj[i]
    if not edge_exists and not _can_add_edge(adj, i, j):
        return -np.inf, 0.0

    delta = delta_star(A, S, i, j, A_inv=A_inv)
    alpha = float(A_inv[j, i])
    sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
    df = _incremental_df(alpha, sa_ij, float(S[i, i]), delta)
    penalty = lambda_l0 if not edge_exists else 0.0
    return -df - penalty, delta


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
    """Apply one directed off-diagonal update if it has positive gain."""
    gain, delta = _directed_coordinate_gain(A, S, i, j, lambda_l0, A_inv, G, adj)
    if gain <= 0.0:
        return False
    _apply_cached_update(A, S, i, j, delta, A_inv, G, f_state, adj)
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
):
    """Shared implementation for the T-step greedy variants."""
    np.random.seed(seed)
    d = S.shape[0]
    A, A_inv, G, f_state, adj = _init_state(S, A_init)

    ii_off, jj_off = np.where(~np.eye(d, dtype=bool))
    history = []

    for _ in range(T):
        offdiag_scores = np.abs(G[ii_off, jj_off])
        if include_diagonal:
            diag_scores = np.abs(np.diag(G))
            scores = np.concatenate([diag_scores, offdiag_scores]).copy()
        else:
            scores = offdiag_scores.copy()

        selected = None
        while True:
            idx = int(np.argmax(scores))
            score = float(scores[idx])
            if score <= 0.0:
                break

            if include_diagonal and idx < d:
                gain, _, _ = _diagonal_coordinate_gain(A, S, idx, A_inv, G)
                if gain > 0.0:
                    selected = ("diag", int(idx))
                    break
                scores[idx] = -1.0
                continue

            off_idx = idx - d if include_diagonal else idx
            i = int(ii_off[off_idx])
            j = int(jj_off[off_idx])
            gain, _ = _directed_coordinate_gain(A, S, i, j, lambda_l0, A_inv, G, adj)
            if gain > 0.0:
                selected = ("offdiag", i, j)
                break
            scores[idx] = -1.0

        if selected is None:
            break

        if selected[0] == "diag":
            update_diagonal_greedy(A, S, selected[1], A_inv, G, f_state)
        else:
            update_directed_coordinate(
                A, S, selected[1], selected[2], lambda_l0, A_inv, G, f_state, adj
            )
        history.append(f_state[0])

    G_binary = weight_to_adjacency(A, threshold)
    if return_history:
        return A, G_binary, f_state[0], history
    return A, G_binary, f_state[0]


def dag_greedy_A(
    S: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    return_history: bool = False,
    A_init: Optional[np.ndarray] = None,
):
    """
    Main T-step greedy solver.

    Uses directed coordinate selection, includes diagonal updates, and stops early
    when no coordinate yields positive objective decrease.
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
    )


def dag_greedy_A_directed(
    S: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    return_history: bool = False,
    A_init: Optional[np.ndarray] = None,
):
    """
    Directed off-diagonal-only T-step greedy solver.

    This is retained as a baseline for experiments that want the old
    directed-without-diagonal behavior.
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
