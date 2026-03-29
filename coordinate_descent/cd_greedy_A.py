import numpy as np
from typing import Optional, List

from coordinate0 import f, delta_star, weight_to_adjacency

# ============================================================
# Adjacency list utilities
# ============================================================

def _build_adjacency(A: np.ndarray, threshold: float = 1e-8) -> List[set]:
    """Build adjacency list from nonzero off-diagonal entries of A."""
    d = A.shape[0]
    adj = [set() for _ in range(d)]
    rows, cols = np.where((np.abs(A) > threshold) & ~np.eye(d, dtype=bool))
    for i, j in zip(rows.tolist(), cols.tolist()):
        adj[i].add(j)
    return adj


def _is_dag_kahn(adj: List[set]) -> bool:
    """
    Classic DAG check via Kahn's topological sort.  O(d + E).
    Does not rely on any invariant about the current graph.
    """
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
    """
    Check if adding directed edge i→j keeps the graph a DAG.
    Temporarily adds the edge, runs Kahn's algorithm, then removes it.  O(d + E).
    """
    if i == j:
        return False
    adj[i].add(j)
    result = _is_dag_kahn(adj)
    adj[i].discard(j)
    return result


# ============================================================
# Core rank-1 update — maintains A, A_inv, G, f_val, adj simultaneously
# ============================================================

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
    Apply the rank-1 perturbation A[i,j] += delta, updating all cached state:

      A_inv  Sherman–Morrison rank-1 update                     O(d²)
      G      outer-product Term-1 + column Term-2 update        O(d²)
      f_val  determinant-lemma + trace closed form              O(1)
      adj    add / remove edge i→j                              O(1)

    Derivation:
      After A ← A + δ eᵢeⱼᵀ:

      A_inv update (SM):
          c     = −δ / (1 + δ α),   α = A_inv[j,i]
          ΔA⁻¹  = c · outer(A_inv[:,i], A_inv[j,:])

      G = −2 A⁻ᵀ + 2 S A  update:
          Term 1  Δ(−2 A⁻ᵀ) = −2c · outer(A_inv[j,:], A_inv[:,i])    rank-1, O(d²)
          Term 2  Δ(2 S A)[:,j] = 2δ · S[:,i]                          column,  O(d)

      f = −2 log det A + tr(Aᵀ S A)  update (S symmetric):
          Δ(−2 log det) = −2 log(1 + δ α)
          Δ tr(AᵀSA)    = 2δ (SA)[i,j] + δ² S[i,i]
          (SA)[i,j] = (G[i,j] + 2 A_inv[j,i]) / 2  recoverable from cached G

    Returns True on success, False if the SM denominator is too small
    (caller should call _recompute_all as fallback).
    """
    if abs(delta) < 1e-30:
        return True

    alpha = float(A_inv[j, i])
    denom = 1.0 + delta * alpha
    if abs(denom) < 1e-15:
        return False

    # Pre-update rows/columns needed for G Term-1
    u = A_inv[:, i].copy()  # column i  (before SM update)
    v = A_inv[j, :].copy()  # row j     (before SM update)
    c = -delta / denom

    # --- f_val update: O(1) ---
    if f_state is not None:
        # Recover (SA)[i,j] from G = −2 A_inv.T + 2 SA
        sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
        delta_f = (
            -2.0 * np.log1p(delta * alpha)          # Δ(−2 log det)
            + 2.0 * delta * sa_ij                   # Δ tr, linear part
            + delta ** 2 * float(S[i, i])           # Δ tr, quadratic part
        )
        f_state[0] += delta_f

    # --- A_inv update (SM): O(d²) ---
    A_inv += c * np.outer(u, v)

    # --- G update: O(d²) + O(d) ---
    G += (-2.0 * c) * np.outer(v, u)   # Term 1: Δ(−2 A_inv.T)
    G[:, j] += 2.0 * delta * S[:, i]  # Term 2: Δ(2 SA) column j

    # --- A update: O(1) ---
    A[i, j] += delta

    # --- adj update: O(1) ---
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
    """
    Full recomputation from A when SM becomes numerically unstable.  O(d³).
    Overwrites A_inv, G, f_state[0] in-place.
    """
    A_inv[:] = np.linalg.inv(A)
    G[:] = -2.0 * A_inv.T + 2.0 * S @ A
    sign, logabsdet = np.linalg.slogdet(A)
    f_state[0] = (
        -2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))
        if sign > 0 else np.inf
    )


# ============================================================
# Greedy pair selection
# ============================================================

def _pair_scores(G: np.ndarray):
    """
    Compute greedy scores for all upper-triangle pairs.

    score(i,j) = max(|G[i,j]|, |G[j,i]|)  for i < j.

    Returns (ii, jj, scores) where ii[k], jj[k] is the k-th pair.
    """
    score_matrix = np.maximum(np.abs(G), np.abs(G.T))
    ii, jj = np.triu_indices(G.shape[0], k=1)
    return ii, jj, score_matrix[ii, jj]


# ============================================================
# Off-diagonal and diagonal updates (greedy variants)
# ============================================================

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
    Greedy off-diagonal update for the A-formulation.  Modifies all arguments in-place.

    Steps:
      1. Zero out A[i,j] and A[j,i] via rank-1 updates (updates A_inv, G, f, adj).
      2. Check DAG feasibility for i→j and j→i via DFS — O(d + E).
      3. Compute improvement Δ via the O(1) incremental formula.
      4. Apply the better feasible direction if Δ > 0.
    """
    if i == j:
        return

    # --- Step 1: zero out both entries ---
    def _zero(r, c):
        old = A[r, c]
        if old == 0.0:
            return
        ok = _rank1_update(A, A_inv, G, S, r, c, -old, f_state, adj)
        if not ok:
            A[r, c] = 0.0
            adj[r].discard(c)
            _recompute_all(A, S, A_inv, G, f_state)

    _zero(i, j)
    _zero(j, i)

    # --- Step 2: feasibility and improvement for each direction ---
    Delta_ij = Delta_ji = -np.inf
    delta_ij = delta_ji = 0.0

    if _can_add_edge(adj, i, j):
        delta_ij = delta_star(A, S, i, j, A_inv=A_inv)
        alpha = float(A_inv[j, i])
        sa_ij = (float(G[i, j]) + 2.0 * alpha) / 2.0
        df = (
            -2.0 * np.log1p(delta_ij * alpha)
            + 2.0 * delta_ij * sa_ij
            + delta_ij ** 2 * float(S[i, i])
        )
        Delta_ij = -df - lambda_l0   # improvement = −Δf − λ

    if _can_add_edge(adj, j, i):
        delta_ji = delta_star(A, S, j, i, A_inv=A_inv)
        alpha = float(A_inv[i, j])
        sa_ji = (float(G[j, i]) + 2.0 * alpha) / 2.0
        df = (
            -2.0 * np.log1p(delta_ji * alpha)
            + 2.0 * delta_ji * sa_ji
            + delta_ji ** 2 * float(S[j, j])
        )
        Delta_ji = -df - lambda_l0

    # --- Step 3: apply the better direction ---
    if Delta_ij <= 0.0 and Delta_ji <= 0.0:
        return  # neither direction improves; keep both zeroed

    def _apply(r, c, delta):
        ok = _rank1_update(A, A_inv, G, S, r, c, delta, f_state, adj)
        if not ok:
            A[r, c] += delta
            if abs(A[r, c]) > 1e-8:
                adj[r].add(c)
            else:
                adj[r].discard(c)
            _recompute_all(A, S, A_inv, G, f_state)

    if Delta_ij > Delta_ji:
        _apply(i, j, delta_ij)
    else:
        _apply(j, i, delta_ji)


def update_diagonal_greedy(
    A: np.ndarray,
    S: np.ndarray,
    i: int,
    A_inv: np.ndarray,
    G: np.ndarray,
    f_state: list,
) -> None:
    """
    Scale update for A[i,i], keeping A_inv, G, f_state in-place.
    Mirrors coordinate0.update_diagonal: reset to 0.3, then apply delta_star.
    """
    # Reset A[i,i] → 0.3 so delta_star finds the global diagonal optimum
    chg = 0.3 - A[i, i]
    if abs(chg) > 1e-30:
        ok = _rank1_update(A, A_inv, G, S, i, i, chg, f_state, adj=None)
        if not ok:
            A[i, i] = 0.3
            _recompute_all(A, S, A_inv, G, f_state)

    delta = delta_star(A, S, i, i, A_inv=A_inv)

    # Check improvement via O(1) incremental formula
    alpha = float(A_inv[i, i])
    sa_ii = (float(G[i, i]) + 2.0 * alpha) / 2.0
    df = (
        -2.0 * np.log1p(delta * alpha)
        + 2.0 * delta * sa_ii
        + delta ** 2 * float(S[i, i])
    )
    if df >= 0.0:   # f would not decrease; skip
        return

    ok = _rank1_update(A, A_inv, G, S, i, i, delta, f_state, adj=None)
    if not ok:
        A[i, i] += delta
        _recompute_all(A, S, A_inv, G, f_state)


# ============================================================
# T-step greedy coordinate descent
# ============================================================

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
    Greedy DAG coordinate descent for the A-formulation, T-step variant.

    Each step selects the pair (i,j) with the highest gradient score
        score(i,j) = max(|G[i,j]|, |G[j,i]|)
    subject to DAG feasibility (at least one direction viable after zeroing),
    then calls update_off_diagonal_greedy on that pair.

    All cached state (A_inv, G, f_val, adj) is updated incrementally;
    no O(d³) operation occurs per step after initialization.

    Parameters
    ----------
    S            Sample covariance matrix, shape (d, d).
    T            Number of iterations.
    seed         Random seed (unused here; kept for API consistency).
    threshold    Edge-weight threshold for binary adjacency extraction.
    lambda_l0    L0 regularisation penalty per edge.
    return_history  If True, also return list of f values per step.
    A_init       Optional warm-start matrix (default: identity).

    Returns
    -------
    (A, G_binary, f_val) or (A, G_binary, f_val, history)
    """
    np.random.seed(seed)
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)

    # --- O(d³) initialisation ---
    A_inv = np.linalg.inv(A)
    G = -2.0 * A_inv.T + 2.0 * S @ A
    sign, logabsdet = np.linalg.slogdet(A)
    f_state = [-2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))]
    adj = _build_adjacency(A)

    ii_all, jj_all = np.triu_indices(d, k=1)
    history = []

    for _ in range(T):
        # Recompute scores from current G — O(d²)
        score_matrix = np.maximum(np.abs(G), np.abs(G.T))
        scores = score_matrix[ii_all, jj_all].copy()

        # Greedy scan: argmax → check feasibility → mask and repeat
        # Each argmax is O(d²); k iterations total O(k·d²), k≈1 in practice
        selected = None
        while True:
            idx = int(np.argmax(scores))
            if scores[idx] < 0:   # all pairs exhausted (masked to -1)
                break
            pi, pj = int(ii_all[idx]), int(jj_all[idx])
            # Temporarily remove existing edges to simulate A_half
            has_ij = pj in adj[pi]
            has_ji = pi in adj[pj]
            if has_ij:
                adj[pi].discard(pj)
            if has_ji:
                adj[pj].discard(pi)
            feasible = _can_add_edge(adj, pi, pj) or _can_add_edge(adj, pj, pi)
            # Restore
            if has_ij:
                adj[pi].add(pj)
            if has_ji:
                adj[pj].add(pi)

            if feasible:
                selected = (pi, pj)
                break
            scores[idx] = -1   # mask and try next best

        if selected is not None:
            update_off_diagonal_greedy(
                A, S, selected[0], selected[1], lambda_l0,
                A_inv, G, f_state, adj
            )

        history.append(f_state[0])

    G_binary = weight_to_adjacency(A, threshold)
    if return_history:
        return A, G_binary, f_state[0], history
    return A, G_binary, f_state[0]


# ============================================================
# Epoch-based greedy coordinate descent
# ============================================================

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
    Greedy DAG coordinate descent for the A-formulation, epoch-based variant.

    Each epoch:
      Block 1 (structure): visit all d(d−1)/2 off-diagonal pairs in
          descending gradient-score order (sorted once per epoch).
      Block 2 (scale):     update all diagonal entries.

    A_inv, G, f_val are recomputed exactly once per epoch (O(d³)) for
    numerical stability; incremental rank-1 updates are used within the epoch.

    Parameters
    ----------
    S            Sample covariance matrix, shape (d, d).
    n_epochs     Maximum number of epochs.
    seed         Random seed (unused here; kept for API consistency).
    threshold    Edge-weight threshold for binary adjacency extraction.
    lambda_l0    L0 regularisation penalty per edge.
    tol          Relative improvement threshold for early stopping.
    patience     Number of epochs below tol before stopping.
    min_epochs   Minimum epochs before early stopping is active.
    verbose      Print per-epoch progress.
    A_init       Optional warm-start matrix (default: identity).

    Returns
    -------
    (A, G_binary, f_val, history)
    """
    np.random.seed(seed)
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)

    history = []
    no_improve = 0
    prev_val = f(A, S)

    for epoch in range(1, n_epochs + 1):
        # --- Exact recomputation at epoch start — O(d³) ---
        A_inv = np.linalg.inv(A)
        G = -2.0 * A_inv.T + 2.0 * S @ A
        sign, logabsdet = np.linalg.slogdet(A)
        f_state = [-2.0 * float(logabsdet) + float(np.trace(A.T @ S @ A))]
        adj = _build_adjacency(A)

        # --- Block 1: structure block, gradient-sorted order — O(d² log d) + O(d⁴) ---
        ii, jj, scores = _pair_scores(G)
        sorted_idx = np.argsort(scores)[::-1]

        for idx in sorted_idx:
            update_off_diagonal_greedy(
                A, S, int(ii[idx]), int(jj[idx]), lambda_l0,
                A_inv, G, f_state, adj
            )

        # --- Block 2: scale block — O(d³) total ---
        for i in range(d):
            update_diagonal_greedy(A, S, i, A_inv, G, f_state)

        # --- Epoch bookkeeping ---
        curr_val = f_state[0]
        history.append(curr_val)
        rel_improve = (prev_val - curr_val) / max(1.0, abs(prev_val))

        if verbose:
            print(
                f"[Epoch {epoch:03d}] "
                f"f = {curr_val:.6f}, "
                f"rel_improve = {rel_improve:.3e}"
            )

        # --- Early stopping ---
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
