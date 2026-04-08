import numpy as np
from typing import Optional, List

# ============================================================
# B-formulation utilities (Transpose-consistent)
# Model: X = B^T X + N
# ============================================================

def f_B(B: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> float:
    """
    Profiled negative log-likelihood (up to a constant):

        f(B) = sum_i log( [(I - B^T) S (I - B)]_{ii} )
               - 2 log det(I - B^T) + d

    consistent with X = B^T X + N.
    """
    d = B.shape[0]
    I_minus_BT = np.eye(d) - B.T
    I_minus_B = np.eye(d) - B

    # log det(I - B^T)
    sign, logabsdet = np.linalg.slogdet(I_minus_BT)
    if sign <= 0:
        return np.inf

    # diagonal residual variances
    v = np.diag(I_minus_BT @ S @ I_minus_B)
    if np.any(v <= 0):
        return np.inf

    return float(np.sum(np.log(v + eps)) - 2.0 * logabsdet) + d


# ============================================================
# DAG utilities (unchanged)
# ============================================================

def is_DAG(W: np.ndarray, tol: float = 1e-8, k: Optional[int] = None) -> bool:
    """
    DAG check via Kahn's topological sort.  O(d + E).
    """
    d = W.shape[0]
    mask = np.abs(W) > tol
    np.fill_diagonal(mask, False)

    if k is not None and int(mask.sum()) > k:
        return False

    adj: List[set] = [set() for _ in range(d)]
    rows, cols = np.where(mask)
    for i, j in zip(rows.tolist(), cols.tolist()):
        adj[i].add(j)

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


def weight_to_adjacency(W: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


# ============================================================
# Sherman–Morrison inverse update  (O(d²) rank-1 update)
# ============================================================

def _sm_update_M_inv(M_inv: np.ndarray, i: int, j: int, delta: float) -> bool:
    """
    Sherman–Morrison rank-1 update of M_inv when B[i,j] changes by *delta*.

    M = I - B^T.  When B[i,j] += delta, M becomes M - delta * e_j e_i^T.
    By Sherman–Morrison:
        M_new^{-1} = M_inv + delta / (1 - delta * M_inv[i,j])
                      * outer(M_inv[:,j], M_inv[i,:])

    Updates M_inv **in-place**.
    Returns True on success, False if the denominator is too small
    (caller should recompute the full inverse).
    """
    if abs(delta) < 1e-30:
        return True
    a = M_inv[i, j]
    denom = 1.0 - delta * a
    if abs(denom) < 1e-15:
        return False
    M_inv += (delta / denom) * np.outer(M_inv[:, j], M_inv[i, :])
    return True


# ============================================================
# Closed-form δ* for B-formulation
# ============================================================

def delta_star_B(
    B: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    M_inv: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> float:
    """
    Closed-form coordinate update for B_{ij} in B-formulation
    under X = B^T X + N.

    Definitions:
        M := I - B^T
        a := [M^{-1}]_{ij}
        v := [(M S M^T)]_{jj}
        m := (M S)_{ji}
        q := S_{ii}

    Closed-form:
        delta* = (m - a v) / (q - m a)

    If *M_inv* is supplied the lookup a = M_inv[i,j] is O(1);
    m and v are computed via the j-th row of M in O(d) and O(d²).
    """
    if i == j:
        return 0.0

    d = B.shape[0]
    if M_inv is None:
        M = np.eye(d) - B.T
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return 0.0

    a = float(M_inv[i, j])

    # Row j of M = I - B^T:  M[j, k] = δ_{jk} - B[k, j]
    M_j = -B[:, j].copy()
    M_j[j] += 1.0

    # m = (M S)_{ji} = M[j,:] @ S[:,i]  — O(d)
    m = float(M_j @ S[:, i])
    q = float(S[i, i])

    # v = (M S M^T)_{jj} = M_j @ S @ M_j  — O(d²)
    v = float((M_j @ S) @ M_j)

    denom = q - m * a
    if abs(denom) < 1e-18:
        return 0.0

    return float((m - a * v) / denom)


# ============================================================
# Single off-diagonal update
# ============================================================

def update_off_diagonal_B(
    B: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float = 0.2,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    debug_list: Optional[list] = None,
    M_inv: Optional[np.ndarray] = None,
) -> np.ndarray:
    if i == j:
        return B

    d = B.shape[0]

    # Track old values for Sherman–Morrison updates
    old_bij = B[i, j]
    old_bji = B[j, i]

    B_half = B.copy()
    B_half[i, j] = 0.0
    B_half[j, i] = 0.0

    # Update cached M_inv to reflect B_half via Sherman–Morrison  — O(d²)
    if M_inv is not None:
        ok = True
        if old_bij != 0.0:
            ok = _sm_update_M_inv(M_inv, i, j, -old_bij)
        if ok and old_bji != 0.0:
            ok = _sm_update_M_inv(M_inv, j, i, -old_bji)
        if not ok:  # fallback: recompute full inverse
            M_inv[:] = np.linalg.inv(np.eye(d) - B_half.T)

    Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
    Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]

    base_val = f_B(B_half, S)

    Delta_ij, Delta_ji = -np.inf, -np.inf
    delta_ij, delta_ji = 0.0, 0.0

    if is_DAG(B_half + Eij, tol=dag_tol, k=k):
        delta_ij = delta_star_B(B_half, S, i, j, M_inv=M_inv)
        Delta_ij = base_val - f_B(B_half + delta_ij * Eij, S) - lambda_l0

    if is_DAG(B_half + Eji, tol=dag_tol, k=k):
        delta_ji = delta_star_B(B_half, S, j, i, M_inv=M_inv)
        Delta_ji = base_val - f_B(B_half + delta_ji * Eji, S) - lambda_l0

    if debug_list is not None:
        debug_list.append({
            "i": i,
            "j": j,
            "delta_ij": float(delta_ij),
            "delta_ji": float(delta_ji),
        })

    if Delta_ij < 0 and Delta_ji < 0:
        return B_half

    if Delta_ij > Delta_ji:
        # Update M_inv for the chosen edge  — O(d²)
        if M_inv is not None:
            if not _sm_update_M_inv(M_inv, i, j, delta_ij):
                M_inv[:] = np.linalg.inv(np.eye(d) - (B_half + delta_ij * Eij).T)
        return B_half + delta_ij * Eij
    else:
        if M_inv is not None:
            if not _sm_update_M_inv(M_inv, j, i, delta_ji):
                M_inv[:] = np.linalg.inv(np.eye(d) - (B_half + delta_ji * Eji).T)
        return B_half + delta_ji * Eji


# ============================================================
# Random coordinate descent (B-formulation)
# ============================================================

def dag_coordinate_descent_B(
    S: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    B_init: Optional[np.ndarray] = None,
):
    np.random.seed(seed)
    d = S.shape[0]
    B = B_init.copy() if B_init is not None else np.zeros((d, d))
    debug_info = []

    M_inv = np.linalg.inv(np.eye(d) - B.T)

    for t in range(T):
        i, j = np.random.choice(d, 2, replace=False)
        B = update_off_diagonal_B(
            B, S, i, j,
            lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
            debug_list=debug_info, M_inv=M_inv
        )

    G = weight_to_adjacency(B, threshold)
    return B, G, f_B(B, S), debug_info


# ============================================================
# Epoch-based coordinate descent (B-formulation)
# ============================================================

def dag_coordinate_descent_B_epoch(
    S: np.ndarray,
    n_epochs: int = 500,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    tol: float = 1e-4,
    patience: int = 10,
    min_epochs: int = 50,
    verbose: bool = False,
    B_init: Optional[np.ndarray] = None,
):
    np.random.seed(seed)
    d = S.shape[0]
    B = B_init.copy() if B_init is not None else np.zeros((d, d))

    edge_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    history = []
    debug_info = []

    prev_val = f_B(B, S)
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # Recompute M_inv each epoch for numerical stability — O(d³) once
        M_inv = np.linalg.inv(np.eye(d) - B.T)

        np.random.shuffle(edge_pairs)
        for (i, j) in edge_pairs:
            B = update_off_diagonal_B(
                B, S, i, j,
                lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
                debug_list=debug_info, M_inv=M_inv
            )

        curr_val = f_B(B, S)
        history.append(curr_val)

        rel_improve = (prev_val - curr_val) / max(1.0, abs(prev_val))

        if verbose:
            print(f"[Epoch {epoch:03d}] f(B)={curr_val:.6f}, rel_improve={rel_improve:.3e}")

        if epoch >= min_epochs:
            if rel_improve < tol:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= patience:
                break

        prev_val = curr_val

    G = weight_to_adjacency(B, threshold)
    return B, G, curr_val, history, debug_info
