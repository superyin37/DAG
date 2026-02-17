import numpy as np
from scipy.linalg import expm
from typing import Optional

# ============================================================
# B-formulation utilities (Transpose-consistent)
# Model: X = B^T X + N
# ============================================================

def f_B(B: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> float:
    """
    Profiled negative log-likelihood (up to a constant):

        f(B) = sum_i log( [(I - B^T) S (I - B)]_{ii} )
               - 2 log det(I - B^T)

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

    return float(np.sum(np.log(v + eps)) - 2.0 * logabsdet)


# ============================================================
# DAG utilities (unchanged)
# ============================================================

def is_DAG(W: np.ndarray, tol: float = 1e-8, k: Optional[int] = None) -> bool:
    """
    DAG check via NOTEARS constraint:
        h(W) = tr(exp(W ∘ W)) - d
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    d = W.shape[0]

    h = np.trace(expm(W * W)) - d
    is_dag = abs(h) < tol

    if k is not None:
        edge_count = int(np.sum(np.abs(W) > tol))
        return is_dag and (edge_count <= k)
    return is_dag


def weight_to_adjacency(W: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


# ============================================================
# Closed-form δ* for B-formulation
# ============================================================

def delta_star_B(
    B: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
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
    """
    if i == j:
        return 0.0

    d = B.shape[0]
    M = np.eye(d) - B.T

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0

    a = float(M_inv[i, j])

    MS = M @ S
    m = float(MS[j, i])
    q = float(S[i, i])

    MSMT = MS @ M.T
    v = float(MSMT[j, j])

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
    debug_list: Optional[list] = None
) -> np.ndarray:
    if i == j:
        return B

    d = B.shape[0]
    B_half = B.copy()
    B_half[i, j] = 0.0
    B_half[j, i] = 0.0

    Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
    Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]

    base_val = f_B(B_half, S)

    Delta_ij, Delta_ji = -np.inf, -np.inf
    delta_ij, delta_ji = 0.0, 0.0

    if is_DAG(B_half + Eij, tol=dag_tol, k=k):
        delta_ij = delta_star_B(B_half, S, i, j)
        Delta_ij = base_val - f_B(B_half + delta_ij * Eij, S) - lambda_l0

    if is_DAG(B_half + Eji, tol=dag_tol, k=k):
        delta_ji = delta_star_B(B_half, S, j, i)
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
        return B_half + delta_ij * Eij
    else:
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
):
    np.random.seed(seed)
    d = S.shape[0]
    B = np.zeros((d, d))
    debug_info = []

    for t in range(T):
        i, j = np.random.choice(d, 2, replace=False)
        B = update_off_diagonal_B(
            B, S, i, j,
            lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
            debug_list=debug_info
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
):
    np.random.seed(seed)
    d = S.shape[0]
    B = np.zeros((d, d))

    edge_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    history = []
    debug_info = []

    prev_val = f_B(B, S)
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(edge_pairs)
        for (i, j) in edge_pairs:
            B = update_off_diagonal_B(
                B, S, i, j,
                lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
                debug_list=debug_info
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
