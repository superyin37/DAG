import numpy as np
from scipy.linalg import expm
from typing import Optional

# ============================================================
# (B, Omega)-Formulation utilities  (Transpose-consistent)
# Model: X = B^T X + N,  N ~ N(0, Omega^{-1}), Omega diagonal
# ============================================================

def ell(B: np.ndarray, Omega: np.ndarray, S: np.ndarray, n: int = 1, eps: float = 1e-12) -> float:
    """
    Negative log-likelihood up to a constant (transpose-consistent with X = B^T X + N):

        ell(B, Omega) = (n/2) * ( log det Omega
                                  - 2 log det(I - B^T)
                                  + tr((I - B) Omega^{-1} (I - B^T) S) )

    Notes:
      - (I - B^T) is the linear operator mapping X to noise: (I - B^T)X = N.
      - tr((I - B) Omega^{-1} (I - B^T) S) is equivalent to
        tr((I - B^T)^T Omega^{-1} (I - B^T) S).
    """
    d = B.shape[0]

    # Core operator for the assumed SEM: M := I - B^T
    I_minus_BT = np.eye(d) - B.T

    omega_diag = np.diag(Omega)
    if np.any(omega_diag <= 0):
        return np.inf
    logdet_Omega = float(np.sum(np.log(omega_diag + eps)))

    sign, logabsdet = np.linalg.slogdet(I_minus_BT)
    if sign <= 0:
        return np.inf
    logdet_IminusBT = float(logabsdet)

    Omega_inv = np.diag(1.0 / (omega_diag + eps))

    # Trace term: tr((I-B) Omega^{-1} (I-B^T) S)
    I_minus_B = np.eye(d) - B
    T0 = float(np.trace(I_minus_B @ Omega_inv @ I_minus_BT @ S))

    return (n / 2.0) * (logdet_Omega - 2.0 * logdet_IminusBT + T0)


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
# Closed-form Omega update (transpose-consistent)
# ============================================================

def update_Omega_closed_form(
    B: np.ndarray,
    S: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    For X = B^T X + N, the residual operator is (I - B^T).
    The diagonal noise *variance* estimate (profile solution) is:

        omega_i^* = [(I - B^T) S (I - B)]_{ii}

    We set Omega = diag(max(omega^*, eps)).
    """
    d = B.shape[0]
    I_minus_BT = np.eye(d) - B.T
    I_minus_B = np.eye(d) - B
    v = np.diag(I_minus_BT @ S @ I_minus_B)
    v = np.maximum(v, eps)
    return np.diag(v)


# ============================================================
# Closed-form δ* for (B, Omega) (transpose-consistent)
# ============================================================

def delta_star_BOmega(
    B: np.ndarray,
    Omega: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    eps: float = 1e-12
) -> float:
    """
    Coordinate update for entry B_{ij} <- B_{ij} + delta under X = B^T X + N.

    Using the transpose-consistent definitions:
      M := I - B^T
      a := [M^{-1}]_{ij}
      m := [Omega^{-1} M S]_{ji}
      q := (Omega^{-1})_{ii} * S_{jj} > 0

    Closed-form:
      delta* = (m a + q - sqrt((m a + q)^2 - 4 q a (m - a))) / (2 q a)
    with feasibility: 1 - delta*a > 0.
    """
    if i == j:
        return 0.0

    d = B.shape[0]
    M = np.eye(d) - B.T  # M = I - B^T

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0

    # a := [M^{-1}]_{ij}
    a = float(M_inv[i, j])

    omega_diag = np.diag(Omega)
    Omega_inv = np.diag(1.0 / (omega_diag + eps))

    # m := [Omega^{-1} M S]_{ji}
    m = float((Omega_inv @ M @ S)[j, i])

    # q := (Omega^{-1})_{ii} S_{jj}
    q = float(Omega_inv[i, i] * S[j, j])

    if abs(q) < 1e-18:
        return 0.0
    if abs(a) < 1e-18:
        # limit a -> 0: minimize quadratic -2m delta + q delta^2
        return m / q

    Delta = (m * a + q) ** 2 - 4.0 * q * a * (m - a)
    Delta = max(Delta, 0.0)
    sqrt_D = float(np.sqrt(Delta))

    delta = (m * a + q - sqrt_D) / (2.0 * q * a)

    # Feasibility: 1 - delta*a > 0
    if 1.0 - delta * a <= 0:
        # project to boundary (strictly inside)
        delta = (1.0 - 1e-12) / a

    return float(delta)


# ============================================================
# Single off-diagonal update (unchanged control flow)
# ============================================================

def update_off_diagonal_BOmega(
    B: np.ndarray,
    Omega: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    lambda_l0: float = 0.2,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    step: Optional[int] = None,
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

    base_val = ell(B_half, Omega, S)

    Delta_ij, Delta_ji = -np.inf, -np.inf
    delta_ij, delta_ji = 0.0, 0.0

    if is_DAG(B_half + Eij, tol=dag_tol, k=k):
        delta_ij = delta_star_BOmega(B_half, Omega, S, i, j)
        Delta_ij = base_val - ell(B_half + delta_ij * Eij, Omega, S) - lambda_l0

    if is_DAG(B_half + Eji, tol=dag_tol, k=k):
        delta_ji = delta_star_BOmega(B_half, Omega, S, j, i)
        Delta_ji = base_val - ell(B_half + delta_ji * Eji, Omega, S) - lambda_l0

    if debug_list is not None:
        debug_list.append({
            "i": i,
            "j": j,
            "delta_ij": float(delta_ij),
            "delta_ji": float(delta_ji),
            "omega_diag": np.diag(Omega).copy()
        })

    if Delta_ij < 0 and Delta_ji < 0:
        return B_half

    if Delta_ij > Delta_ji:
        return B_half + delta_ij * Eij
    else:
        return B_half + delta_ji * Eji


# ============================================================
# Random coordinate descent (two-block)
# ============================================================

def dag_coordinate_descent_BOmega(
    S: np.ndarray,
    Omega: np.ndarray,
    T: int = 100,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.0,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    eps_omega: float = 1e-8,
):
    np.random.seed(seed)
    d = S.shape[0]
    B = np.zeros((d, d))
    Omega_curr = Omega.copy()
    debug_info = []

    for t in range(T):
        i, j = np.random.choice(d, 2, replace=False)
        B = update_off_diagonal_BOmega(
            B, Omega_curr, S, i, j,
            lambda_l0=lambda_l0, k=k, dag_tol=dag_tol, step=t,
            debug_list=debug_info
        )
        Omega_curr = update_Omega_closed_form(B, S, eps=eps_omega)

    G = weight_to_adjacency(B, threshold)
    return B, G, ell(B, Omega_curr, S), debug_info


# ============================================================
# Epoch-based coordinate descent (two-block)
# ============================================================

def dag_coordinate_descent_BOmega_epoch(
    S: np.ndarray,
    Omega: np.ndarray,
    n_epochs: int = 500,
    seed: int = 0,
    threshold: float = 0.05,
    lambda_l0: float = 0.2,
    k: Optional[int] = None,
    dag_tol: float = 1e-8,
    tol: float = 1e-4,
    patience: int = 10,
    min_epochs: int = 50,
    eps_omega: float = 1e-8,
    verbose: bool = False,
):
    np.random.seed(seed)
    d = S.shape[0]
    B = np.zeros((d, d))
    Omega_curr = Omega.copy()

    edge_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

    history = []
    debug_info = []
    prev_val = ell(B, Omega_curr, S)
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(edge_pairs)
        for (i, j) in edge_pairs:
            B = update_off_diagonal_BOmega(
                B, Omega_curr, S, i, j,
                lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
                debug_list=debug_info
            )

        Omega_curr = update_Omega_closed_form(B, S, eps=eps_omega)

        curr_val = ell(B, Omega_curr, S)
        history.append(curr_val)

        rel_improve = (prev_val - curr_val) / max(1.0, abs(prev_val))

        if verbose:
            print(f"[Epoch {epoch:03d}] ell={curr_val:.6f}, rel_improve={rel_improve:.3e}")

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
