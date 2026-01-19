import numpy as np
from scipy.linalg import expm


# ============================================================
# (B, Omega)-Formulation utilities
# ============================================================

def ell(B: np.ndarray, Omega: np.ndarray, S: np.ndarray, n: int = 1, eps: float = 1e-12) -> float:
    """
    Negative log-likelihood up to a constant:
        ell(B, Omega) = (n/2) * ( log det Omega
                                  - 2 log det(I - B)
                                  + tr((I - B)^T Omega^{-1} (I - B) S) )
    """
    d = B.shape[0]
    I_minus_B = np.eye(d) - B

    omega_diag = np.diag(Omega)
    if np.any(omega_diag <= 0):
        return np.inf
    logdet_Omega = float(np.sum(np.log(omega_diag + eps)))

    sign, logabsdet = np.linalg.slogdet(I_minus_B)
    if sign <= 0:
        return np.inf
    logdet_IminusB = float(logabsdet)

    Omega_inv = np.diag(1.0 / (omega_diag + eps))
    T0 = float(np.trace(I_minus_B.T @ Omega_inv @ I_minus_B @ S))

    return (n / 2.0) * (logdet_Omega - 2.0 * logdet_IminusB + T0)


from typing import Optional

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
# Closed-form Omega update
# ============================================================

def update_Omega_closed_form(
    B: np.ndarray,
    S: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Omega = diag((I - B) S (I - B)^T) with positivity floor.
    """
    d = B.shape[0]
    I_minus_B = np.eye(d) - B
    v = np.diag(I_minus_B @ S @ I_minus_B.T)
    v = np.maximum(v, eps)
    return np.diag(v)


# ============================================================
# Closed-form δ* for (B, Omega)
# ============================================================

def delta_star_BOmega(
    B: np.ndarray,
    Omega: np.ndarray,
    S: np.ndarray,
    i: int,
    j: int,
    eps: float = 1e-12
) -> float:
    if i == j:
        return 0.0

    d = B.shape[0]
    I_minus_B = np.eye(d) - B

    try:
        I_minus_B_inv = np.linalg.inv(I_minus_B)
    except np.linalg.LinAlgError:
        return 0.0

    a = float(I_minus_B_inv[j, i])

    omega_diag = np.diag(Omega)
    Omega_inv = np.diag(1.0 / (omega_diag + eps))

    m = float((Omega_inv @ I_minus_B @ S)[i, j])
    q = float(Omega_inv[i, i] * S[j, j])

    if abs(q) < 1e-18:
        return 0.0
    if abs(a) < 1e-18:
        return m / q

    Delta = (m * a + q) ** 2 - 4.0 * q * a * (m - a)
    Delta = max(Delta, 0.0)
    sqrt_D = np.sqrt(Delta)

    delta = (m * a + q - sqrt_D) / (2.0 * q * a)

    if 1.0 - delta * a <= 0:
        delta = (1.0 - 1e-12) / a

    return float(delta)


# ============================================================
# Single off-diagonal update
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
    # print(f"[step {step:05d}] Updating edge ({i}, {j})")

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
        # print(f"[step {step:05d}] delta {i}->{j}: {delta_ij:.6e}")
    else:
        pass
        # print(f"[step {step:05d}] delta {i}->{j}: infeasible (not DAG)")
    if is_DAG(B_half + Eji, tol=dag_tol, k=k):
        delta_ji = delta_star_BOmega(B_half, Omega, S, j, i)
        Delta_ji = base_val - ell(B_half + delta_ji * Eji, Omega, S) - lambda_l0
        # print(f"[step {step:05d}] delta {j}->{i}: {delta_ji:.6e}")
    else:
        pass
        # print(f"[step {step:05d}] delta {j}->{i}: infeasible (not DAG)")

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
    #print("Starting coordinate descent (B, Omega)...")

    for t in range(T):
        i, j = np.random.choice(d, 2, replace=False)
        B = update_off_diagonal_BOmega(
            B, Omega_curr, S, i, j,
            lambda_l0=lambda_l0, k=k, dag_tol=dag_tol, step=t,
            debug_list=debug_info
        )
        #print(f"[Iteration {t+1:03d}/{T}]")
        #print(f"B matrix:\n{B}")
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
    debug_info = [] # Store debug information
    prev_val = ell(B, Omega_curr, S)
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(edge_pairs)
        for (i, j) in edge_pairs:
            B = update_off_diagonal_BOmega(
                B, Omega_curr, S, i, j,
                lambda_l0=lambda_l0, k=k, dag_tol=dag_tol,
                debug_list=debug_info # Pass list to update function
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
