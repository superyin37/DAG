import numpy as np
from typing import Optional, Union

def generate_scm_data(scm_id, n_samples = 1000, seed = None):
    """
    Generate data for one of six SCMs, returning raw NumPy arrays A, B, C.
    
    scm_id:
        1: A, B, C independent
        2: A -> B, C independent
        3: A -> B <- C
        4: A -> B -> C
        5: A -> B -> C and A -> C
        6: A -> B <- C and A -> C
    n_samples: number of data points
    seed: random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    def scm1():
        # A, B, C ~ N(0, 1) independent
        G = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        CPDAG = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        A = np.random.randn(n_samples)
        B = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm2():
        # A -> B, C independent
        G = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        CPDAG = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 0]])
        A = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)  # B = A + noise
        return A, B, C, G, CPDAG

    def scm3():
        # A -> B <- C
        G = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        CPDAG = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
        A = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        B = A + C + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm4():
        # A -> B -> C
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        CPDAG = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]])
        A = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)
        C = B + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm5():
        # A -> B -> C and A -> C
        G = np.array([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        CPDAG = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
        A = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)
        C = 2*A + 3*B + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm6():
        # A -> B <- C and A -> C
        G = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
        CPDAG = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
        A = np.random.randn(n_samples)
        C = A + np.random.randn(n_samples)
        B = 2*A + 3*C + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    funcs = {1: scm1, 2: scm2, 3: scm3, 4: scm4, 5: scm5, 6: scm6}
    if scm_id not in funcs:
        raise ValueError(f"Invalid scm_id {scm_id}, must be 1–6.")

    return funcs[scm_id]()

import numpy as np

def generate_scm_from_BN(
    B: np.ndarray,
    n_samples: int,
    N: Union[np.ndarray, float, None] = None,
    *,
    seed: Optional[int] = None
):
    """
    Generate data from a given linear SCM:  X = B^T X + N,   N ~ Normal(0, Sigma)
    
    Parameters
    ----------
    B : np.ndarray
        (d, d) weight matrix, where B[i,j] represents the effect of X_j on X_i.
    n_samples : int
        Number of samples to generate.
    N : None | float | np.ndarray
        Noise dispersion parameter:
            - None: independent standard normal noise (Sigma = I)
            - scalar: same variance for all nodes (Sigma = scalar * I)
            - 1D array (d,): variance per node (Sigma = diag(N))
            - 2D array (d,d): directly treated as covariance
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, d)
        Generated data.
    G : np.ndarray, shape (d, d)
        Binary adjacency matrix derived from B (1 where |B_ij|>0, else 0).
    B : np.ndarray
        Original weight matrix.
    Sigma : np.ndarray
        Noise covariance (diagonal matrix of variances).
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=float)
    d = B.shape[0]
    assert B.shape == (d, d), "B must be square (d,d)."
    np.fill_diagonal(B, 0.0)

    # 1. Adjacency matrix G
    G = (np.abs(B) > 1e-12).astype(float)

    # 2. Construct Sigma from N
    if N is None:
        Sigma = np.eye(d)
    else:
        N = np.asarray(N, dtype=float)
        if N.ndim == 0:
            Sigma = float(N) * np.eye(d)
        elif N.ndim == 1:
            assert N.shape[0] == d, "N length must match number of variables."
            Sigma = np.diag(N)
        elif N.ndim == 2:
            assert N.shape == (d, d), "N covariance must be (d,d)."
            Sigma = 0.5 * (N + N.T)
        else:
            raise ValueError("N must be scalar, (d,), or (d,d).")

    # 3. Sample noise
    L = np.linalg.cholesky(Sigma)
    E = rng.standard_normal(size=(n_samples, d)) @ L.T

    # 4. Solve (I - B^T) X^T = E^T  ->  X = (I - B^T)^(-1) E
    M = np.eye(d) - B.T
    try:
        X = np.linalg.solve(M.T, E.T).T
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Matrix (I - B^T) is singular or not invertible.")

    return X, G, B, Sigma


