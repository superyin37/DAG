import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(r"C:\Users\super\DAG")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from MEC import is_in_markov_equiv_class
print(os.getcwd())
from SCM_data import generate_scm_from_BN 
from numpy.linalg import inv
from scipy.linalg import sqrtm

import numpy as np
from scipy.linalg import expm


# -----------------------------
# Utility functions
# -----------------------------

def f(A, S):
    """Objective: f(A) = -2 log det(A) + tr(A^T S A)."""
    return -2 * np.log(np.linalg.det(A)) + np.trace(A.T @ S @ A)


def delta_star(A, S, i, j,eps = 1e-6):
    """
    Compute δ* = argmin_δ f(A + δ E_ij)
    following Theorem 1.
    """

    A_reg = A + eps * np.eye(A.shape[0])
    try:
        A_inv = np.linalg.inv(A_reg)
    except np.linalg.LinAlgError:
        return 0.0  
    c = S[i, i]
    b = (S @ A)[i, j]
    alpha = np.linalg.inv(A)[j, i]

    D = (c + alpha * b) ** 2 - 4 * alpha * c * (b - alpha)

    # Handle α = 0 case separately
    if abs(alpha) < 1e-12:
        return -b / c

    # Closed-form stable expression
    delta = 2 * (b - alpha) / (-(c + alpha * b) - np.sqrt(D))
    return delta


def is_DAG(W, tol=1e-8, k=None):
    """
    Acyclicity check using NOTEARS constraint:
        h(W) = tr(exp(W ∘ W)) - d = 0
        True if W represents a DAG, False otherwise
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    d = W.shape[0]

    # NOTEARS acyclicity constraint
    h = np.trace(expm(W * W)) - d
    is_dag = abs(h) < tol

    # Edge count check (nonzero entries)
    if k is not None:
        edge_count = np.sum(np.abs(W) > tol)
        return is_dag and (edge_count <= k)
    else:
        return is_dag


# def is_DAG(B):
#     """
#     Simple acyclicity check via (I + B/n)^n ≈ exp(B) test.
#     For small graphs, this brute-force method is acceptable.
#     """
#     n = B.shape[0]
#     M = np.eye(n) + B / n
#     return np.linalg.matrix_power(M, n).trace() == n  # exp(B) has trace == n if acyclic


def weight_to_adjacency(W, threshold=0.05):
    """
    Convert weighted matrix W to binary adjacency matrix G.
    Diagonal entries set to 0.
    """
    if not isinstance(W, np.ndarray):
        raise TypeError("Input W must be a numpy array.")
    if W.shape[0] != W.shape[1]:
        raise ValueError("Input W must be a square matrix.")

    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


# -----------------------------
# Main algorithm
# -----------------------------

def dag_coordinate_descent(S, T=100, seed=0, threshold=0.05, k=None):
    """
    Simplified DAG-Constrained Coordinate Descent.
    Returns (A, G, f(A))
    """
    np.random.seed(seed)
    d = S.shape[0]
    A = np.eye(d)

    for t in range(T):
        i, j = np.random.choice(d, 2, replace=True)
        
        #test
        # print(f"t = {t}, (i, j) = ({i}, {j})")
        if i == j: A[i, i] = 0.3
        else: A[i, j] = A[j, i] = 0.0        
        Δ, Δ_bar = -np.inf, -np.inf 
        
        #test
        # try direction i→j
        A_ij = A.copy()
        Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
        if is_DAG(A_ij + Eij,k=k):
            δ_t = delta_star(A, S, i, j)
            Δ = f(A, S) - f(A + δ_t * Eij, S)

        # try direction j→i
        A_ji = A.copy()
        Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]
        if is_DAG(A_ji + Eji, k=k):
            δ_bar_t = delta_star(A, S, j, i)
            Δ_bar = f(A, S) - f(A + δ_bar_t * Eji, S)

        if Δ == -np.inf and Δ_bar == -np.inf:
            # print("DAG/k constraint, continue")
            continue

        if Δ < 0 and Δ_bar < 0:
            # print("Δ & Δ_bar < 0, continue")
            continue

        # choose better direction
        if Δ > Δ_bar:
            A = A + δ_t * Eij
        else:
            A = A + δ_bar_t * Eji
        # print("A = \n", A)
    G = weight_to_adjacency(A, threshold)
    return A, G, f(A, S)
