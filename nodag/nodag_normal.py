import numpy as np
#from scipy.linalg import lu, solve_triangular, det, inv
from numpy.linalg import LinAlgError, inv
from MEC import is_in_markov_equiv_class
import SCM_data
import MEC
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(r"C:\Users\super\DAG")

# Soft-thresholding operator for matrices
def soft_threshold_matrix(A, threshold):
    return np.sign(A) * np.maximum(np.abs(A) - threshold, 0.0)

# Compute the gradient of f(A) = -2 log det A + trace(A^T R_hat A)
def compute_gradient(A, R_hat):
    epsilon = 1e-6
    try:
        A_inv = inv(A)
    except LinAlgError:
        print("Warning: singular matrix encountered. Adding epsilon * I.")
        A_inv = inv(A + epsilon * np.eye(A.shape[0]))
    return 2 * R_hat @ A - 2 * A_inv

# Objective function value
def objective(A, R_hat, lam):
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return np.inf
    trace_term = np.trace(A.T @ R_hat @ A)
    return -2 * logdet + trace_term + lam * np.sum(np.abs(A))

# Proximal gradient algorithm for matrix A
def nodag(R_hat, lam = 0.1, alpha=0.5, max_iter=100, tol=1e-5,init = None, verbose=False):
    p = R_hat.shape[0]
    if init is None:
        A = np.eye(p)  # Initialization
    else:
        A = init
    step = 1.0

    for k in range(max_iter):
        A_old = A.copy()
        grad = compute_gradient(A, R_hat)

        # Line search loop
        for _ in range(100):
            A_temp = soft_threshold_matrix(A - step * grad, step * lam)
            try:
                f_temp = objective(A_temp, R_hat, lam)
                f_curr = objective(A, R_hat, lam)
                g_temp = lam * np.sum(np.abs(A_temp))
                g_curr = lam * np.sum(np.abs(A))

                # Beck & Teboulle condition
                diff = A_temp - A
                v = np.sum((grad * diff)) + (np.linalg.norm(diff, 'fro') ** 2) / (2 * step)
                if f_temp <= f_curr + v and f_temp + g_temp <= f_curr + g_curr:
                    break
                else:
                    step *= alpha  # reduce step
            except np.linalg.LinAlgError:
                step *= alpha  # if det or inv fails, shrink step

        A = A_temp
        if verbose:
            if k % 1000 == 0:
                likelihood = -2 * np.log(np.linalg.det(A)) + np.trace(A.T @ R_hat @ A)
                print(f"Iteration {k}: likelihood = {likelihood}")

        # Convergence check
        if np.linalg.norm(A - A_old, ord='fro') < tol:
            if verbose == True:
                print(f"Iteration {k}: break")
            break
        
    likelihood = -2 * np.log(np.linalg.det(A)) + np.trace(A.T @ R_hat @ A)
    sparsity = lam * np.sum(np.abs(A))

    return A, likelihood, sparsity

def weight_to_adjacency(W, threshold=0.05):
    """
    Convert a weight matrix to an adjacency matrix.
    
    Parameters:
        W (np.ndarray): Weight matrix (square matrix).
        threshold (float): Values with absolute weight <= threshold are treated as 0.
    
    Returns:
        np.ndarray: Binary adjacency matrix of the same shape.
    """
    if not isinstance(W, np.ndarray):
        raise TypeError("Input W must be a numpy array.")
    if W.shape[0] != W.shape[1]:
        raise ValueError("Input W must be a square matrix.")
    
    G = (np.abs(W) > threshold).astype(int)
    return G