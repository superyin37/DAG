import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(r"C:\Users\super\DAG")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(os.getcwd())
from SCM_data import generate_scm_from_BN 
from numpy.linalg import inv
from scipy.linalg import sqrtm

import numpy as np



# -----------------------------
# Utility functions
# -----------------------------

def f(A, S):
    """Objective: f(A) = -2 log det(A) + tr(A^T S A)."""
    return -2 * np.log(np.linalg.det(A)) + np.trace(A.T @ S @ A)


def delta_star(A, S, i, j, A_inv=None, eps=1e-6):
    """
    Compute δ* = argmin_δ f(A + δ E_ij)
    following Theorem 1.

    If *A_inv* is supplied, alpha = A_inv[j,i] is O(1);
    otherwise computes np.linalg.inv(A) as fallback (O(d³)).
    b is computed via S[i,:] @ A[:,j] in O(d).
    """
    if A_inv is None:
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return 0.0
    c = S[i, i]
    b = float(S[i, :] @ A[:, j])  # O(d) instead of O(d³)
    alpha = float(A_inv[j, i])    # O(1) with cached inverse

    D = (c + alpha * b) ** 2 - 4 * alpha * c * (b - alpha)

    # Handle α = 0 case separately
    if abs(alpha) < 1e-12:
        return -b / c

    D = max(D, 0.0)  # numerical safety
    # Closed-form stable expression
    delta = 2 * (b - alpha) / (-(c + alpha * b) - np.sqrt(D))
    return delta


def is_DAG(W, tol=1e-8, k=None):
    """
    DAG check via Kahn's topological sort.  O(d + E).
    """
    d = W.shape[0]
    mask = np.abs(W) > tol
    np.fill_diagonal(mask, False)

    if k is not None and int(mask.sum()) > k:
        return False

    adj = [set() for _ in range(d)]
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


def _graph_snapshot(A, threshold=0.05):
    """Binary adjacency snapshot stored compactly for history logging."""
    return weight_to_adjacency(A, threshold).astype(np.uint8, copy=False)


# -----------------------------
# Sherman–Morrison inverse update  (O(d²) rank-1 update)
# -----------------------------

def _sm_update_A_inv(A_inv, i, j, delta):
    """
    Sherman–Morrison rank-1 update of A_inv when A[i,j] changes by *delta*.

    A_new = A + delta * e_i e_j^T.
    By Sherman–Morrison:
        A_inv_new = A_inv - delta / (1 + delta * A_inv[j,i])
                     * outer(A_inv[:,i], A_inv[j,:])

    Updates A_inv **in-place**.
    Returns True on success, False if the denominator is too small
    (caller should recompute the full inverse).
    """
    if abs(delta) < 1e-30:
        return True
    alpha = A_inv[j, i]
    denom = 1.0 + delta * alpha
    if abs(denom) < 1e-15:
        return False
    A_inv -= (delta / denom) * np.outer(A_inv[:, i], A_inv[j, :])
    return True


# -----------------------------
# Main algorithm
# -----------------------------

# def dag_coordinate_descent_l0(S, T=100, seed=0, threshold=0.05, k=None, lambda_l0 = 0.2):
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
            Δ = f(A, S) - f(A + δ_t * Eij, S) - lambda_l0
        # try direction j→i
        A_ji = A.copy()
        Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]
        if is_DAG(A_ji + Eji, k=k):
            δ_bar_t = delta_star(A, S, j, i)
            Δ_bar = f(A, S) - f(A + δ_bar_t * Eji, S) - lambda_l0

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


def update_diagonal(A, S, i, A_inv=None):
    d = S.shape[0]
    old_aii = A[i, i]
    A[i, i] = 0.3

    # Update A_inv for diagonal reset via Sherman–Morrison — O(d²)
    if A_inv is not None:
        chg = 0.3 - old_aii
        if abs(chg) > 1e-30:
            if not _sm_update_A_inv(A_inv, i, i, chg):
                A_inv[:] = np.linalg.inv(A)

    Eii = np.eye(d)[i][:, None] * np.eye(d)[i][None, :]
    δ = delta_star(A, S, i, i, A_inv=A_inv)
    if f(A, S) - f(A + δ * Eii, S) < 0:
        print(f"error: i=j={i}, Δ < 0")
        return A

    # Update A_inv for the accepted delta — O(d²)
    if A_inv is not None:
        if not _sm_update_A_inv(A_inv, i, i, δ):
            A_inv[:] = np.linalg.inv(A + δ * Eii)

    return A + δ * Eii


def update_off_diagonal(A, S, i, j, lambda_l0=0.2, A_inv=None):
    d = S.shape[0]
    old_aij = A[i, j]
    old_aji = A[j, i]
    A[i, j] = A[j, i] = 0.0

    # Update A_inv for zeroing via Sherman–Morrison — O(d²)
    if A_inv is not None:
        ok = True
        if old_aij != 0.0:
            ok = _sm_update_A_inv(A_inv, i, j, -old_aij)
        if ok and old_aji != 0.0:
            ok = _sm_update_A_inv(A_inv, j, i, -old_aji)
        if not ok:
            A_inv[:] = np.linalg.inv(A)

    Δ, Δ_bar = -np.inf, -np.inf
    # try direction i→j
    A_ij = A.copy()
    Eij = np.eye(d)[i][:, None] * np.eye(d)[j][None, :]
    if is_DAG(A_ij + Eij):
        δ_t = delta_star(A, S, i, j, A_inv=A_inv)
        Δ = f(A, S) - f(A + δ_t * Eij, S) - lambda_l0
    # try direction j→i
    A_ji = A.copy()
    Eji = np.eye(d)[j][:, None] * np.eye(d)[i][None, :]
    if is_DAG(A_ji + Eji):
        δ_bar_t = delta_star(A, S, j, i, A_inv=A_inv)
        Δ_bar = f(A, S) - f(A + δ_bar_t * Eji, S) - lambda_l0

    if Δ == -np.inf and Δ_bar == -np.inf:
        return A

    if Δ < 0 and Δ_bar < 0:
        return A

    # choose better direction and update A_inv — O(d²)
    if Δ > Δ_bar:
        if A_inv is not None:
            if not _sm_update_A_inv(A_inv, i, j, δ_t):
                A_inv[:] = np.linalg.inv(A + δ_t * Eij)
        A = A + δ_t * Eij
    else:
        if A_inv is not None:
            if not _sm_update_A_inv(A_inv, j, i, δ_bar_t):
                A_inv[:] = np.linalg.inv(A + δ_bar_t * Eji)
        A = A + δ_bar_t * Eji
    return A


def dag_coordinate_descent_l0(
    S,
    T=100,
    seed=0,
    threshold=0.05,
    lambda_l0=0.2,
    return_history=False,
    return_graph_history=False,
    A_init=None,
    early_stop=False,
    check_every=None,
    tol=1e-4,
    patience=10,
    min_steps=None,
):
    """
    Simplified DAG-Constrained Coordinate Descent.
    Returns:
        - (A, G, f(A)) when return_history=False (default)
        - (A, G, f(A), history) when return_history=True

    history records objective values f(A_t, S) after each iteration t.
    graph_history records binary adjacency snapshots G_t after each iteration t.

    Early stopping (early_stop=True):
        Every *check_every* steps (default d*(d+1)//2), compute relative
        improvement.  If rel_improve < tol for *patience* consecutive
        checks after *min_steps* steps, stop early.
    """
    np.random.seed(seed)
    d = S.shape[0]
    A = A_init.copy() if A_init is not None else np.eye(d)
    history = []
    graph_history_list = [] if return_graph_history else None

    A_inv = np.linalg.inv(A)

    # early stopping state
    if early_stop:
        if check_every is None:
            check_every = d * (d + 1) // 2
        if min_steps is None:
            min_steps = check_every * 10
        prev_check_f = f(A, S)
        no_improve_count = 0

    actual_T = T
    for t in range(T):
        i, j = np.random.choice(d, 2, replace=True)

        if i == j:
            A = update_diagonal(A, S, i, A_inv=A_inv)
        else:
            A = update_off_diagonal(A, S, i, j, lambda_l0, A_inv=A_inv)

        history.append(f(A, S))
        if graph_history_list is not None:
            graph_history_list.append(_graph_snapshot(A, threshold))

        # early stopping check
        if early_stop and (t + 1) % check_every == 0:
            curr_f = history[-1]
            rel_improve = (prev_check_f - curr_f) / max(1.0, abs(prev_check_f))
            prev_check_f = curr_f
            if t + 1 >= min_steps:
                if rel_improve < tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    actual_T = t + 1
                    break

    G = weight_to_adjacency(A, threshold)
    final_obj = history[-1] if len(history) > 0 else f(A, S)

    if graph_history_list is not None:
        graph_history = np.array(graph_history_list, dtype=np.uint8)
    else:
        graph_history = None

    if return_history and return_graph_history:
        return A, G, final_obj, history, graph_history
    if return_history:
        return A, G, final_obj, history
    if return_graph_history:
        return A, G, final_obj, graph_history
    return A, G, final_obj


def dag_coordinate_descent_l0_multi(
    S,
    T=100,
    seed=0,
    threshold=0.05,
    lambda_l0=0.2,
    n_restarts=10,
    return_history=False,
    A_init=None,
    early_stop=False,
    check_every=None,
    tol=1e-4,
    patience=10,
    min_steps=None,
):
    """
    Multi-restart DAG-Constrained Coordinate Descent.

    Runs dag_coordinate_descent_l0 *n_restarts* times with different seeds
    and returns the result with the lowest objective value f(A, S).

    Early stopping parameters are forwarded to each individual run.

    Returns:
        - (A, G, f(A)) when return_history=False (default)
        - (A, G, f(A), history) when return_history=True
          where history is from the best run.
    """
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31, size=n_restarts)

    best_A = None
    best_G = None
    best_obj = np.inf
    best_history = None

    for s in seeds:
        result = dag_coordinate_descent_l0(
            S,
            T=T,
            seed=int(s),
            threshold=threshold,
            lambda_l0=lambda_l0,
            return_history=return_history,
            A_init=A_init,
            early_stop=early_stop,
            check_every=check_every,
            tol=tol,
            patience=patience,
            min_steps=min_steps,
        )
        if return_history:
            A_run, G_run, obj_run, history_run = result
        else:
            A_run, G_run, obj_run = result
            history_run = None

        if obj_run < best_obj:
            best_A = A_run
            best_G = G_run
            best_obj = obj_run
            best_history = history_run

    if return_history:
        return best_A, best_G, best_obj, best_history
    return best_A, best_G, best_obj


def dag_coordinate_descent_l0_epoch(
    S,
    n_epochs=500,
    seed=0,
    threshold=0.05,
    lambda_l0=0.2,
    tol=1e-4,
    patience=10,
    min_epochs=50,
    verbose=False,
    return_graph_history=False,
    graph_history_granularity="epoch",
    A_init=None,
):
    np.random.seed(seed)
    d = S.shape[0]

    # initialization
    A = A_init.copy() if A_init is not None else np.eye(d)

    # unordered edge pairs
    edge_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    updates_per_epoch = len(edge_pairs) + d

    history = []
    if return_graph_history:
        if graph_history_granularity == "epoch":
            graph_history = np.empty((n_epochs, d, d), dtype=np.uint8)
        elif graph_history_granularity == "update":
            graph_history = np.empty((n_epochs * updates_per_epoch, d, d), dtype=np.uint8)
        else:
            raise ValueError("graph_history_granularity must be 'epoch' or 'update'.")
    else:
        graph_history = None
    no_improve_count = 0
    prev_f = f(A, S)
    curr_f = prev_f
    graph_step = 0

    for epoch in range(1, n_epochs + 1):
        # Recompute A_inv each epoch for numerical stability — O(d³) once
        A_inv = np.linalg.inv(A)

        # ============================
        # Block 1: structure block
        # ============================
        np.random.shuffle(edge_pairs)
        for (i, j) in edge_pairs:
            A = update_off_diagonal(A, S, i, j, lambda_l0, A_inv=A_inv)
            if graph_history is not None and graph_history_granularity == "update":
                graph_history[graph_step] = _graph_snapshot(A, threshold)
                graph_step += 1

        # ============================
        # Block 2: scale block
        # ============================
        for i in range(d):
            A = update_diagonal(A, S, i, A_inv=A_inv)
            if graph_history is not None and graph_history_granularity == "update":
                graph_history[graph_step] = _graph_snapshot(A, threshold)
                graph_step += 1

        # ============================
        # Epoch evaluation
        # ============================
        curr_f = f(A, S)
        history.append(curr_f)
        if graph_history is not None and graph_history_granularity == "epoch":
            graph_history[epoch - 1] = _graph_snapshot(A, threshold)

        rel_improve = (prev_f - curr_f) / max(1.0, abs(prev_f))

        if verbose:
            print(
                f"[Epoch {epoch:03d}] "
                f"f = {curr_f:.6f}, "
                f"rel_improve = {rel_improve:.3e}"
            )

        # ============================
        # Early stopping logic
        # ============================
        if epoch >= min_epochs:
            if rel_improve < tol:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                break

        prev_f = curr_f

    G = weight_to_adjacency(A, threshold)
    if graph_history is not None:
        if graph_history_granularity == "epoch":
            graph_history = graph_history[: len(history)]
        else:
            graph_history = graph_history[:graph_step]

    if return_graph_history:
        return A, G, curr_f, history, graph_history
    return A, G, curr_f, history
