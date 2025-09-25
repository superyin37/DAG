import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import numpy as np

# -------------- W to B ------------------
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

# ---------- utils ----------
def offdiag_mask(d, device):
    eye = torch.eye(d, device=device, dtype=torch.bool)
    return ~eye

# ---------- model: dense A ----------env:KMP_DUPLICATE_LIB_OK = "TRUE"
class DenseA(nn.Module):
    """Simple model: A is just a trainable dense matrix."""
    def __init__(self, d, A_init=None, init_scale=0.01, device="cpu", dtype=torch.float64):
        super().__init__()
        if A_init is not None:
            # use provided init, convert to torch tensor
            A0 = torch.as_tensor(A_init, dtype=dtype, device=device)
            assert A0.shape == (d, d), f"A_init must be shape {(d,d)}, got {A0.shape}"
        else:
            # fallback: random init
            A0 = init_scale * torch.randn(d, d, device=device, dtype=dtype)
        self.A = nn.Parameter(A0)

    def forward(self):
        return self.A

# # ---------- objective ----------
# @torch.no_grad()
# def _safe_logdet_spd(M):
#     # M is SPD; use Cholesky for stable logdet
#     L = torch.linalg.cholesky(M)
#     return 2.0 * torch.log(torch.diag(L)).sum()

def objective_simple(A, Rhat, lam=1e-2, beta=10.0, mask_off=None, return_parts=False):
    """
    F(A) = (-2 * logdet(A) + tr(A^T Rhat A)) + lam * sum_{i!=j} tanh(beta * a_ij^2)
    """
    d = A.shape[0]
    if mask_off is None:
        mask_off = offdiag_mask(d, A.device)

    # 第一项: -2*logdet(A)
    sign, logdetA = torch.linalg.slogdet(A)
    if torch.any(sign <= 0):
        # 如果A不可逆或行列式<=0，返回大惩罚防止NaN
        if return_parts:
            return torch.tensor(1e10, device=A.device, dtype=A.dtype), None, None
        return torch.tensor(1e10, device=A.device, dtype=A.dtype)

    term_logdet = -2.0 * logdetA

    # 第二项: tr(A^T Rhat A)
    quad = torch.trace(A.T @ Rhat @ A)

    # l0-like penalty
    penalty_raw = torch.tanh(beta * (A.abs()))[mask_off].sum()
    penalty = lam * penalty_raw

    llikelihood = term_logdet + quad
    loss = llikelihood + penalty

    if return_parts:
        return loss, llikelihood.detach().cpu().item(), penalty.detach().cpu().item()
    return loss


def nocalm(
    Rhat_np,
    lam=1e-2,
    beta=10.0,
    max_steps=1000,
    optimizer_type="adam",   # "adam" or "lbfgs"
    lr=0.05,
    history_every=50,
    device="cpu",
    dtype=torch.float64,
    seed=0,
    A_init=None, 
):
    torch.manual_seed(seed)
    device = torch.device(device)
    Rhat = torch.tensor(Rhat_np, dtype=dtype, device=device)
    d = Rhat.shape[0]
    mask_off = offdiag_mask(d, device)

    model = DenseA(d=d, A_init=A_init, device=device, dtype=dtype)

    if optimizer_type.lower() == "adam":
        opt = Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "lbfgs":
        opt = LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    else:
        raise ValueError("optimizer_type must be 'adam' or 'lbfgs'.")

    hist = []

    def closure():
        opt.zero_grad()
        A = model()
        loss = objective_simple(A, Rhat, lam=lam, beta=beta, mask_off=mask_off)
        loss.backward()
        return loss

    last_ll, last_penalty = None, None

    for step in range(1, max_steps + 1):
        if isinstance(opt, LBFGS):
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()

        if step % history_every == 0 or step == 1 or step == max_steps:
            with torch.no_grad():
                A = model()
                _, llikelihood, penalty = objective_simple(
                    A, Rhat, lam=lam, beta=beta, mask_off=mask_off, return_parts=True
                )
                sign, logdetA = torch.linalg.slogdet(A)
                logdet_val = logdetA.item() if sign > 0 else float("nan")
                hist.append(
                    (step, float(loss.detach().cpu()), llikelihood, penalty, logdet_val)
                )
                last_ll, last_penalty = llikelihood, penalty

    A_final = model().detach().cpu().numpy()
    B_final = weight_to_adjacency(A_final, 0.05) - np.eye(d)
    return A_final, B_final, {
        "history": hist,
        "final_loss": float(loss.detach().cpu()),
        "final_llikelihood": last_ll,
        "final_penalty": last_penalty,
    }


# ---------- demo ----------
if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    d, n = 6, 800
    X = np.random.randn(n, d)
    Rhat = (X.T @ X) / n + 1e-3 * np.eye(d)

    A_adam, B_adam, info = nocalm(
        Rhat,
        lam=1e-2,
        beta=10.0,
        max_steps=500,
        optimizer_type="adam",
        lr=0.03,
        history_every=100,
        dtype=torch.float64,
    )
    print("final loss:", info["final_loss"])
    print("final llikelihood:", info["final_llikelihood"])
    print("final penalty:", info["final_penalty"])
    print("Estimated B:\n", B_adam)
