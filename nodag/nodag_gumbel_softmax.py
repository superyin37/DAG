#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gumbel-Softmax (Binary-Concrete) gating with SGD optimizer.
B = g_tau(U) ⊙ P ⊙ M_off
"""

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

import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np

# ---------- utils ----------
def offdiag_mask(d, device):
    """Bool mask with True on off-diagonal, False on diagonal."""
    eye = torch.eye(d, device=device, dtype=torch.bool)
    return ~eye

def sample_binary_concrete(logits, tau=0.5, hard=False):
    """
    Binary-Concrete / Gumbel-Sigmoid sampler using Logistic noise.
    logits: real tensor
    tau: temperature
    hard: straight-through hardening if True
    """
    u = torch.rand_like(logits)
    g = torch.log(u + 1e-20) - torch.log(1 - u + 1e-20)  # Logistic(0,1)
    y = torch.sigmoid((logits + g) / tau)
    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard - y.detach() + y
    return y

# ---------- model ----------
class GumbelAdjacency(nn.Module):
    def __init__(self, d, tau=0.5, device="cpu", dtype=torch.float64, B_init=None):
        super().__init__()
        self.d = d
        self.tau = tau
        self.device = torch.device(device)

        if B_init is not None:
            # 使用用户提供的初始矩阵 B_init
            B_init = torch.tensor(B_init, dtype=dtype, device=self.device)
            assert B_init.shape == (d, d), "B_init must be a square matrix of shape (d, d)"
            # 我们可以把 B_init 分解成门控 g 和权重 P
            # 简单做法：U = logit(|B_init|)，P = B_init 的符号 * |B_init|
            # 为了稳定，这里只直接用 B_init 初始化 P，U 初始化为 sigmoid^-1(|B_init|)
            eps = 1e-6
            abs_B = torch.clamp(B_init.abs(), eps, 1-eps)
            self.U = nn.Parameter(torch.log(abs_B / (1 - abs_B)))  # logit
            self.P = nn.Parameter(B_init.clone())  # 初始权重 = B_init 本身
        else:
            # ✅ 修改后的默认初始化
            # U = 0 矩阵
            self.U = nn.Parameter(torch.zeros((d, d), dtype=dtype, device=self.device))
            # P ~ Uniform(-0.001, 0.001)
            self.P = nn.Parameter(
                (0.002 * torch.rand((d, d), dtype=dtype, device=self.device) - 0.001)
            )
            # τ 固定为 0.5
            self.tau = 0.5

        # off-diagonal mask
        self.register_buffer("M_off_bool", offdiag_mask(d, self.device))
        self.register_buffer("M_off", offdiag_mask(d, self.device).to(dtype))

    def forward(self, tau=None, hard=False, deterministic=False):
        if tau is None:
            tau = self.tau
        if deterministic:
            g = torch.sigmoid(self.U / tau)
        else:
            g = sample_binary_concrete(self.U, tau=tau, hard=hard)
        B = g * self.P
        return B, g


# ---------- objective ---------- 
def loss_fn(B, Rhat, lam=1e-2, delta=0, gates=None, return_terms=False):
    """
    L(U,P) = -2*logdet(B) + tr(B^T Rhat B) + λ * sum_offdiag(gates)
    """
    # ---- -2*logdet(B) ----
    sign, logabsdet = torch.slogdet(B + delta * torch.eye(B.shape[0], device=B.device, dtype=B.dtype))
    term_logdet = -2.0 * logabsdet


    # ---- trace term ----
    term_trace = torch.trace(B.T @ Rhat @ B)

    likelihood = term_logdet + term_trace

    # ---- penalty term ----
    penalty = 0.0
    if gates is not None:
        mask = ~torch.eye(B.shape[0], dtype=torch.bool, device=B.device)
        penalty = lam * gates[mask].sum()
        loss = likelihood + penalty
    else:
        loss = likelihood

    if return_terms:
        return loss, term_logdet, term_trace, likelihood, penalty
    return loss


# ---------- training (SGD) ----------
def train_gumbel_sgd(Rhat_np,
                     lam=1e-2,
                     delta=0,
                     max_steps=1000,
                     lr=1e-2,
                     momentum=0.9,
                     nesterov=True,
                     weight_decay=0.0,   # L2 on U/P if needed
                     tau_start=1.0,
                     tau_end=0.2,
                     hard_st=True,
                     device="cpu",
                     dtype=torch.float64,
                     seed=0,
                     history_every=100,
                     B_init = None):
    """
    SGD-based training with temperature annealing and optional ST estimator.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    Rhat = torch.tensor(Rhat_np, dtype=dtype, device=device)
    d = Rhat.shape[0]

    model = GumbelAdjacency(d, tau=tau_start, device=device, dtype=dtype, B_init = B_init)
    opt = SGD(model.parameters(), lr=lr, momentum=momentum,
              nesterov=nesterov, weight_decay=weight_decay)

    history = []

    for step in range(1, max_steps + 1):
        # linear annealing for tau
        t = (step - 1) / max(1, max_steps - 1)
        tau = tau_start + (tau_end - tau_start) * t

        opt.zero_grad()
        B, g = model(tau=tau, hard=hard_st, deterministic=False)
        loss, logdet, trace, likelihood, penalty = loss_fn(B, Rhat, lam=lam, delta=delta, gates=g, return_terms=True)
        loss.backward()
        opt.step()
        # if step == 1: 
        #     print("First loss = ", float(loss.detach().cpu()))
        #     print("First logdet = ", float(logdet.detach().cpu()))
        if step % history_every == 0 or step == 1 or step == max_steps:
            with torch.no_grad():
                B_eval, g_eval = model(tau=tau_end, hard=False, deterministic=True)
                sparsity = float(g_eval[~torch.eye(d, dtype=torch.bool, device=device)].mean().detach().cpu())
                history.append((step,
                                float(loss.detach().cpu()),
                                float(likelihood.detach().cpu()),
                                float(penalty.detach().cpu()),
                                sparsity))

    with torch.no_grad():
        B_final, g_final = model(tau=tau_end, hard=False, deterministic=True)
        final_loss, final_logdet, final_trace, final_likelihood, final_penalty = loss_fn(B_final, Rhat, lam=lam, delta=delta, gates=g_final, return_terms=True)
    
    P_final = model.P.detach().cpu().numpy()
    U_final = model.U.detach().cpu().numpy()
    G_final = weight_to_adjacency(B_final.detach().cpu().numpy(), 0.05) - np.eye(d)
    np.fill_diagonal(G_final, 0)

    return B_final.detach().cpu().numpy(), G_final, {
        "history": history,
        "final_loss": float(final_loss.detach().cpu()),
        "final_likelihood": float(final_likelihood.detach().cpu()),
        "final_penalty": float(final_penalty.detach().cpu()),
        "P_final": P_final,
        "U_final": U_final
    }

# ---------- demo ----------
if __name__ == "__main__":
    import numpy as np
    from numpy.linalg import LinAlgError, inv
    # True Graph
    n = 10000
    B_init = np.array([[1, -1, 0], [0, 1, 0], [0, -1, 1]])
    W_true = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    Omega_true = np.eye(3)
    Theta_true = (np.eye(3) - W_true) @ inv(Omega_true) @ (np.eye(3) - W_true.T)
    print("Theta_true\n", Theta_true)
    Sigma_true = inv(Theta_true)
    print("Sigma_true\n",Sigma_true)

    B_final,G_final, info = train_gumbel_sgd(
        Rhat_np = Sigma_true,
        lam = 0.5,
        delta = 1e-6,
        B_init = B_init
    )

    print("Final loss:", info["final_loss"])
    print("Finel likelihood", info["final_likelihood"])
    print("Final penalty: ", info["final_penalty"])
    print("B (final):\n", B_final)
    print("G_final:\n", G_final)
    #print("History (step, loss, avg_offdiag_gate):")
    #for h in info["history"]:
    #    print(h)