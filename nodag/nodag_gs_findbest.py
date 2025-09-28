import sys, os
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from nodag.nodag_gumbel_softmax import train_gumbel_sgd
from SCM_data import generate_scm_data
import numpy as np
from numpy.linalg import LinAlgError, inv
from scipy.linalg import sqrtm


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
os.chdir("/home/yin/DAG")
# print(os.getcwd())


from collections import Counter
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.SHD import SHD


def adjacency_to_dag(G: np.ndarray) -> Dag:
    """nparray to Dag class"""
    d = G.shape[0]
    nodes = [GraphNode(f"X{i}") for i in range(d)]
    dag = Dag(nodes)
    for i in range(d):
        for j in range(d):
            if G[i, j] == 1:
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                dag.add_edge(edge)
    return dag


def nodag_findbest_loss(R_hat, lam=0.5, delta=1e-6, max_steps=5000, tau_start=0.2, tau_end=0.2, times=100):
    best_loss = np.inf
    best_seed = 0
    for t in range(times):
        # if t % (times // 10) == 0:
        #     percent = int(t / times * 100)
        #     print(f"Progress: {percent}%")
        seed = t
        np.random.seed(seed) 
        B_init = np.random.randn(*R_hat.shape)
        B_final,G_final, info = train_gumbel_sgd(
            Rhat_np = R_hat,
            lam = lam,
            delta = delta,
            max_steps = max_steps,
            tau_start = tau_start,
            tau_end = tau_end,
            B_init = B_init
            )
        if info["final_loss"] < best_loss:
            best_loss = info["final_loss"]
            best_likelihood = info["final_likelihood"]
            best_penalty = info["final_penalty"]
            best_seed = seed
            best_G = G_final
            best_B = B_final
    return best_G, best_B, best_loss, best_likelihood, best_penalty, best_seed


def nodag_findbest_likelihood_penalty(
    R_hat, lam=0.5, delta=1e-6, max_steps=5000,
    tau_start=0.2, tau_end=0.2, times=100, eps=1e-3
):
    results = []
    for t in range(times):
        seed = t
        np.random.seed(seed)
        B_init = np.random.randn(*R_hat.shape)
        if t % 100 == 0: print("t = ",t)

        B_final, G_final, info = train_gumbel_sgd(
            Rhat_np=R_hat,
            lam=lam,
            delta=delta,
            max_steps=max_steps,
            tau_start=tau_start,
            tau_end=tau_end,
            B_init=B_init
        )

        results.append({
            "seed": seed,
            "G": G_final,
            "B": B_final,
            "loss": info["final_loss"],
            "likelihood": info["final_likelihood"],
            "penalty": info["final_penalty"],
        })

    min_likelihood = min(r["likelihood"] for r in results)

    candidates = [
        r for r in results
        if abs(r["likelihood"] - min_likelihood) <= eps
    ]

    best = min(candidates, key=lambda r: r["penalty"])

    return (
        best["G"], best["B"],
        best["loss"], best["likelihood"],
        best["penalty"], best["seed"]
    )

def calculate_shd(est, true):
    est_dag = adjacency_to_dag(est)
    true_dag = adjacency_to_dag(true)
    est_cpdag = dag2cpdag(est_dag)
    true_cpdag = dag2cpdag(true_dag)
    shd = SHD(true_cpdag, est_cpdag).get_shd()
    return shd, est_cpdag, true_cpdag


A = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,1,0]])
B = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]])

shd, est_cpdag, true_cpdag = calculate_shd(A,B)
print(est_cpdag)
print(true_cpdag)
print(shd)