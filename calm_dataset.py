"""CALM-compatible synthetic data generation.

This module vendors the data-generation utilities from
https://github.com/kaifeng-jin/CALM/blob/main/simulate_data.py with small
project-local wrappers.  The core functions keep CALM's global NumPy RNG
semantics so experiments can reproduce that generator independently from the
older ``synthetic_dataset.py`` path.
"""

import random
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import igraph as _igraph
except ImportError:
    _igraph = None


WeightRanges = Tuple[Tuple[float, float], ...]
NoiseScaleMode = Literal["variance", "std"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _require_igraph():
    if _igraph is None:
        raise ImportError(
            "calm_dataset requires python-igraph for CALM-compatible graph generation. "
            "Install igraph in the active environment before using CalmDataset."
        )
    return _igraph


def is_dag(W: np.ndarray) -> bool:
    ig = _require_igraph()
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
    """Simulate a random DAG using CALM's graph-generation scheme.

    Args:
        d: Number of nodes.
        s0: Expected/target number of edges. For ER and BP, igraph receives
            this as an exact edge count.
        graph_type: One of ``"ER"``, ``"SF"``, or ``"BP"``.
    """

    ig = _require_igraph()

    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError(f"unknown graph type {graph_type!r}")

    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(
    B: np.ndarray,
    w_ranges: WeightRanges = ((-2.0, -0.5), (0.5, 2.0)),
) -> np.ndarray:
    """Simulate weighted SEM parameters for a binary DAG."""
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(
    W: np.ndarray,
    n: Union[int, float],
    sem_type: str,
    noise_scale: Optional[Union[float, Sequence[float], np.ndarray]] = None,
) -> np.ndarray:
    """Simulate samples from CALM's linear SEM.

    Args:
        W: Weighted adjacency matrix of a DAG.
        n: Number of samples. ``np.inf`` is supported for linear Gaussian
            population risk, matching CALM's original utility.
        sem_type: One of ``"gauss"``, ``"exp"``, ``"gumbel"``, ``"uniform"``,
            ``"logistic"``, or ``"poisson"``.
        noise_scale: Additive-noise scale. A scalar is broadcast to all nodes;
            an array supplies one scale per node.
    """

    def _simulate_single_equation(X, w, scale):
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == "uniform":
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == "logistic":
            x = np.random.binomial(1, _sigmoid(X @ w)) * 1.0
        elif sem_type == "poisson":
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError(f"unknown sem type {sem_type!r}")
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = float(noise_scale) * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or have length d")
        scale_vec = np.asarray(noise_scale, dtype=float)

    if not is_dag(W):
        raise ValueError("W must be a DAG")

    if np.isinf(n):
        if sem_type == "gauss":
            return np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
        raise ValueError("population risk is only available for linear gauss SEM")

    ig = _require_igraph()
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([int(n), d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def make_gaussian_nv_noise_scale(
    d: int,
    noise_ratio: float,
    mode: NoiseScaleMode = "variance",
    include_endpoints: bool = True,
) -> np.ndarray:
    """Generate a node-wise Gaussian non-equal-variance noise scale vector.

    CALM's SEM function accepts standard deviations as ``noise_scale``.  The
    paper-style ``mode="variance"`` treats ``noise_ratio`` as the max variance
    in ``[1, noise_ratio]`` and returns square roots.  ``mode="std"`` returns
    the sampled values directly as standard deviations.
    """
    noise_values = np.random.uniform(1.0, noise_ratio, size=d)
    if include_endpoints and d >= 2:
        idx = np.random.choice(d, size=2, replace=False)
        noise_values[idx[0]] = 1.0
        noise_values[idx[1]] = noise_ratio

    if mode == "variance":
        return np.sqrt(noise_values)
    if mode == "std":
        return noise_values
    raise ValueError(f"unknown noise scale mode {mode!r}")


def weight_to_binary_adj(W: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


class CalmDataset:
    """Generate a CALM-compatible synthetic dataset.

    This wrapper mirrors the convenience of ``SyntheticDataset`` while keeping
    CALM's graph, weight, and SEM generation semantics.
    """

    def __init__(
        self,
        n: int,
        d: int,
        graph_type: str,
        degree: float,
        sem_type: str = "gauss",
        seed: int = 1,
        noise_scale: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        noise_ratio: Optional[float] = None,
        noise_scale_mode: NoiseScaleMode = "variance",
        b_scale: float = 1.0,
        B_scale: Optional[float] = None,
    ):
        if B_scale is not None:
            b_scale = B_scale

        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.seed = seed
        self.noise_ratio = noise_ratio
        self.noise_scale_mode = noise_scale_mode
        self.s0 = int(round(degree * d))
        self.B_ranges = (
            (b_scale * -2.0, b_scale * -0.5),
            (b_scale * 0.5, b_scale * 2.0),
        )

        set_random_seed(seed)
        self.B_bin = simulate_dag(d=d, s0=self.s0, graph_type=graph_type)
        self.B = simulate_parameter(self.B_bin, self.B_ranges)

        if noise_scale is None and noise_ratio is not None and sem_type == "gauss":
            noise_scale = make_gaussian_nv_noise_scale(d, noise_ratio, noise_scale_mode)
        self.noise_scale = noise_scale
        self.X = simulate_linear_sem(self.B, n, sem_type=sem_type, noise_scale=noise_scale)
