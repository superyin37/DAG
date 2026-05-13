"""DAG / CPDAG structural metrics, aligned with kaifeng-jin/CALM (paper).

Primary entry points:
    cpdag_shd(G_true, G_est)              -> float    # paper-aligned CPDAG-SHD
    shd_directed(G_true, G_est)           -> float    # directed SHD on adjacency
    skeleton_precision_recall(G_true,     -> (p, r)   # on undirected skeleton
                              G_est)
    evaluate(G_true, G_est, algorithm='') -> dict     # bundle of all the above

CPDAG-SHD backend priority (matches paper's metrics.compute_shd_cpdag):
    1. causaldag    -- ★ paper's library; identical implementation
    2. cdt.metrics.SHD_CPDAG -- different internal implementation, treated as
       a noisy approximation; can disagree with causaldag by O(1) edges
    3. manual skeleton + v-structure approximation -- known to overcount by
       roughly 2-3x relative to causaldag; emits a one-time warning when used

The DAG path also handles PDAG/CPDAG estimates (PC, FGES output): bidirectional
encoding G[i,j]=G[j,i]=1 is interpreted as an undirected edge by causaldag.

The reference paper code (kaifeng-jin/CALM, metrics.py) requires `G_est` to be
a DAG and returns None otherwise. We relax that: PDAG/CPDAG estimates are
routed through cd.PDAG.from_amat directly so PC/FGES can also be evaluated.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np


# ─── optional backends ──────────────────────────────────────────────────────

try:
    import causaldag as _cd
    _HAS_CAUSALDAG = True
except ImportError:
    _cd = None
    _HAS_CAUSALDAG = False

try:
    import igraph as _ig
    _HAS_IGRAPH = True
except ImportError:
    _ig = None
    _HAS_IGRAPH = False

try:
    import networkx as _nx
    _HAS_NETWORKX = True
except ImportError:
    _nx = None
    _HAS_NETWORKX = False

try:
    # cdt's SHD_CPDAG; may need GPUtil + R; treat as best-effort fallback.
    from cdt.metrics import SHD_CPDAG as _CDT_SHD_CPDAG
    _HAS_CDT_SHD_CPDAG = True
except Exception:
    _CDT_SHD_CPDAG = None
    _HAS_CDT_SHD_CPDAG = False


_FALLBACK_WARNED = False


def _warn_fallback_once():
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        warnings.warn(
            "dag_metrics.cpdag_shd: causaldag and cdt.SHD_CPDAG both unavailable; "
            "falling back to skeleton+v-structure approximation, which is known to "
            "overcount by roughly 2-3x relative to the paper's implementation. "
            "Install causaldag (`pip install causaldag`) for paper-aligned numbers.",
            RuntimeWarning,
        )
        _FALLBACK_WARNED = True


# ─── Algorithms whose output is a CPDAG/PDAG (not a DAG) ────────────────────
# Directed SHD is meaningless for these (undirected edges have no direction);
# CPDAG-SHD still works because we route through cd.PDAG.from_amat.
SHD_SKIP_ALGORITHMS = frozenset({'PC', 'FGES'})


# ─── small helpers (self-contained; no project imports) ─────────────────────


def _to_int01(G) -> np.ndarray:
    G = np.asarray(G)
    G_int = (np.abs(G) > 0).astype(int) if G.dtype.kind == 'f' else G.astype(int)
    return G_int


def is_dag(G) -> bool:
    """True iff `G` represents a DAG.

    Uses igraph (matching paper's metrics.py) if available; otherwise networkx.
    Bidirectional encoding (G[i,j]=G[j,i]=1) yields a 2-cycle and returns False.
    """
    G_int = _to_int01(G)
    if _HAS_IGRAPH:
        return _ig.Graph.Weighted_Adjacency(G_int.tolist()).is_dag()
    if _HAS_NETWORKX:
        return _nx.is_directed_acyclic_graph(_nx.DiGraph(G_int))
    # Last-resort: any bidirectional pair = not DAG; otherwise assume DAG.
    return not bool(np.any((G_int & G_int.T) > 0))


def _skeleton(G) -> np.ndarray:
    G_int = _to_int01(G)
    return ((G_int + G_int.T) > 0).astype(int)


def _v_structures(G) -> set:
    """Return v-structures (i -> j <- k where i and k are not adjacent)."""
    G_int = _to_int01(G)
    p = G_int.shape[0]
    out = set()
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            for k in range(p):
                if k == i or k == j:
                    continue
                if (G_int[i, j] == 1 and G_int[k, j] == 1
                        and G_int[i, k] == 0 and G_int[k, i] == 0):
                    out.add(tuple(sorted([i, k])) + (j,))
    return out


# ─── primary API ────────────────────────────────────────────────────────────


def cpdag_shd(G_true, G_est) -> float:
    """Structural Hamming Distance on CPDAGs, aligned with paper's metric.

    Paper's reference implementation (kaifeng-jin/CALM/metrics.py):
        cpdag_true = cd.DAG.from_amat(B_true).cpdag().to_amat()[0]
        cpdag_est  = cd.DAG.from_amat(B_est ).cpdag().to_amat()[0]
        return cd.PDAG.from_amat(cpdag_true).shd(cd.PDAG.from_amat(cpdag_est))

    This function reproduces that path when `causaldag` is available, with one
    extension: when `G_est` is a PDAG/CPDAG (not a DAG) -- as produced by PC
    or FGES under bidirectional encoding -- we feed it directly to
    cd.PDAG.from_amat instead of refusing. The paper's reference code returns
    None in that case.

    Parameters
    ----------
    G_true : (d, d) array-like
        Binary adjacency matrix of the true DAG.
    G_est : (d, d) array-like
        Binary adjacency of the estimate. Either a DAG or a PDAG (encoding
        undirected edges as bidirectional G[i,j]=G[j,i]=1).

    Returns
    -------
    float
        SHD on CPDAGs. NaN if `G_true` is not a DAG.
    """
    G_true_i = _to_int01(G_true)
    G_est_i  = _to_int01(G_est)

    if _HAS_CAUSALDAG:
        if not is_dag(G_true_i):
            return float('nan')
        cpdag_true = _cd.DAG.from_amat(G_true_i).cpdag()
        if is_dag(G_est_i):
            pdag_est = _cd.DAG.from_amat(G_est_i).cpdag()
        else:
            pdag_est = _cd.PDAG.from_amat(G_est_i)
        return float(cpdag_true.shd(pdag_est))

    if _HAS_CDT_SHD_CPDAG:
        try:
            return float(_CDT_SHD_CPDAG(G_true_i, G_est_i))
        except Exception:
            pass

    _warn_fallback_once()
    skel_diff = int(np.sum(np.abs(_skeleton(G_true_i) - _skeleton(G_est_i))) // 2)
    v_diff    = len(_v_structures(G_true_i).symmetric_difference(_v_structures(G_est_i)))
    return float(skel_diff + v_diff)


def shd_directed(G_true, G_est) -> float:
    """Directed SHD: count node pairs (i, j) where the directed edge encoding
    differs between true and estimate. Captures missing/extra edges and
    wrong-direction errors with a single count per pair (no double-counting).

    Note: meaningless for CPDAG/PDAG estimates with bidirectional encoding,
    since every undirected edge contributes a "wrong direction" penalty.
    Use `SHD_SKIP_ALGORITHMS` to gate this metric per algorithm.
    """
    G_true_i = _to_int01(G_true)
    G_est_i  = _to_int01(G_est)
    d = G_true_i.shape[0]
    dist = 0
    for i in range(d):
        for j in range(i + 1, d):
            if (G_true_i[i, j] != G_est_i[i, j]
                    or G_true_i[j, i] != G_est_i[j, i]):
                dist += 1
    return float(dist)


def skeleton_precision_recall(G_true, G_est) -> tuple[float, float]:
    """Precision and recall on the undirected skeleton (upper triangle)."""
    G_true_i = _to_int01(G_true)
    G_est_i  = _to_int01(G_est)
    skel_true = np.triu(_skeleton(G_true_i), k=1).astype(bool)
    skel_est  = np.triu(_skeleton(G_est_i),  k=1).astype(bool)
    tp = int(np.sum(skel_true & skel_est))
    fp = int(np.sum((~skel_true) & skel_est))
    fn = int(np.sum(skel_true & (~skel_est)))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(p), float(r)


def evaluate(G_true, G_est, algorithm: str = '') -> dict:
    """Bundle the four metrics used in the paper's tables.

    Parameters
    ----------
    G_true : (d, d) array-like
    G_est  : (d, d) array-like
    algorithm : str, optional
        Algorithm name. If listed in `SHD_SKIP_ALGORITHMS` (PC / FGES), the
        directed-SHD field is set to NaN. CPDAG-SHD is always computed (it
        handles PDAG inputs natively).

    Returns
    -------
    dict with keys: cpdag_shd, shd, sk_p, sk_r, n_edges_est
    """
    G_est_i = _to_int01(G_est).copy()
    np.fill_diagonal(G_est_i, 0)
    sk_p, sk_r = skeleton_precision_recall(G_true, G_est_i)
    shd_val = float('nan') if algorithm in SHD_SKIP_ALGORITHMS else shd_directed(G_true, G_est_i)
    return {
        'cpdag_shd':   cpdag_shd(G_true, G_est_i),
        'shd':         shd_val,
        'sk_p':        sk_p,
        'sk_r':        sk_r,
        'n_edges_est': int(G_est_i.sum()),
    }


def get_cpdag_shd_backend() -> str:
    """Return the active backend name: 'causaldag' (paper) | 'cdt' | 'fallback'."""
    if _HAS_CAUSALDAG:
        return 'causaldag'
    if _HAS_CDT_SHD_CPDAG:
        return 'cdt'
    return 'fallback'


__all__ = [
    'cpdag_shd',
    'shd_directed',
    'skeleton_precision_recall',
    'evaluate',
    'is_dag',
    'get_cpdag_shd_backend',
    'SHD_SKIP_ALGORITHMS',
]


# ─── smoke test ─────────────────────────────────────────────────────────────


def _smoke_test():
    rng = np.random.default_rng(0)
    print(f'CPDAG-SHD backend: {get_cpdag_shd_backend()}')

    # Identity must be 0
    G = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assert cpdag_shd(G, G) == 0.0, 'identity should give 0'
    print('  identity        :', cpdag_shd(G, G))

    # PDAG path: 5 edges flipped to bidirectional in a real-ish DAG
    d = 30
    p = rng.permutation(d)
    L = np.tril((rng.random((d, d)) < 0.06).astype(int), -1)
    G_true = L[np.ix_(p, p)]
    G_pdag = G_true.copy()
    edge_ij = list(zip(*np.where(G_pdag == 1)))[:5]
    for i, j in edge_ij:
        G_pdag[j, i] = 1
    print('  G_true is DAG   :', is_dag(G_true))
    print('  G_pdag is DAG   :', is_dag(G_pdag), '(expected False)')
    print('  cpdag_shd(t,t)  :', cpdag_shd(G_true, G_true))
    print('  cpdag_shd(t,p)  :', cpdag_shd(G_true, G_pdag))

    # evaluate bundle
    metrics = evaluate(G_true, G_pdag, algorithm='PC')
    print(f'  evaluate (PC)   : {metrics}')
    assert np.isnan(metrics['shd']), 'PC should skip directed SHD'

    metrics = evaluate(G_true, G_true, algorithm='cd_A')
    print(f'  evaluate (cd_A) : {metrics}')
    assert metrics['shd'] == 0.0
    assert metrics['cpdag_shd'] == 0.0

    print('OK')


if __name__ == '__main__':
    _smoke_test()
