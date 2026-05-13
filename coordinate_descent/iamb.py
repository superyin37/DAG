"""IAMB (Incremental Association Markov Blanket) for moral-graph estimation.

Ported verbatim from kaifeng-jin/CALM (https://github.com/kaifeng-jin/CALM,
file: ``iamb.py``). Used by ``cd_A_weakfaith`` when ``screening='moral_graph'``
to produce the same moral-graph mask CALM uses for its DAG search.

The only differences from the upstream version are:

* ``causallearn`` is imported lazily so the rest of the project still works
  when it isn't installed; a clear error is raised on first use.
* Trivial type / docstring polishing.

References
----------
- Tsamardinos, Aliferis, & Statnikov (2003), Algorithms for Large Scale
  Markov Blanket Discovery (IAMB).
- Jin et al. (2024), CALM: Continuous and Acyclicity-constrained L0-penalized
  likelihood with estimated Moral graph (CALM Appendix A.1).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _require_cit():
    try:
        import causallearn.utils.cit as cit  # type: ignore
    except ImportError as e:
        raise ImportError(
            "moral_graph screening requires causallearn for Fisher-Z "
            "conditional independence tests. Install with `pip install causal-learn`."
        ) from e
    return cit


def iamb_markov_network(X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, int]:
    """Estimate the moral graph (Markov network) by running IAMB per node.

    Mirrors CALM's ``iamb.iamb_markov_network`` exactly:

    1. For each node ``i``, run IAMB to get its Markov blanket ``MB(i)``.
    2. Form the symmetric "raw" matrix where ``M[i, j] = 1`` iff ``j ∈ MB(i)``.
    3. Apply the AND rule: keep edge ``{i, j}`` only if both ``j ∈ MB(i)`` and
       ``i ∈ MB(j)``. (CALM source comment notes this is by design; AND rule
       gives stricter / more reliable moral-graph estimates than OR.)

    Parameters
    ----------
    X : (n, d) ndarray
        Raw data matrix.
    alpha : float, default 0.05
        Significance level for the Fisher-Z conditional-independence tests.
        CALM's run_experiment.py passes the main CALM ``alpha`` here, which
        defaults to 0.01.

    Returns
    -------
    markov_network : (d, d) float ndarray
        Symmetric 0/1 adjacency matrix of the estimated moral graph.
    total_num_ci : int
        Total number of conditional-independence tests performed (diagnostics).
    """
    cit = _require_cit()
    n, d = X.shape
    markov_network_raw = np.zeros((d, d))
    total_num_ci = 0
    cond_indep_test = cit.CIT(X, 'fisherz')

    for i in range(d):
        markov_blanket, num_ci = iamb(cond_indep_test, d, i, alpha)
        total_num_ci += num_ci
        if len(markov_blanket) > 0:
            markov_network_raw[i, markov_blanket] = 1
            markov_network_raw[markov_blanket, i] = 1

    # AND rule: edge (i, j) only if both endpoints have the other in their MB.
    markov_network = np.logical_and(
        markov_network_raw, markov_network_raw.T
    ).astype(float)
    return markov_network, total_num_ci


def iamb(cond_indep_test, d: int, target: int, alpha: float) -> Tuple[List[int], int]:
    """Single-node IAMB: estimate the Markov blanket of ``target``.

    Two phases per CALM's implementation:

    - Forward growth: repeatedly add the variable with smallest p-value among
      those still dependent given the current MB; stop when none are left
      below ``alpha``.
    - Backward shrink: remove any variable now independent of ``target`` given
      the rest of the MB.
    """
    markov_blanket: List[int] = []
    num_ci = 0

    # Forward growth phase
    circulate_flag = True
    while circulate_flag:
        circulate_flag = False
        min_pval = float('inf')
        y = None
        variables = [i for i in range(d) if i != target and i not in markov_blanket]
        for x in variables:
            num_ci += 1
            pval = cond_indep_test(target, x, markov_blanket)
            if pval <= alpha and pval < min_pval:
                min_pval = pval
                y = x
        if y is not None:
            markov_blanket.append(y)
            circulate_flag = True

    # Backward shrink phase
    markov_blanket_temp = markov_blanket.copy()
    for x in markov_blanket_temp:
        condition_variables = [i for i in markov_blanket if i != x]
        num_ci += 1
        pval = cond_indep_test(target, x, condition_variables)
        if pval > alpha:
            markov_blanket.remove(x)

    return list(set(markov_blanket)), num_ci


__all__ = ['iamb_markov_network', 'iamb']
