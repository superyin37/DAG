import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from coordinate import dag_coordinate_descent
from MEC import is_in_markov_equiv_class
from SCM_data import generate_scm_data


def main(seed: int = 0, scm_id: int = 1, sample_size: int = 5000):
    """Run a short DAG coordinate descent demo on synthetic SCM data."""
    X, Y, Z, G_true, _ = generate_scm_data(scm_id, sample_size, seed=seed)
    data = np.stack([X, Y, Z], axis=1)
    S = data.T @ data / data.shape[0]

    A_hat, G_est, objective = dag_coordinate_descent(
        S=S,
        T=500,
        seed=seed,
        threshold=0.05,
    )
    is_match = is_in_markov_equiv_class(G_true, G_est)

    print("Demo summary")
    print(f"  SCM id: {scm_id}")
    print(f"  samples: {sample_size}")
    print(f"  objective: {objective:.4f}")
    print(f"  MEC match: {is_match}")
    print("  estimated adjacency:\n", G_est)
    print("  ground-truth adjacency:\n", G_true)


if __name__ == "__main__":
    main()
