"""
Test script for dag_coordinate_descent_B_epoch function from cd_B module.
Follows the structure of test_cd_B_Omega_20260118.ipynb but for the B formulation only.
"""

import sys
import os
import numpy as np

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(r"C:\Users\super\DAG")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Imports
from MEC import is_in_markov_equiv_class
from SCM_data import generate_scm_from_BN
from numpy.linalg import inv
from scipy.linalg import sqrtm
from coordinate_descent.cd_B import (
    dag_coordinate_descent_B_epoch,
    weight_to_adjacency,
    f_B
)

print(f"Working directory: {os.getcwd()}")


# ============================================================
# Define Test Experiments
# ============================================================

experiments = []

# ----------- Experiment 1 -----------
experiments.append({
    "name": "d=3, A→B←C",
    "B_true": np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 2, 0]
    ]),
    "N": np.array([1, 2, 3]),
})

# ----------- Experiment 2 -----------
experiments.append({
    "name": "d=3, A→B→C",
    "B_true": np.array([
        [0, 1, 0],
        [0, 0, 3],
        [0, 0, 0]
    ]),
    "N": np.array([1, 3, 4]),
})

# ----------- Experiment 3 -----------
experiments.append({
    "name": "d=3, A→B→C + A→C",
    "B_true": np.array([
        [0, 1, 2],
        [0, 0, 3],
        [0, 0, 0]
    ]),
    "N": np.array([5, 4, 3]),
})

# ----------- Experiment 4 -----------
experiments.append({
    "name": "d=4, A→B, B→C, B→D",
    "B_true": np.array([
        [0, 3, 0, 0],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    "N": np.array([1, 3, 3, 2]),
})

# ----------- Experiment 5 -----------
experiments.append({
    "name": "d=4, A→C, A→D, B→C, B→D",
    "B_true": np.array([
        [0, 0, 2, 3],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    "N": np.array([2, 4, 3, 5]),
})

# ----------- Experiment 6 -----------
experiments.append({
    "name": "d=4, A→D, B→D, C→D",
    "B_true": np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 3],
        [0, 0, 0, 5],
        [0, 0, 0, 0]
    ]),
    "N": np.array([5, 4, 3, 2]),
})

# ----------- Experiment 7 -----------
experiments.append({
    "name": "d=5, e=4, |v|=0",
    "B_true": np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 3, 0, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    "N": np.array([1, 2, 3, 2, 1]),
})

# ----------- Experiment 8 -----------
experiments.append({
    "name": "d=5, e=4, |v|=1",
    "B_true": np.array([
        [0, 0, 1, 2, 0],
        [0, 0, 0, 2, 3],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    "N": np.array([1, 2, 3, 2, 1]),
})

# ----------- Experiment 9 -----------
experiments.append({
    "name": "d=5, e=4, |v|=2",
    "B_true": np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 2, 3],
        [0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]),
    "N": np.array([1, 2, 3, 2, 1]),
})


# ============================================================
# Test CD-B EPOCH version on all experiments
# ============================================================

def main():
    # Setup Parameters
    n_samples = 5000
    seed = 10
    lambda_l0 = 0.0
    threshold = 0.00
    n_epochs = 500  # max number of epochs
    
    # Store results
    results_epoch = []
    
    # Loop through all experiments
    for idx, exp_config in enumerate(experiments):
        print("\n" + "=" * 80)
        print(f"=== Experiment {idx + 1} (EPOCH): {exp_config['name']} ===")
        print("=" * 80)
        
        B_true = exp_config["B_true"]
        N = exp_config["N"]
        
        # Generate Data
        data, G_true_gen, _, _ = generate_scm_from_BN(
            B_true.T,  # Transposing as per reference usage
            n_samples=n_samples,
            N=N,
            seed=seed
        )
        n, d = data.shape
        
        # Compute covariance matrix
        S = data.T @ data / n
        
        # Compute score with true parameters
        score_true = f_B(B_true, S)
        
        # Run Algorithm (EPOCH version)
        print(f"Running CD-B EPOCH with n={n_samples}...")
        # Note: returns (B, G, curr_val, history, debug_info)
        B_est, G_est, score, history, debug_info = dag_coordinate_descent_B_epoch(
            S,
            n_epochs=n_epochs,
            seed=seed,
            lambda_l0=lambda_l0,
            k=None,
            verbose=True
        )
        
        # Evaluation
        G_true = weight_to_adjacency(B_true, threshold=threshold)
        is_mec = is_in_markov_equiv_class(G_true, G_est)
        
        # Print results
        print("\nGround Truth B:\n", B_true)
        print("\nGround Truth Adjacency:\n", G_true)
        print("\nEstimated Adjacency:\n", G_est)
        print("\nEstimated B:\n", np.round(B_est, 3))
        print(f"\nScore (true params): {score_true:.4f}")
        print(f"Score (estimated): {score:.4f}")
        print(f"Score improvement: {score_true - score:.4f}")
        print(f"Convergence history: {len(history)} epochs")
        print(f"\nIs in Markov Equivalence Class: {is_mec}")
        
        # Store result
        results_epoch.append({
            'experiment': exp_config['name'],
            'is_mec': is_mec,
            'score': score,
            'score_true': score_true,
            'score_improvement': score_true - score,
            'n_epochs': len(history)
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("=== SUMMARY OF ALL EXPERIMENTS (EPOCH VERSION) ===")
    print("=" * 80)
    for i, result in enumerate(results_epoch):
        status = "✓ PASS" if result['is_mec'] else "✗ FAIL"
        improvement = result['score_improvement']
        n_epochs = result['n_epochs']
        print(f"{i+1}. {result['experiment']}: {status} | Score improvement: {improvement:.4f} | Epochs: {n_epochs}")
    
    total = len(results_epoch)
    passed = sum(1 for r in results_epoch if r['is_mec'])
    avg_improvement = np.mean([r['score_improvement'] for r in results_epoch])
    avg_epochs = np.mean([r['n_epochs'] for r in results_epoch])
    print(f"\nTotal: {passed}/{total} experiments recovered correct MEC")
    print(f"Average score improvement: {avg_improvement:.4f}")
    print(f"Average epochs to convergence: {avg_epochs:.1f}")


if __name__ == "__main__":
    main()
