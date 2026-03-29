"""
Benchmark cd_A / cd_B / cd_BOmega on automatically generated ER DAGs.

Usage example:
    python experiments/run_er_graph_cd_benchmark.py --trials 20 --d 10 --n 3000 --degree 2
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from MEC import is_in_markov_equiv_class
from synthetic_dataset import SyntheticDataset
from coordinate_descent.coordinate0 import dag_coordinate_descent_l0_epoch as cd_A
from coordinate_descent.cd_B import dag_coordinate_descent_B_epoch as cd_B
from coordinate_descent.cd_B_Omega import dag_coordinate_descent_BOmega_epoch as cd_BOmega


@dataclass
class TrialResult:
    trial_id: int
    seed: int
    algorithm: str
    objective: float
    mec_match: int
    exact_match: int
    shd: int
    precision: float
    recall: float
    f1: float
    runtime_sec: float
    n_edges_true: int
    n_edges_est: int


def weight_to_binary_adj(W: np.ndarray, threshold: float) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


def shd_score(G_true: np.ndarray, G_est: np.ndarray) -> int:
    return int(np.sum(np.abs(G_true - G_est)))


def precision_recall_f1(G_true: np.ndarray, G_est: np.ndarray) -> Tuple[float, float, float]:
    true_edge = (G_true == 1)
    pred_edge = (G_est == 1)

    tp = int(np.sum(true_edge & pred_edge))
    fp = int(np.sum((~true_edge) & pred_edge))
    fn = int(np.sum(true_edge & (~pred_edge)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def run_one_trial(
    trial_id: int,
    seed: int,
    args: argparse.Namespace,
) -> List[TrialResult]:
    dataset = SyntheticDataset(
        n=args.n,
        d=args.d,
        graph_type="ER",
        degree=args.degree,
        noise_type=args.noise_type,
        B_scale=args.b_scale,
        seed=seed,
    )

    X = dataset.X
    n, _ = X.shape
    S = X.T @ X / n

    G_true = weight_to_binary_adj(dataset.B, threshold=0.0)
    n_edges_true = int(np.sum(G_true))

    results: List[TrialResult] = []

    # ---------- cd_A ----------
    t0 = time.perf_counter()
    A_est, G_A, obj_A, _ = cd_A(
        S=S,
        n_epochs=args.epochs_a,
        seed=seed,
        threshold=args.threshold,
        lambda_l0=args.lambda_l0,
        tol=args.tol,
        patience=args.patience,
        min_epochs=args.min_epochs,
        verbose=args.verbose,
    )
    t1 = time.perf_counter()

    prec, rec, f1 = precision_recall_f1(G_true, G_A)
    results.append(
        TrialResult(
            trial_id=trial_id,
            seed=seed,
            algorithm="cd_A",
            objective=float(obj_A),
            mec_match=int(is_in_markov_equiv_class(G_true, G_A)),
            exact_match=int(np.array_equal(G_true, G_A)),
            shd=shd_score(G_true, G_A),
            precision=prec,
            recall=rec,
            f1=f1,
            runtime_sec=float(t1 - t0),
            n_edges_true=n_edges_true,
            n_edges_est=int(np.sum(G_A)),
        )
    )

    # ---------- cd_B ----------
    t0 = time.perf_counter()
    B_est, G_B, obj_B, _, _ = cd_B(
        S=S,
        n_epochs=args.epochs_b,
        seed=seed,
        threshold=args.threshold,
        lambda_l0=args.lambda_l0,
        k=args.k,
        dag_tol=args.dag_tol,
        tol=args.tol,
        patience=args.patience,
        min_epochs=args.min_epochs,
        verbose=args.verbose,
    )
    t1 = time.perf_counter()

    prec, rec, f1 = precision_recall_f1(G_true, G_B)
    results.append(
        TrialResult(
            trial_id=trial_id,
            seed=seed,
            algorithm="cd_B",
            objective=float(obj_B),
            mec_match=int(is_in_markov_equiv_class(G_true, G_B)),
            exact_match=int(np.array_equal(G_true, G_B)),
            shd=shd_score(G_true, G_B),
            precision=prec,
            recall=rec,
            f1=f1,
            runtime_sec=float(t1 - t0),
            n_edges_true=n_edges_true,
            n_edges_est=int(np.sum(G_B)),
        )
    )

    # ---------- cd_BOmega ----------
    Omega0 = np.eye(args.d)
    t0 = time.perf_counter()
    B_omega_est, G_BOmega, obj_BOmega, _, _ = cd_BOmega(
        S=S,
        Omega=Omega0,
        n_epochs=args.epochs_bomega,
        seed=seed,
        threshold=args.threshold,
        lambda_l0=args.lambda_l0,
        k=args.k,
        dag_tol=args.dag_tol,
        tol=args.tol,
        patience=args.patience,
        min_epochs=args.min_epochs,
        eps_omega=args.eps_omega,
        verbose=args.verbose,
    )
    t1 = time.perf_counter()

    prec, rec, f1 = precision_recall_f1(G_true, G_BOmega)
    results.append(
        TrialResult(
            trial_id=trial_id,
            seed=seed,
            algorithm="cd_BOmega",
            objective=float(obj_BOmega),
            mec_match=int(is_in_markov_equiv_class(G_true, G_BOmega)),
            exact_match=int(np.array_equal(G_true, G_BOmega)),
            shd=shd_score(G_true, G_BOmega),
            precision=prec,
            recall=rec,
            f1=f1,
            runtime_sec=float(t1 - t0),
            n_edges_true=n_edges_true,
            n_edges_est=int(np.sum(G_BOmega)),
        )
    )

    return results


def summarize(results: List[TrialResult]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}

    algorithms = sorted(set(r.algorithm for r in results))
    for alg in algorithms:
        subset = [r for r in results if r.algorithm == alg]

        summary[alg] = {
            "trials": float(len(subset)),
            "mec_rate": float(np.mean([r.mec_match for r in subset])),
            "exact_rate": float(np.mean([r.exact_match for r in subset])),
            "shd_mean": float(np.mean([r.shd for r in subset])),
            "shd_std": float(np.std([r.shd for r in subset])),
            "precision_mean": float(np.mean([r.precision for r in subset])),
            "recall_mean": float(np.mean([r.recall for r in subset])),
            "f1_mean": float(np.mean([r.f1 for r in subset])),
            "runtime_mean_sec": float(np.mean([r.runtime_sec for r in subset])),
            "runtime_std_sec": float(np.std([r.runtime_sec for r in subset])),
        }

    return summary


def save_trial_csv(results: List[TrialResult], output_path: str) -> None:
    fieldnames = list(TrialResult.__dataclass_fields__.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def save_summary_csv(summary: Dict[str, Dict[str, float]], output_path: str) -> None:
    all_keys = sorted({k for v in summary.values() for k in v.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm"] + all_keys)
        writer.writeheader()
        for alg, stats in summary.items():
            row = {"algorithm": alg}
            row.update(stats)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ER-graph benchmark for cd_A / cd_B / cd_BOmega")

    # Data generation
    parser.add_argument("--trials", type=int, default=20, help="Number of random ER datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--n", type=int, default=3000, help="Sample size per trial.")
    parser.add_argument("--d", type=int, default=10, help="Number of nodes.")
    parser.add_argument("--degree", type=float, default=2.0, help="Expected ER degree.")
    parser.add_argument("--noise-type", type=str, default="gaussian_ev", choices=["gaussian_ev", "gaussian_nv", "exponential", "gumbel"])
    parser.add_argument("--b-scale", type=float, default=1.0, help="Weight scale for synthetic SEM.")

    # Algorithm settings
    parser.add_argument("--threshold", type=float, default=0.05, help="Edge threshold for binarization.")
    parser.add_argument("--k", type=int, default=None, help="Optional edge-count cap in DAG check.")
    parser.add_argument("--dag-tol", type=float, default=1e-8)

    parser.add_argument("--epochs-a", type=int, default=200, help="Max epochs for cd_A (coordinate0 epoch).")
    parser.add_argument("--epochs-b", type=int, default=200, help="Max epochs for cd_B.")
    parser.add_argument("--epochs-bomega", type=int, default=200, help="Max epochs for cd_BOmega.")

    parser.add_argument("--lambda-l0", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--eps-omega", type=float, default=1e-8)

    parser.add_argument("--verbose", action="store_true", help="Print epoch logs for B/BOmega.")

    # Output
    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "results"))
    parser.add_argument("--tag", type=str, default="er_cd_benchmark")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(low=0, high=10**9, size=args.trials)

    all_results: List[TrialResult] = []

    print("=" * 88)
    print("ER-Graph Benchmark: cd_A / cd_B / cd_BOmega")
    print(f"trials={args.trials}, n={args.n}, d={args.d}, degree={args.degree}, noise={args.noise_type}")
    print("=" * 88)

    for idx, seed in enumerate(seeds, start=1):
        trial_results = run_one_trial(idx, int(seed), args)
        all_results.extend(trial_results)

        msg = []
        for r in trial_results:
            msg.append(
                f"{r.algorithm}: MEC={r.mec_match}, SHD={r.shd}, F1={r.f1:.3f}, time={r.runtime_sec:.2f}s"
            )
        print(f"[Trial {idx:03d}/{args.trials}] seed={int(seed)} | " + " | ".join(msg))

    summary = summarize(all_results)

    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    for alg in sorted(summary.keys()):
        s = summary[alg]
        print(
            f"{alg:<10} | MEC={s['mec_rate']:.3f} | Exact={s['exact_rate']:.3f} "
            f"| SHD={s['shd_mean']:.2f}±{s['shd_std']:.2f} "
            f"| F1={s['f1_mean']:.3f} | Time={s['runtime_mean_sec']:.2f}s"
        )

    trial_csv = os.path.join(args.outdir, f"{args.tag}_trials.csv")
    summary_csv = os.path.join(args.outdir, f"{args.tag}_summary.csv")
    save_trial_csv(all_results, trial_csv)
    save_summary_csv(summary, summary_csv)

    print("\nSaved:")
    print(f"  - {trial_csv}")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()
