"""
Run cd_A / cd_B / cd_BOmega on manually specified B and Omega settings.

Style reference: test_all.ipynb

Usage examples:
    python experiments/test_manual_B_Omega_cd_algorithms.py
    python experiments/test_manual_B_Omega_cd_algorithms.py --n-samples-list 10000,1000,200 --n-repeats 50
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
from MEC import get_skeleton, find_v_structures
from SCM_data import generate_scm_from_BN
from coordinate_descent.coordinate0 import dag_coordinate_descent_l0_epoch as cd_A
from coordinate_descent.cd_B import dag_coordinate_descent_B_epoch as cd_B
from coordinate_descent.cd_B_Omega import dag_coordinate_descent_BOmega_epoch as cd_BOmega

try:
    from toolbox.cdt.metrics import SHD_CPDAG as cdt_shd_cpdag
except Exception:
    cdt_shd_cpdag = None


@dataclass
class TrialRow:
    n_samples: int
    experiment: str
    algorithm: str
    repeat_id: int
    seed: int
    mec_match: int
    exact_match: int
    cpdag_shd: float
    runtime_sec: float


def get_manual_experiments() -> List[Dict[str, np.ndarray]]:
    experiments: List[Dict[str, np.ndarray]] = []

    experiments.append({
        "name": "d=3, A→B←C",
        "B_true": np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 2, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 2, 3]).astype(float),
    })

    experiments.append({
        "name": "d=3, A→B→C",
        "B_true": np.array([
            [0, 1, 0],
            [0, 0, 3],
            [0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 3, 4]).astype(float),
    })

    experiments.append({
        "name": "d=3, A→B→C + A→C",
        "B_true": np.array([
            [0, 1, 2],
            [0, 0, 3],
            [0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([5, 4, 3]).astype(float),
    })

    experiments.append({
        "name": "d=4, A→B, B→C, B→D",
        "B_true": np.array([
            [0, 2, 0, 0],
            [0, 0, 3, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 4, 3, 2]).astype(float),
    })

    experiments.append({
        "name": "d=4, A→C, A→D, B→C, B→D",
        "B_true": np.array([
            [0, 0, 2, 3],
            [0, 0, 3, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([2, 4, 3, 5]).astype(float),
    })

    experiments.append({
        "name": "d=4, A→D, B→D, C→D",
        "B_true": np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 3],
            [0, 0, 0, 5],
            [0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([5, 4, 3, 2]).astype(float),
    })

    experiments.append({
        "name": "d=5, e=4, |v|=0",
        "B_true": np.array([
            [0, 1, 0, 2, 0],
            [0, 0, 3, 0, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 2, 3, 2, 1]).astype(float),
    })

    experiments.append({
        "name": "d=5, e=4, |v|=1",
        "B_true": np.array([
            [0, 0, 1, 2, 0],
            [0, 0, 0, 2, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 2, 3, 2, 1]).astype(float),
    })

    experiments.append({
        "name": "d=5, e=4, |v|=2",
        "B_true": np.array([
            [0, 0, 0, 1, 0],
            [0, 0, 0, 2, 3],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=float),
        "Omega_true": np.diag([1, 2, 3, 2, 1]).astype(float),
    })

    return experiments


def weight_to_binary_adj(W: np.ndarray, threshold: float) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


def shd_score(G_true: np.ndarray, G_est: np.ndarray) -> int:
    return int(np.sum(np.abs(G_true - G_est)))


def cpdag_shd_score(G_true: np.ndarray, G_est: np.ndarray) -> float:
    if cdt_shd_cpdag is not None:
        try:
            return float(cdt_shd_cpdag(G_true, G_est))
        except Exception:
            pass

    # Fallback without R dependency: compare MEC-defining components.
    skel_true = get_skeleton(G_true)
    skel_est = get_skeleton(G_est)
    skeleton_diff = int(np.sum(np.abs(skel_true - skel_est)) // 2)

    v_true = find_v_structures(G_true)
    v_est = find_v_structures(G_est)
    v_diff = len(v_true.symmetric_difference(v_est))

    return float(skeleton_diff + v_diff)


def run_one_experiment(
    exp: Dict[str, np.ndarray],
    n_samples: int,
    args: argparse.Namespace,
) -> List[TrialRow]:
    name = str(exp["name"])
    B_true = np.array(exp["B_true"], dtype=float)
    Omega_true = np.array(exp["Omega_true"], dtype=float)
    N = np.diag(Omega_true).copy()

    print("=" * 88)
    print(f"Running: {name}")

    data, _, _, _ = generate_scm_from_BN(
        B_true.T,
        n_samples=n_samples,
        N=N,
        seed=args.data_seed,
    )

    S = np.cov(data.T)
    G_true = weight_to_binary_adj(B_true, args.threshold)

    rows: List[TrialRow] = []

    for repeat_id in range(args.n_repeats):
        seed = args.seed + repeat_id

        t0 = time.perf_counter()
        _, G_A, _, _ = cd_A(
            S=S,
            n_epochs=args.epochs_a,
            seed=seed,
            threshold=args.threshold,
            lambda_l0=args.lambda_l0,
            tol=args.tol,
            patience=args.patience,
            min_epochs=args.min_epochs,
            verbose=False,
        )
        t1 = time.perf_counter()
        cpdag_shd = cpdag_shd_score(G_true, G_A)
        rows.append(
            TrialRow(
                n_samples=n_samples,
                experiment=name,
                algorithm="cd_A",
                repeat_id=repeat_id,
                seed=seed,
                mec_match=int(is_in_markov_equiv_class(G_true, G_A)),
                exact_match=int(np.array_equal(G_true, G_A)),
                cpdag_shd=cpdag_shd,
                runtime_sec=float(t1 - t0),
            )
        )

        t0 = time.perf_counter()
        _, G_B, _, _, _ = cd_B(
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
            verbose=False,
        )
        t1 = time.perf_counter()
        cpdag_shd = cpdag_shd_score(G_true, G_B)
        rows.append(
            TrialRow(
                n_samples=n_samples,
                experiment=name,
                algorithm="cd_B",
                repeat_id=repeat_id,
                seed=seed,
                mec_match=int(is_in_markov_equiv_class(G_true, G_B)),
                exact_match=int(np.array_equal(G_true, G_B)),
                cpdag_shd=cpdag_shd,
                runtime_sec=float(t1 - t0),
            )
        )

        t0 = time.perf_counter()
        _, G_BOmega, _, _, _ = cd_BOmega(
            S=S,
            Omega=Omega_true,
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
            verbose=False,
        )
        t1 = time.perf_counter()
        cpdag_shd = cpdag_shd_score(G_true, G_BOmega)
        rows.append(
            TrialRow(
                n_samples=n_samples,
                experiment=name,
                algorithm="cd_BOmega",
                repeat_id=repeat_id,
                seed=seed,
                mec_match=int(is_in_markov_equiv_class(G_true, G_BOmega)),
                exact_match=int(np.array_equal(G_true, G_BOmega)),
                cpdag_shd=cpdag_shd,
                runtime_sec=float(t1 - t0),
            )
        )

    for alg in ["cd_A", "cd_B", "cd_BOmega"]:
        subset = [r for r in rows if r.algorithm == alg]
        correct_rate = float(np.mean([r.mec_match for r in subset]))
        cpdag_shd_mean = float(np.mean([r.cpdag_shd for r in subset]))
        runtime_mean = float(np.mean([r.runtime_sec for r in subset]))
        print(
            f"{alg:<10} Correct rate={correct_rate:.3f} | CPDAG_SHD={cpdag_shd_mean:.2f} | "
            f"Time={runtime_mean:.3f}s"
        )

    return rows


def save_trial_csv(rows: List[TrialRow], output_path: str) -> None:
    fieldnames = list(TrialRow.__dataclass_fields__.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def save_summary_csv(rows: List[TrialRow], output_path: str) -> None:
    grouped: Dict[Tuple[int, str, str], List[TrialRow]] = {}
    for row in rows:
        key = (row.n_samples, row.experiment, row.algorithm)
        grouped.setdefault(key, []).append(row)

    fieldnames = [
        "n_samples",
        "experiment",
        "algorithm",
        "repeats",
        "mec_rate",
        "exact_rate",
        "cpdag_shd_mean",
        "cpdag_shd_std",
        "runtime_mean_sec",
        "runtime_std_sec",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (n_samples, experiment, algorithm), subset in sorted(grouped.items()):
            writer.writerow(
                {
                    "n_samples": n_samples,
                    "experiment": experiment,
                    "algorithm": algorithm,
                    "repeats": len(subset),
                    "mec_rate": float(np.mean([r.mec_match for r in subset])),
                    "exact_rate": float(np.mean([r.exact_match for r in subset])),
                    "cpdag_shd_mean": float(np.mean([r.cpdag_shd for r in subset])),
                    "cpdag_shd_std": float(np.std([r.cpdag_shd for r in subset])),
                    "runtime_mean_sec": float(np.mean([r.runtime_sec for r in subset])),
                    "runtime_std_sec": float(np.std([r.runtime_sec for r in subset])),
                }
            )


def parse_n_samples_list(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("--n-samples-list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test cd_A / cd_B / cd_BOmega on manual B and Omega settings."
    )

    parser.add_argument("--n-samples-list", type=str, default="10000,1000,200")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dag-tol", type=float, default=1e-8)

    parser.add_argument("--epochs-a", type=int, default=1000)
    parser.add_argument("--epochs-b", type=int, default=1000)
    parser.add_argument("--epochs-bomega", type=int, default=1000)

    parser.add_argument("--lambda-l0", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--eps-omega", type=float, default=1e-8)

    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "results"))
    parser.add_argument("--tag", type=str, default="manual_B_Omega_cd")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_samples_list = parse_n_samples_list(args.n_samples_list)
    os.makedirs(args.outdir, exist_ok=True)

    experiments = get_manual_experiments()
    all_rows: List[TrialRow] = []

    print("=" * 88)
    print("Manual B/Omega Benchmark: cd_A / cd_B / cd_BOmega")
    print(
        f"n_samples_list={n_samples_list}, repeats={args.n_repeats}, "
        f"threshold={args.threshold}, lambda_l0={args.lambda_l0}"
    )
    print("=" * 88)

    for n_samples in n_samples_list:
        print(f"\nN_samples = {n_samples}")
        for exp in experiments:
            rows = run_one_experiment(exp, n_samples=n_samples, args=args)
            all_rows.extend(rows)

    trial_csv = os.path.join(args.outdir, f"{args.tag}_trials.csv")
    summary_csv = os.path.join(args.outdir, f"{args.tag}_summary.csv")

    save_trial_csv(all_rows, trial_csv)
    save_summary_csv(all_rows, summary_csv)

    print("\nSaved:")
    print(f"  - {trial_csv}")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()
