"""
Benchmark runtime of cd_A / GOLEM / SP / GES on random ER graphs for multiple d.

Usage example:
    python experiments/run_er_graph_runtime_scaling.py --trials 5 --n 5000 --d-list 5,10,15,20
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from synthetic_dataset import SyntheticDataset
from coordinate_descent.coordinate0 import dag_coordinate_descent_l0_epoch as cd_A

GOLEM_IMPORT_ERROR = None
try:
    GOLEM_SRC = os.path.join(REPO_ROOT, "golemMain", "src")
    if GOLEM_SRC not in sys.path:
        sys.path.append(GOLEM_SRC)
    from golem import golem as golem_fit

    HAS_GOLEM = True
except Exception as _golem_err:
    HAS_GOLEM = False
    GOLEM_IMPORT_ERROR = _golem_err

GES_IMPORT_ERROR = None
try:
    from causallearn.search.ScoreBased.GES import ges as ges_fit

    HAS_GES = True
except Exception as _ges_err:
    HAS_GES = False
    GES_IMPORT_ERROR = _ges_err


@dataclass
class RuntimeRow:
    d: int
    trial_id: int
    seed: int
    algorithm: str
    status: str
    runtime_sec: float
    message: str


def parse_d_list(s: str) -> List[int]:
    values = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("d-list is empty")
    return values


def ges_graph_to_adj(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    d = g.shape[0]
    A = np.zeros((d, d), dtype=int)

    for i in range(d):
        for j in range(i + 1, d):
            a, b = g[i, j], g[j, i]
            if a == -1 and b == 1:
                A[i, j] = 1
            elif a == 1 and b == -1:
                A[j, i] = 1
            elif a == -1 and b == -1:
                A[i, j] = 1
                A[j, i] = 1
            elif a != 0 or b != 0:
                A[i, j] = int(a != 0)
                A[j, i] = int(b != 0)

    np.fill_diagonal(A, 0)
    return A


def sp_estimate_W(X: np.ndarray) -> np.ndarray:
    """Exhaustive SP reference implementation (practical only for small d)."""
    from itertools import permutations
    from numpy.linalg import cholesky, inv, LinAlgError

    _, p = X.shape
    if p > 8:
        raise ValueError(f"SP exhaustive search is too expensive for d={p}; please use d<=8.")

    sigma_hat = np.cov(X, rowvar=False)

    def l0_norm(U: np.ndarray, threshold: float = 0.05) -> int:
        return int(np.sum(np.abs(U) > threshold))

    best_score = np.inf
    best_W: Optional[np.ndarray] = None
    best_P: Optional[np.ndarray] = None

    for perm in permutations(range(p)):
        P = np.eye(p)[list(perm)]
        sigma_perm = P @ sigma_hat @ P.T
        try:
            theta = inv(sigma_perm)
            L = cholesky(theta)
            diag_L = np.diag(L)
            sqrt_omega = np.diag(1.0 / diag_L)
            W = np.eye(p) - L @ sqrt_omega
            score = l0_norm(W)
            if score < best_score:
                best_score = score
                best_W = W
                best_P = P
        except LinAlgError:
            continue

    if best_W is None or best_P is None:
        raise RuntimeError("SP failed to find a valid structure.")

    W_est = best_P.T @ best_W @ best_P
    np.fill_diagonal(W_est, 0.0)
    return W_est


def run_algorithm(
    algorithm: str,
    X: np.ndarray,
    S: np.ndarray,
    d: int,
    seed: int,
    args: argparse.Namespace,
) -> Tuple[str, float, str]:
    """Return (status, runtime_sec, message)."""
    t0 = time.perf_counter()

    try:
        if algorithm == "cd_A":
            cd_A(
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

        elif algorithm == "golem":
            if not HAS_GOLEM:
                return "unavailable", np.nan, str(GOLEM_IMPORT_ERROR)
            golem_fit(
                X,
                lambda_1=args.golem_lambda1,
                lambda_2=args.golem_lambda2,
                equal_variances=args.golem_equal_variances,
                num_iter=args.golem_num_iter,
                learning_rate=args.golem_learning_rate,
                seed=seed,
            )

        elif algorithm == "sp":
            if d > args.sp_max_d:
                return "skipped", np.nan, f"d={d} > sp_max_d={args.sp_max_d}"
            sp_estimate_W(X)

        elif algorithm == "ges":
            if not HAS_GES:
                return "unavailable", np.nan, str(GES_IMPORT_ERROR)
            ges_rec = ges_fit(X)
            _ = ges_graph_to_adj(ges_rec["G"].graph)

        else:
            return "failed", np.nan, f"Unknown algorithm: {algorithm}"

        t1 = time.perf_counter()
        return "ok", float(t1 - t0), ""

    except Exception as exc:
        t1 = time.perf_counter()
        return "failed", float(t1 - t0), str(exc)


def summarize(rows: List[RuntimeRow]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []

    d_values = sorted(set(r.d for r in rows))
    alg_values = sorted(set(r.algorithm for r in rows))

    for d in d_values:
        for alg in alg_values:
            subset = [r for r in rows if r.d == d and r.algorithm == alg]
            if not subset:
                continue

            ok_subset = [r for r in subset if r.status == "ok"]
            runtimes = [r.runtime_sec for r in ok_subset]

            status_counts: Dict[str, int] = {}
            for item in subset:
                status_counts[item.status] = status_counts.get(item.status, 0) + 1

            summary.append(
                {
                    "d": d,
                    "algorithm": alg,
                    "n_total": len(subset),
                    "n_ok": len(ok_subset),
                    "runtime_mean_sec": float(np.mean(runtimes)) if len(runtimes) > 0 else np.nan,
                    "runtime_std_sec": float(np.std(runtimes)) if len(runtimes) > 0 else np.nan,
                    "status_counts": str(status_counts),
                }
            )

    summary.sort(key=lambda x: (int(x["d"]), str(x["algorithm"])))
    return summary


def save_trial_csv(rows: List[RuntimeRow], output_path: str) -> None:
    fieldnames = list(RuntimeRow.__dataclass_fields__.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def save_summary_csv(summary_rows: List[Dict[str, object]], output_path: str) -> None:
    fieldnames = ["d", "algorithm", "n_total", "n_ok", "runtime_mean_sec", "runtime_std_sec", "status_counts"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ER-graph runtime scaling benchmark for cd_A / GOLEM / SP / GES")

    parser.add_argument("--trials", type=int, default=5, help="Trials per d.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--n", type=int, default=3000, help="Sample size per trial.")
    parser.add_argument("--d-list", type=str, default="5,10,15,20", help="Comma-separated d values.")
    parser.add_argument("--degree", type=float, default=2.0, help="Expected ER degree.")
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian_nv",
        choices=["gaussian_ev", "gaussian_nv", "exponential", "gumbel"],
    )
    parser.add_argument("--b-scale", type=float, default=5.0, help="Weight scale for synthetic SEM.")

    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--epochs-a", type=int, default=500)
    parser.add_argument("--lambda-l0", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--golem-num-iter", type=int, default=20000)
    parser.add_argument("--golem-learning-rate", type=float, default=1e-3)
    parser.add_argument("--golem-lambda1", type=float, default=2e-3)
    parser.add_argument("--golem-lambda2", type=float, default=5.0)
    parser.add_argument("--golem-equal-variances", action="store_true")

    parser.add_argument("--sp-max-d", type=int, default=8, help="Skip SP when d > sp_max_d.")

    parser.add_argument("--outdir", type=str, default=os.path.join("experiments", "results"))
    parser.add_argument("--tag", type=str, default="er_runtime_scaling")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_values = parse_d_list(args.d_list)
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    all_rows: List[RuntimeRow] = []
    algorithms = ["cd_A", "golem", "sp", "ges"]

    print("=" * 88)
    print("ER-Graph Runtime Scaling: cd_A / GOLEM / SP / GES")
    print(f"d_list={d_values}, trials_per_d={args.trials}, n={args.n}, degree={args.degree}, noise={args.noise_type}")
    print(f"HAS_GOLEM={HAS_GOLEM}, HAS_GES={HAS_GES}, sp_max_d={args.sp_max_d}")
    print("=" * 88)

    for d in d_values:
        seeds = rng.integers(low=0, high=10**9, size=args.trials)
        print(f"\n[d={d}] starting {args.trials} trial(s)")

        for trial_idx, seed in enumerate(seeds, start=1):
            dataset = SyntheticDataset(
                n=args.n,
                d=d,
                graph_type="ER",
                degree=args.degree,
                noise_type=args.noise_type,
                B_scale=args.b_scale,
                seed=int(seed),
            )

            X = dataset.X
            S = X.T @ X / X.shape[0]

            logs = []
            for alg in algorithms:
                status, runtime_sec, message = run_algorithm(
                    algorithm=alg,
                    X=X,
                    S=S,
                    d=d,
                    seed=int(seed),
                    args=args,
                )
                all_rows.append(
                    RuntimeRow(
                        d=d,
                        trial_id=trial_idx,
                        seed=int(seed),
                        algorithm=alg,
                        status=status,
                        runtime_sec=runtime_sec,
                        message=message,
                    )
                )

                if status == "ok":
                    logs.append(f"{alg}: {runtime_sec:.2f}s")
                else:
                    logs.append(f"{alg}: {status}")

            print(f"[d={d} | trial {trial_idx:03d}/{args.trials}] seed={int(seed)} | " + " | ".join(logs))

    summary_rows = summarize(all_rows)

    print("\n" + "=" * 88)
    print("Summary (runtime over successful runs)")
    print("=" * 88)
    for row in summary_rows:
        d = row["d"]
        alg = row["algorithm"]
        n_ok = row["n_ok"]
        n_total = row["n_total"]
        rt_mean = row["runtime_mean_sec"]
        rt_std = row["runtime_std_sec"]
        if np.isnan(rt_mean):
            print(f"d={d:<2} | {alg:<6} | ok={n_ok}/{n_total} | runtime=N/A | status={row['status_counts']}")
        else:
            print(
                f"d={d:<2} | {alg:<6} | ok={n_ok}/{n_total} "
                f"| runtime={float(rt_mean):.2f}±{float(rt_std):.2f}s | status={row['status_counts']}"
            )

    trial_csv = os.path.join(args.outdir, f"{args.tag}_trials.csv")
    summary_csv = os.path.join(args.outdir, f"{args.tag}_summary.csv")
    save_trial_csv(all_rows, trial_csv)
    save_summary_csv(summary_rows, summary_csv)

    print("\nSaved:")
    print(f"  - {trial_csv}")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()
