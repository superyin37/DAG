"""Benchmark NOTEARS on ER DAGs across different noise ratios.

Example:
    python experiments/scripts/run_notears_noise_ratio_benchmark.py \
        --trials 5 --d-list 30,40,50 --n-list 20000 \
        --degree 4 --noise-ratios 0.5,1.0,2.0
"""

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from MEC import is_in_markov_equiv_class, get_skeleton, find_v_structures
from calm_dataset import CalmDataset
from synthetic_dataset import SyntheticDataset


@contextmanager
def maybe_suppress_notears_logs(verbose: bool):
    if verbose:
        yield
        return

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


NOTEARS_IMPORT_ERROR = None
_previous_disable_level = logging.root.manager.disable
logging.disable(logging.INFO)
try:
    from castle.algorithms import Notears

    HAS_NOTEARS = True
except Exception as _notears_err:
    HAS_NOTEARS = False
    NOTEARS_IMPORT_ERROR = _notears_err
finally:
    logging.disable(_previous_disable_level)


try:
    TOOLBOX_ROOT = os.path.join(REPO_ROOT, "toolbox")
    if TOOLBOX_ROOT not in sys.path:
        sys.path.append(TOOLBOX_ROOT)
    from cdt.metrics import SHD_CPDAG as cdt_shd_cpdag
except Exception:
    cdt_shd_cpdag = None


@dataclass
class TrialRow:
    noise_ratio: float
    d: int
    n_samples: int
    trial_id: int
    seed: int
    algorithm: str
    data_generator: str
    status: str
    mec_match: int
    shd: float
    cpdag_shd: float
    precision: float
    recall: float
    f1: float
    n_edges_true: int
    n_edges_est: int
    runtime_sec: float
    message: str


def parse_int_list(value: str) -> List[int]:
    parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return parsed


def parse_float_list(value: str) -> List[float]:
    parsed = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("list must contain at least one float")
    return parsed


def weight_to_binary_adj(W: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    G = (np.abs(W) > threshold).astype(int)
    np.fill_diagonal(G, 0)
    return G


def shd_score(G_true: np.ndarray, G_est: np.ndarray) -> float:
    """Pairwise SHD used by the ER benchmark notebook.

    Reversing an edge counts as one error, not two.
    """
    G_true = np.asarray(G_true, dtype=int)
    G_est = np.asarray(G_est, dtype=int)
    d = G_true.shape[0]
    dist = 0
    for i in range(d):
        for j in range(i + 1, d):
            if G_true[i, j] != G_est[i, j] or G_true[j, i] != G_est[j, i]:
                dist += 1
    return float(dist)


def precision_recall_f1(G_true: np.ndarray, G_est: np.ndarray) -> Tuple[float, float, float]:
    true_edge = G_true == 1
    pred_edge = G_est == 1

    tp = int(np.sum(true_edge & pred_edge))
    fp = int(np.sum((~true_edge) & pred_edge))
    fn = int(np.sum(true_edge & (~pred_edge)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def cpdag_shd_score(G_true: np.ndarray, G_est: np.ndarray) -> float:
    if cdt_shd_cpdag is not None:
        try:
            return float(cdt_shd_cpdag(G_true.astype(int), G_est.astype(int)))
        except Exception:
            pass

    skel_true = get_skeleton(G_true)
    skel_est = get_skeleton(G_est)
    skel_diff = int(np.sum(np.abs(skel_true - skel_est)) // 2)
    v_diff = len(find_v_structures(G_true).symmetric_difference(find_v_structures(G_est)))
    return float(skel_diff + v_diff)


def make_failure_row(
    noise_ratio: float,
    d: int,
    n_samples: int,
    trial_id: int,
    seed: int,
    data_generator: str,
    message: str,
) -> TrialRow:
    return TrialRow(
        noise_ratio=noise_ratio,
        d=d,
        n_samples=n_samples,
        trial_id=trial_id,
        seed=seed,
        algorithm="notears",
        data_generator=data_generator,
        status="failed",
        mec_match=0,
        shd=np.nan,
        cpdag_shd=np.nan,
        precision=np.nan,
        recall=np.nan,
        f1=np.nan,
        n_edges_true=0,
        n_edges_est=0,
        runtime_sec=np.nan,
        message=message,
    )


def run_trial(
    noise_ratio: float,
    d: int,
    n_samples: int,
    trial_id: int,
    seed: int,
    args: argparse.Namespace,
) -> TrialRow:
    if args.data_generator == "synthetic":
        dataset = SyntheticDataset(
            n=n_samples,
            d=d,
            graph_type="ER",
            degree=args.degree,
            noise_type=args.noise_type,
            noise_ratio=noise_ratio,
            B_scale=args.b_scale,
            seed=seed,
        )
    elif args.data_generator == "calm":
        if args.noise_type == "gaussian_ev":
            dataset = CalmDataset(
                n=n_samples,
                d=d,
                graph_type="ER",
                degree=args.degree,
                sem_type="gauss",
                noise_scale=noise_ratio,
                b_scale=args.b_scale,
                seed=seed,
            )
        elif args.noise_type == "gaussian_nv":
            dataset = CalmDataset(
                n=n_samples,
                d=d,
                graph_type="ER",
                degree=args.degree,
                sem_type="gauss",
                noise_ratio=noise_ratio,
                noise_scale_mode=args.noise_scale_mode,
                b_scale=args.b_scale,
                seed=seed,
            )
        elif args.noise_type == "exponential":
            dataset = CalmDataset(
                n=n_samples,
                d=d,
                graph_type="ER",
                degree=args.degree,
                sem_type="exp",
                noise_scale=noise_ratio,
                b_scale=args.b_scale,
                seed=seed,
            )
        elif args.noise_type == "gumbel":
            dataset = CalmDataset(
                n=n_samples,
                d=d,
                graph_type="ER",
                degree=args.degree,
                sem_type="gumbel",
                noise_scale=noise_ratio,
                b_scale=args.b_scale,
                seed=seed,
            )
        else:
            raise ValueError(f"unsupported noise type {args.noise_type!r}")
    else:
        raise ValueError(f"unknown data generator {args.data_generator!r}")

    X = dataset.X
    G_true = weight_to_binary_adj(dataset.B, threshold=0.0)

    model = Notears(
        lambda1=args.notears_lambda1,
        loss_type=args.notears_loss_type,
        w_threshold=args.notears_threshold,
    )

    t0 = time.perf_counter()
    with maybe_suppress_notears_logs(args.verbose_notears):
        model.learn(X)
    runtime_sec = time.perf_counter() - t0

    G_est = np.asarray(model.causal_matrix, dtype=int)
    np.fill_diagonal(G_est, 0)

    precision, recall, f1 = precision_recall_f1(G_true, G_est)
    return TrialRow(
        noise_ratio=noise_ratio,
        d=d,
        n_samples=n_samples,
        trial_id=trial_id,
        seed=seed,
        algorithm="notears",
        data_generator=args.data_generator,
        status="ok",
        mec_match=int(is_in_markov_equiv_class(G_true, G_est)),
        shd=shd_score(G_true, G_est),
        cpdag_shd=cpdag_shd_score(G_true, G_est),
        precision=precision,
        recall=recall,
        f1=f1,
        n_edges_true=int(G_true.sum()),
        n_edges_est=int(G_est.sum()),
        runtime_sec=float(runtime_sec),
        message="",
    )


def summarize(df_trials: pd.DataFrame) -> pd.DataFrame:
    ok = df_trials[df_trials["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(["noise_ratio", "d", "n_samples", "algorithm", "data_generator"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            mec_match_mean=("mec_match", "mean"),
            shd_mean=("shd", "mean"),
            shd_std=("shd", "std"),
            cpdag_shd_mean=("cpdag_shd", "mean"),
            cpdag_shd_std=("cpdag_shd", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            n_edges_true_mean=("n_edges_true", "mean"),
            n_edges_est_mean=("n_edges_est", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
        )
        .sort_values(["d", "n_samples", "noise_ratio"])
        .reset_index(drop=True)
    )

    failures = (
        df_trials[df_trials["status"] != "ok"]
        .groupby(["noise_ratio", "d", "n_samples", "algorithm", "data_generator"], as_index=False)
        .size()
        .rename(columns={"size": "failures"})
    )
    summary = summary.merge(
        failures,
        on=["noise_ratio", "d", "n_samples", "algorithm", "data_generator"],
        how="left",
    )
    summary["failures"] = summary["failures"].fillna(0).astype(int)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NOTEARS on ER DAGs across noise_ratio values."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-list", type=parse_int_list, default=[30, 40, 50])
    parser.add_argument("--n-list", type=parse_int_list, default=[20000])
    parser.add_argument("--degree", type=float, default=4.0)
    parser.add_argument(
        "--data-generator",
        choices=["synthetic", "calm"],
        default="synthetic",
        help="Use the original SyntheticDataset generator or the CALM-compatible generator.",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian_nv",
        choices=["gaussian_ev", "gaussian_nv", "exponential", "gumbel"],
    )
    parser.add_argument(
        "--noise-ratios",
        type=parse_float_list,
        default=[0.5, 1.0, 2.0],
        help="Comma-separated noise scale multipliers.",
    )
    parser.add_argument(
        "--noise-scale-mode",
        choices=["variance", "std"],
        default="variance",
        help="For --data-generator calm and gaussian_nv, interpret noise_ratio as variance range or direct SEM std.",
    )
    parser.add_argument("--b-scale", type=float, default=1.0)
    parser.add_argument("--notears-lambda1", type=float, default=0.1)
    parser.add_argument("--notears-loss-type", type=str, default="l2")
    parser.add_argument("--notears-threshold", type=float, default=0.3)
    parser.add_argument(
        "--verbose-notears",
        action="store_true",
        help="Keep NOTEARS/Castle INFO logs enabled during model fitting.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "results"),
    )
    parser.add_argument("--tag", type=str, default="notears_noise_ratio_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HAS_NOTEARS:
        raise RuntimeError(f"NOTEARS unavailable: {NOTEARS_IMPORT_ERROR}")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: List[TrialRow] = []

    print("NOTEARS noise-ratio benchmark")
    print(f"  d_list       : {args.d_list}")
    print(f"  n_list       : {args.n_list}")
    print(f"  trials       : {args.trials}")
    print(f"  degree       : {args.degree}")
    print(f"  generator    : {args.data_generator}")
    print(f"  noise_type   : {args.noise_type}")
    print(f"  noise_ratios : {args.noise_ratios}")
    if args.data_generator == "calm" and args.noise_type == "gaussian_nv":
        print(f"  noise_scale_mode : {args.noise_scale_mode}")
    print(f"  lambda1      : {args.notears_lambda1}")
    print(f"  threshold    : {args.notears_threshold}")

    for d in args.d_list:
        for n_samples in args.n_list:
            seeds = rng.integers(0, 10**9, size=args.trials)
            for trial_idx, seed_raw in enumerate(seeds, start=1):
                seed = int(seed_raw)
                for noise_ratio in args.noise_ratios:
                    try:
                        row = run_trial(
                            noise_ratio=noise_ratio,
                            d=d,
                            n_samples=n_samples,
                            trial_id=trial_idx,
                            seed=seed,
                            args=args,
                        )
                        print(
                            f"[notears] ratio={noise_ratio:g} d={d} n={n_samples} "
                            f"trial={trial_idx} mec={row.mec_match} "
                            f"shd={row.shd:.0f} cpdag_shd={row.cpdag_shd:.0f} "
                            f"rt={row.runtime_sec:.3f}s"
                        )
                    except Exception as exc:
                        row = make_failure_row(
                            noise_ratio=noise_ratio,
                            d=d,
                            n_samples=n_samples,
                            trial_id=trial_idx,
                            seed=seed,
                            data_generator=args.data_generator,
                            message=str(exc),
                        )
                        print(
                            f"[SKIP] ratio={noise_ratio:g} d={d} n={n_samples} "
                            f"trial={trial_idx}: {exc}"
                        )
                    rows.append(row)

    df_trials = pd.DataFrame([asdict(r) for r in rows])
    df_summary = summarize(df_trials)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trials_path = os.path.join(args.outdir, f"{args.tag}_trials_{ts}.csv")
    summary_path = os.path.join(args.outdir, f"{args.tag}_summary_{ts}.csv")
    latest_trials_path = os.path.join(args.outdir, f"{args.tag}_trials.csv")
    latest_summary_path = os.path.join(args.outdir, f"{args.tag}_summary.csv")

    df_trials.to_csv(trials_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_trials.to_csv(latest_trials_path, index=False)
    df_summary.to_csv(latest_summary_path, index=False)

    print()
    print("Summary:")
    if df_summary.empty:
        print("  No successful trials.")
    else:
        print(df_summary.to_string(index=False))
    print()
    print("Saved:")
    print(f"  - {trials_path}")
    print(f"  - {summary_path}")
    print(f"  - {latest_trials_path}")
    print(f"  - {latest_summary_path}")


if __name__ == "__main__":
    main()
