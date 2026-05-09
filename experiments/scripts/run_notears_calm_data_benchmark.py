"""Benchmark NOTEARS using the CALM simulate_data.py data-generation scheme.

The generator mirrors https://github.com/kaifeng-jin/CALM/blob/main/simulate_data.py
for ER DAGs, linear Gaussian SEMs, and edge-weight sampling.  It is intentionally
separate from ``synthetic_dataset.py`` so existing experiments remain unchanged.
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

from MEC import find_v_structures, get_skeleton, is_in_markov_equiv_class
from calm_dataset import (
    make_gaussian_nv_noise_scale,
    set_random_seed,
    simulate_dag,
    simulate_linear_sem,
    simulate_parameter,
    weight_to_binary_adj,
)


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
    degree: float
    trial_id: int
    seed: int
    algorithm: str
    status: str
    mec_match: int
    shd: float
    cpdag_shd: float
    directed_precision: float
    directed_recall: float
    directed_f1: float
    skeleton_precision: float
    skeleton_recall: float
    n_edges_true: int
    n_edges_est: int
    noise_scale_min: float
    noise_scale_max: float
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


def make_noise_scales(d: int, noise_ratio: float, mode: str) -> np.ndarray:
    """Generate the NV noise vector used by the CALM paper.

    The paper describes node noise variances in [1, noise_ratio].  Since
    ``simulate_linear_sem`` takes normal standard deviations as ``noise_scale``,
    ``mode='variance'`` passes sqrt(variance).  ``mode='std'`` is available to
    reproduce callers that interpret ``noise_scale`` directly as standard
    deviations.
    """
    return make_gaussian_nv_noise_scale(d, noise_ratio, mode)


def standardize_columns(X: np.ndarray) -> np.ndarray:
    scale = X.std(axis=0, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return (X - X.mean(axis=0, keepdims=True)) / scale


def shd_score(G_true: np.ndarray, G_est: np.ndarray) -> float:
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


def skeleton_precision_recall(G_true: np.ndarray, G_est: np.ndarray) -> Tuple[float, float]:
    skel_true = np.triu(get_skeleton(G_true), k=1).astype(bool)
    skel_est = np.triu(get_skeleton(G_est), k=1).astype(bool)
    tp = int(np.sum(skel_true & skel_est))
    fp = int(np.sum((~skel_true) & skel_est))
    fn = int(np.sum(skel_true & (~skel_est)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(precision), float(recall)


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


def make_failure_row(noise_ratio, d, n_samples, degree, trial_id, seed, message):
    return TrialRow(
        noise_ratio=noise_ratio,
        d=d,
        n_samples=n_samples,
        degree=degree,
        trial_id=trial_id,
        seed=seed,
        algorithm="notears",
        status="failed",
        mec_match=0,
        shd=np.nan,
        cpdag_shd=np.nan,
        directed_precision=np.nan,
        directed_recall=np.nan,
        directed_f1=np.nan,
        skeleton_precision=np.nan,
        skeleton_recall=np.nan,
        n_edges_true=0,
        n_edges_est=0,
        noise_scale_min=np.nan,
        noise_scale_max=np.nan,
        runtime_sec=np.nan,
        message=message,
    )


def run_trial(noise_ratio, d, n_samples, degree, trial_id, seed, args):
    set_random_seed(seed)
    s0 = int(round(degree * d))
    B_bin = simulate_dag(d=d, s0=s0, graph_type=args.graph_type)
    W_true = simulate_parameter(B_bin)
    noise_scale = make_noise_scales(d, noise_ratio, args.noise_scale_mode)
    X = simulate_linear_sem(W_true, n_samples, sem_type="gauss", noise_scale=noise_scale)
    if args.standardize:
        X = standardize_columns(X)
    G_true = weight_to_binary_adj(W_true)

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

    directed_precision, directed_recall, directed_f1 = precision_recall_f1(G_true, G_est)
    skeleton_precision, skeleton_recall = skeleton_precision_recall(G_true, G_est)
    return TrialRow(
        noise_ratio=noise_ratio,
        d=d,
        n_samples=n_samples,
        degree=degree,
        trial_id=trial_id,
        seed=seed,
        algorithm="notears",
        status="ok",
        mec_match=int(is_in_markov_equiv_class(G_true, G_est)),
        shd=shd_score(G_true, G_est),
        cpdag_shd=cpdag_shd_score(G_true, G_est),
        directed_precision=directed_precision,
        directed_recall=directed_recall,
        directed_f1=directed_f1,
        skeleton_precision=skeleton_precision,
        skeleton_recall=skeleton_recall,
        n_edges_true=int(G_true.sum()),
        n_edges_est=int(G_est.sum()),
        noise_scale_min=float(noise_scale.min()),
        noise_scale_max=float(noise_scale.max()),
        runtime_sec=float(runtime_sec),
        message="",
    )


def summarize(df_trials: pd.DataFrame) -> pd.DataFrame:
    ok = df_trials[df_trials["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(["noise_ratio", "d", "n_samples", "degree", "algorithm"], as_index=False)
        .agg(
            trials=("trial_id", "count"),
            mec_match_mean=("mec_match", "mean"),
            shd_mean=("shd", "mean"),
            shd_std=("shd", "std"),
            cpdag_shd_mean=("cpdag_shd", "mean"),
            cpdag_shd_std=("cpdag_shd", "std"),
            directed_precision_mean=("directed_precision", "mean"),
            directed_recall_mean=("directed_recall", "mean"),
            directed_f1_mean=("directed_f1", "mean"),
            skeleton_precision_mean=("skeleton_precision", "mean"),
            skeleton_recall_mean=("skeleton_recall", "mean"),
            n_edges_true_mean=("n_edges_true", "mean"),
            n_edges_est_mean=("n_edges_est", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
        )
        .sort_values(["d", "n_samples", "degree", "noise_ratio"])
        .reset_index(drop=True)
    )

    failures = (
        df_trials[df_trials["status"] != "ok"]
        .groupby(["noise_ratio", "d", "n_samples", "degree", "algorithm"], as_index=False)
        .size()
        .rename(columns={"size": "failures"})
    )
    summary = summary.merge(
        failures, on=["noise_ratio", "d", "n_samples", "degree", "algorithm"], how="left"
    )
    summary["failures"] = summary["failures"].fillna(0).astype(int)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NOTEARS using CALM-compatible synthetic data."
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-list", type=parse_int_list, default=[50])
    parser.add_argument("--n-list", type=parse_int_list, default=[20000])
    parser.add_argument("--degree", type=float, default=1.0)
    parser.add_argument("--graph-type", type=str, default="ER", choices=["ER", "SF", "BP"])
    parser.add_argument("--noise-ratios", type=parse_float_list, default=[16.0])
    parser.add_argument(
        "--noise-scale-mode",
        choices=["variance", "std"],
        default="variance",
        help="Interpret paper noise ratio as variance range (sqrt passed to SEM) or direct SEM std.",
    )
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--notears-lambda1", type=float, default=0.1)
    parser.add_argument("--notears-loss-type", type=str, default="l2")
    parser.add_argument("--notears-threshold", type=float, default=0.1)
    parser.add_argument("--verbose-notears", action="store_true")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "results"),
    )
    parser.add_argument("--tag", type=str, default="notears_calm_data_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    if not HAS_NOTEARS:
        raise RuntimeError(f"NOTEARS unavailable: {NOTEARS_IMPORT_ERROR}")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    print("NOTEARS CALM-data benchmark")
    print(f"  d_list           : {args.d_list}")
    print(f"  n_list           : {args.n_list}")
    print(f"  trials           : {args.trials}")
    print(f"  degree           : {args.degree} (s0 = degree * d)")
    print(f"  graph_type       : {args.graph_type}")
    print(f"  noise_ratios     : {args.noise_ratios}")
    print(f"  noise_scale_mode : {args.noise_scale_mode}")
    print(f"  standardize      : {args.standardize}")
    print(f"  lambda1          : {args.notears_lambda1}")
    print(f"  threshold        : {args.notears_threshold}")

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
                            degree=args.degree,
                            trial_id=trial_idx,
                            seed=seed,
                            args=args,
                        )
                        print(
                            f"[notears] ratio={noise_ratio:g} d={d} n={n_samples} "
                            f"trial={trial_idx} edges={row.n_edges_true} "
                            f"mec={row.mec_match} shd={row.shd:.0f} "
                            f"cpdag_shd={row.cpdag_shd:.0f} "
                            f"skel_p={row.skeleton_precision:.3f} "
                            f"skel_r={row.skeleton_recall:.3f} "
                            f"rt={row.runtime_sec:.3f}s"
                        )
                    except Exception as exc:
                        row = make_failure_row(
                            noise_ratio, d, n_samples, args.degree, trial_idx, seed, str(exc)
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
