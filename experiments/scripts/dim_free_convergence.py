"""
实验：验证坐标下降的 Dimension-Independent Convergence
=======================================================

目标：研究 CD_B 算法达到精度 ε 所需的坐标更新次数是否与维度 d 无关。

实验设计原则：
  1. 使用人口协方差矩阵 S（n → ∞），排除采样噪声干扰。
  2. 控制初始距离 ‖B₀ − B*‖_F ≈ σ，不随 d 变化。
  3. 限定在已知拓扑序的上三角可行域内搜索，保证 B* 是唯一极小值。
  4. 随机坐标选择（uniform random coordinate），每次只更新一个 B[i,j]。
  5. 计数单位：单次坐标更新（非 epoch），这是最粒度化的度量。

输出：
  - experiments/results/dim_free_convergence/summary.csv  — 每个 d 的 mean/std(t_ε)
  - experiments/results/dim_free_convergence/trajectories.npz — 收敛曲线数据
  - experiments/results/dim_free_convergence/figure1_t_eps_vs_d.png
  - experiments/results/dim_free_convergence/figure2_convergence_curves.png
  - experiments/results/dim_free_convergence/figure3_normalized_curves.png
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
RESULTS_DIR = os.path.join(REPO_ROOT, "experiments", "results", "dim_free_convergence")
os.makedirs(RESULTS_DIR, exist_ok=True)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from coordinate_descent.cd_B import f_B, delta_star_B, _sm_update_M_inv


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def generate_random_dag_upper(d: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    生成随机稀疏 DAG，以上三角矩阵形式存储（固定拓扑序 0 → 1 → ... → d-1）。

    B[i,j] 表示边 i → j 的权重（i < j），对应模型 X = B^T X + N。
    边概率 edge_prob ≈ 2/d 使期望边数约为 d。
    """
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                # 权重从 U[0.5, 2.0] 随机正负，避免退化
                sign = rng.choice([-1.0, 1.0])
                B[i, j] = sign * rng.uniform(0.5, 2.0)
    return B


def compute_population_S(B_star: np.ndarray) -> np.ndarray:
    """
    计算人口协方差矩阵（等效于 n → ∞）。

    模型：X = B*^T X + N，N ~ N(0, I)
    Sigma = (I - B*^T)^{-1} (I - B*)^{-T}

    在此设定下：
      (I - B*^T) S (I - B*) = I
      f_B(B*) = sum_i log(1) - 2 log det(I - B*^T) + d = d
    （因 B* 上三角，det(I - B*^T) = 1）
    """
    d = B_star.shape[0]
    M_star = np.eye(d) - B_star.T          # M* = I - B*^T，下三角，det = 1
    M_star_inv = inv(M_star)               # O(d³)，只算一次
    S = M_star_inv @ M_star_inv.T          # S = (M*)^{-1} (M*)^{-T}
    return S


def make_initial_point(B_star: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    生成初始点 B₀ = B* + noise（只在上三角区域加噪）。

    noise[i,j] ~ N(0, σ²/d²)  →  E[‖noise‖_F²] ≈ d²/2 · σ²/d² = σ²/2
    因此 ‖B₀ − B*‖_F ≈ σ/√2，与 d 无关。
    """
    d = B_star.shape[0]
    noise_scale = sigma / d
    B0 = B_star.copy()
    for i in range(d):
        for j in range(i + 1, d):
            B0[i, j] += rng.normal(0.0, noise_scale)
    return B0


# ══════════════════════════════════════════════════════════════════════════════
# 核心：带追踪的坐标下降
# ══════════════════════════════════════════════════════════════════════════════

def run_cd_b_convergence(
    B_init: np.ndarray,
    S: np.ndarray,
    f_star: float,
    epsilon: float,
    max_updates: int,
    check_interval: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, float]], int | None]:
    """
    在固定拓扑序的上三角可行域内运行随机 CD_B，追踪目标值差距。

    参数
    ----
    B_init        : 初始权重矩阵（仅上三角有效）
    S             : 人口协方差矩阵
    f_star        : 真实最优值 f_B(B*)，用于计算 gap
    epsilon       : 收敛阈值
    max_updates   : 最大坐标更新次数
    check_interval: 每隔多少次更新评估一次目标函数
    rng           : 随机数生成器

    返回
    ----
    history    : [(update_count, gap), ...] 的列表
    t_epsilon  : 首次 gap ≤ ε 时的更新次数（未达到则为 None）
    """
    d = B_init.shape[0]
    B = B_init.copy()

    # 初始化 M_inv = (I - B^T)^{-1}
    M = np.eye(d) - B.T
    M_inv = inv(M)

    # 可行坐标集：上三角 (i < j)
    coords = np.array([(i, j) for i in range(d) for j in range(i + 1, d)])
    n_coords = len(coords)

    # 初始 gap
    f_cur = f_B(B, S)
    history = [(0, f_cur - f_star)]
    t_epsilon = None
    updates_since_check = 0

    for t in range(1, max_updates + 1):
        # 随机选择坐标
        idx = rng.integers(n_coords)
        i, j = int(coords[idx, 0]), int(coords[idx, 1])

        # 闭型最优步长
        delta = delta_star_B(B, S, i, j, M_inv=M_inv)

        if abs(delta) > 1e-30:
            B[i, j] += delta
            ok = _sm_update_M_inv(M_inv, i, j, delta)
            if not ok:
                # SM 更新失败：重新计算完整逆矩阵
                M = np.eye(d) - B.T
                M_inv = inv(M)

        updates_since_check += 1

        # 每 check_interval 次评估目标函数
        if updates_since_check >= check_interval:
            f_cur = f_B(B, S)
            gap = f_cur - f_star
            history.append((t, max(gap, 0.0)))
            updates_since_check = 0

            if t_epsilon is None and gap <= epsilon:
                t_epsilon = t
                break  # 已收敛，提前终止

    # 末尾补一次精确评估
    if updates_since_check > 0:
        f_cur = f_B(B, S)
        gap = f_cur - f_star
        history.append((max_updates, max(gap, 0.0)))
        if t_epsilon is None and gap <= epsilon:
            t_epsilon = max_updates

    return history, t_epsilon


# ══════════════════════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    d_list: list[int],
    n_repeats: int,
    sigma: float,
    epsilon: float,
    max_updates_multiplier: int,
    master_seed: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    对每个 d，重复 n_repeats 次实验，记录 t_ε。

    返回
    ----
    df_summary   : pd.DataFrame，每行一个 d，含 mean/std/median(t_ε)
    all_histories: dict，d → list of history（每次运行的收敛曲线）
    """
    rng_master = np.random.default_rng(master_seed)
    rows = []
    all_histories: dict[int, list] = {}

    for d in d_list:
        # max_updates 设为 d² 的若干倍，足够大
        max_updates = max_updates_multiplier * d * d
        check_interval = max(1, d)       # 每 d 次更新检查一次
        f_star = float(d)                # 见 compute_population_S 注释

        t_eps_list = []
        histories = []

        if verbose:
            print(f"\n{'='*50}")
            print(f"d = {d}  |  max_updates = {max_updates:,}  |  check_interval = {check_interval}")
            print(f"{'='*50}")

        t_wall_start = time.time()

        for rep in range(n_repeats):
            rep_seed = int(rng_master.integers(1, 2**31))
            rng_rep = np.random.default_rng(rep_seed)

            # 生成真实 DAG
            B_star = generate_random_dag_upper(d, edge_prob=2.0 / d, rng=rng_rep)

            # 人口协方差矩阵
            S = compute_population_S(B_star)

            # 初始点
            B_init = make_initial_point(B_star, sigma=sigma, rng=rng_rep)
            init_dist = float(np.linalg.norm(B_init - B_star, 'fro'))

            # 运行 CD_B 收敛追踪
            history, t_eps = run_cd_b_convergence(
                B_init=B_init,
                S=S,
                f_star=f_star,
                epsilon=epsilon,
                max_updates=max_updates,
                check_interval=check_interval,
                rng=rng_rep,
            )

            histories.append(history)

            if t_eps is not None:
                t_eps_list.append(t_eps)
                if verbose and (rep < 5 or rep == n_repeats - 1):
                    print(f"  rep {rep+1:3d}/{n_repeats}: t_eps = {t_eps:6d}  "
                          f"dist = {init_dist:.3f}")
            else:
                if verbose and (rep < 5 or rep == n_repeats - 1):
                    print(f"  rep {rep+1:3d}/{n_repeats}: NOT CONVERGED  "
                          f"dist = {init_dist:.3f}")

        all_histories[d] = histories
        t_wall = time.time() - t_wall_start

        row = {
            'd': d,
            'n_converged': len(t_eps_list),
            'convergence_rate': len(t_eps_list) / n_repeats,
            'mean_t_eps': np.mean(t_eps_list) if t_eps_list else np.nan,
            'std_t_eps': np.std(t_eps_list) if t_eps_list else np.nan,
            'median_t_eps': np.median(t_eps_list) if t_eps_list else np.nan,
            'p25_t_eps': np.percentile(t_eps_list, 25) if t_eps_list else np.nan,
            'p75_t_eps': np.percentile(t_eps_list, 75) if t_eps_list else np.nan,
            'wall_time_s': t_wall,
        }
        rows.append(row)

        if verbose:
            print(f"  -> converged: {len(t_eps_list)}/{n_repeats}  "
                  f"mean t_eps = {row['mean_t_eps']:.1f}  "
                  f"wall = {t_wall:.1f}s")

    df = pd.DataFrame(rows)
    return df, all_histories


# ══════════════════════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(vals: list[float], window: int = 5) -> list[float]:
    """简单滑动平均，用于平滑收敛曲线。"""
    if window <= 1 or len(vals) < window:
        return vals
    out = []
    for k in range(len(vals)):
        lo = max(0, k - window // 2)
        hi = min(len(vals), k + window // 2 + 1)
        out.append(float(np.mean(vals[lo:hi])))
    return out


def plot_figure1(df: pd.DataFrame, epsilon: float, save_path: str) -> None:
    """图1：t_ε vs d，展示维度无关性。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ds = df['d'].values
    means = df['mean_t_eps'].values
    stds = df['std_t_eps'].values
    p25 = df['p25_t_eps'].values
    p75 = df['p75_t_eps'].values

    mask = ~np.isnan(means)

    # 左图：线性坐标
    ax = axes[0]
    ax.errorbar(ds[mask], means[mask], yerr=stds[mask],
                fmt='o-', capsize=5, color='steelblue', linewidth=2,
                label='mean ± std')
    ax.fill_between(ds[mask], p25[mask], p75[mask],
                    alpha=0.25, color='steelblue', label='IQR')
    # 参考线：constant（取 d=5 的均值）
    if mask[0]:
        ax.axhline(means[mask][0], color='gray', linestyle='--',
                   alpha=0.7, label=f'constant = {means[mask][0]:.0f}')
    # 参考线：O(d)
    d_ref = ds[mask]
    if mask[0] and means[mask][0] > 0:
        slope = means[mask][0] / d_ref[0]
        ax.plot(d_ref, slope * d_ref, 'r--', alpha=0.7, label='O(d)')
    ax.set_xlabel('Dimension d', fontsize=13)
    ax.set_ylabel(f'Updates to reach ε = {epsilon}', fontsize=13)
    ax.set_title('Required Updates vs Dimension (linear scale)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右图：对数坐标
    ax = axes[1]
    ax.errorbar(ds[mask], means[mask], yerr=stds[mask],
                fmt='o-', capsize=5, color='steelblue', linewidth=2)
    ax.fill_between(ds[mask], p25[mask], p75[mask],
                    alpha=0.25, color='steelblue')
    if mask[0] and means[mask][0] > 0:
        ax.axhline(means[mask][0], color='gray', linestyle='--',
                   alpha=0.7, label='constant')
        slope = means[mask][0] / d_ref[0]
        ax.plot(d_ref, slope * d_ref, 'r--', alpha=0.7, label='O(d)')
        ax.plot(d_ref, slope / d_ref[0] * d_ref**2, 'g--', alpha=0.7, label='O(d²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimension d', fontsize=13)
    ax.set_ylabel(f'Updates to reach ε = {epsilon}', fontsize=13)
    ax.set_title('Required Updates vs Dimension (log-log scale)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('Dimension-Independent Convergence Study (CD_B)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_figure2(all_histories: dict, d_list: list[int],
                 f_star_map: dict[int, float],
                 epsilon: float, save_path: str,
                 n_sample: int = 10) -> None:
    """图2：不同 d 的收敛曲线（f gap vs update count）。"""
    cmap = plt.cm.viridis
    colors = [cmap(k / max(len(d_list) - 1, 1)) for k in range(len(d_list))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for color, d in zip(colors, d_list):
        histories = all_histories.get(d, [])
        if not histories:
            continue

        # 取最多 n_sample 条轨迹，取中位数
        sample_hist = histories[:n_sample]

        # 插值到公共横轴
        max_t = max(h[-1][0] for h in sample_hist if h)
        xs = np.linspace(0, max_t, 500)
        ys_all = []
        for hist in sample_hist:
            ts = np.array([p[0] for p in hist])
            gs = np.array([p[1] for p in hist])
            gs_interp = np.interp(xs, ts, gs)
            ys_all.append(gs_interp)

        ys_med = np.median(ys_all, axis=0)
        ys_lo = np.percentile(ys_all, 25, axis=0)
        ys_hi = np.percentile(ys_all, 75, axis=0)

        ax.semilogy(xs, ys_med, color=color, linewidth=2, label=f'd = {d}')
        ax.fill_between(xs, np.maximum(ys_lo, 1e-12),
                        np.maximum(ys_hi, 1e-12),
                        color=color, alpha=0.15)

    ax.axhline(epsilon, color='black', linestyle=':', linewidth=1.5,
               label=f'ε = {epsilon}')
    ax.set_xlabel('Coordinate Updates', fontsize=13)
    ax.set_ylabel('f(B) − f(B*)', fontsize=13)
    ax.set_title(f'Convergence Curves for Different Dimensions (CD_B)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_figure3_normalized(all_histories: dict, d_list: list[int],
                            save_path: str, n_sample: int = 10) -> None:
    """图3：按 d² 归一化的横轴 (= epoch fraction)，验证 epoch 层面是否也维度无关。"""
    cmap = plt.cm.plasma
    colors = [cmap(k / max(len(d_list) - 1, 1)) for k in range(len(d_list))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for color, d in zip(colors, d_list):
        histories = all_histories.get(d, [])
        if not histories:
            continue

        sample_hist = histories[:n_sample]
        max_t = max(h[-1][0] for h in sample_hist if h)
        xs = np.linspace(0, max_t / (d * d), 500)   # 归一化为 epoch 数
        ys_all = []
        for hist in sample_hist:
            ts = np.array([p[0] / (d * d) for p in hist])
            gs = np.array([p[1] for p in hist])
            gs_interp = np.interp(xs, ts, gs)
            ys_all.append(gs_interp)

        ys_med = np.median(ys_all, axis=0)
        ax.semilogy(xs, ys_med, color=color, linewidth=2, label=f'd = {d}')

    ax.set_xlabel('Epochs  (= updates / d²)', fontsize=13)
    ax.set_ylabel('f(B) − f(B*)', fontsize=13)
    ax.set_title('Convergence Curves Normalized by d² (epoch scale)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_trajectories(all_histories: dict, d_list: list[int], save_path: str) -> None:
    """将各 d 的收敛曲线保存为 .npz（供 notebook 使用）。"""
    data = {}
    for d in d_list:
        histories = all_histories.get(d, [])
        # 保存最多 20 条轨迹
        for k, hist in enumerate(histories[:20]):
            ts = np.array([p[0] for p in hist])
            gs = np.array([p[1] for p in hist])
            data[f'd{d}_rep{k}_t'] = ts
            data[f'd{d}_rep{k}_g'] = gs
    np.savez(save_path, **data)
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Dimension-independent convergence experiment for CD_B"
    )
    p.add_argument('--d_list', nargs='+', type=int,
                   default=[5, 10, 20, 40, 80],
                   help='List of dimensions to test (default: 5 10 20 40 80)')
    p.add_argument('--n_repeats', type=int, default=50,
                   help='Repetitions per dimension (default: 50)')
    p.add_argument('--sigma', type=float, default=1.0,
                   help='Initial noise scale σ (default: 1.0)')
    p.add_argument('--epsilon', type=float, default=1e-3,
                   help='Convergence threshold ε (default: 1e-3)')
    p.add_argument('--max_mult', type=int, default=500,
                   help='max_updates = max_mult * d² (default: 500)')
    p.add_argument('--seed', type=int, default=42,
                   help='Master random seed (default: 42)')
    p.add_argument('--no_plot', action='store_true',
                   help='Skip plotting (useful on headless servers)')
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Dimension-Independent Convergence Experiment (CD_B)")
    print("=" * 60)
    print(f"  d_list      = {args.d_list}")
    print(f"  n_repeats   = {args.n_repeats}")
    print(f"  sigma       = {args.sigma}")
    print(f"  epsilon     = {args.epsilon}")
    print(f"  max_mult    = {args.max_mult}  (max_updates = mult * d^2)")
    print(f"  seed        = {args.seed}")
    print(f"  results_dir = {RESULTS_DIR}")
    print()

    # 运行实验
    df_summary, all_histories = run_experiment(
        d_list=args.d_list,
        n_repeats=args.n_repeats,
        sigma=args.sigma,
        epsilon=args.epsilon,
        max_updates_multiplier=args.max_mult,
        master_seed=args.seed,
        verbose=True,
    )

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df_summary[['d', 'n_converged', 'convergence_rate',
                       'mean_t_eps', 'std_t_eps', 'median_t_eps']].to_string(index=False))

    # ── 判断维度无关性 ─────────────────────────────────────────────────────────
    converged_df = df_summary.dropna(subset=['mean_t_eps']).copy()
    if len(converged_df) >= 2:
        d_vals = converged_df['d'].values
        t_vals = converged_df['mean_t_eps'].values

        # epochs = t_eps / (d*(d-1)/2)  —  随机 CD 每选满一轮的期望次数
        n_coords = d_vals * (d_vals - 1) / 2
        epoch_vals = t_vals / n_coords
        converged_df['mean_epochs'] = epoch_vals

        # 对 log(t) ~ α * log(d) 做线性回归，估计幂次 α（原始更新次数）
        log_d = np.log(d_vals)
        log_t = np.log(t_vals)
        alpha_raw = np.polyfit(log_d, log_t, 1)[0]

        # 对 epoch 数做同样的回归
        log_ep = np.log(epoch_vals)
        alpha_ep = np.polyfit(log_d, log_ep, 1)[0]

        print(f"\n  Scaling analysis (coordinate updates):  t_eps ~ d^{alpha_raw:.3f}")
        print(f"  Scaling analysis (epochs = t / n_coords): epochs ~ d^{alpha_ep:.3f}")

        print("\n  Epoch counts per dimension:")
        for d_v, ep_v in zip(d_vals, epoch_vals):
            print(f"    d = {int(d_v):3d}  ->  mean epochs = {ep_v:.1f}")

        if abs(alpha_ep) < 0.3:
            print("\n  -> EPOCH-DIMENSION-INDEPENDENT (alpha_epoch ~ 0): strong property")
        elif abs(alpha_ep - 1.0) < 0.3:
            print("\n  -> Epochs scale linearly with d (alpha_epoch ~ 1)")
        else:
            print(f"\n  -> Epoch scaling: d^{alpha_ep:.2f}")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    npz_path = os.path.join(RESULTS_DIR, "trajectories.npz")
    save_trajectories(all_histories, args.d_list, npz_path)

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\nGenerating figures...")

        plot_figure1(
            df_summary,
            epsilon=args.epsilon,
            save_path=os.path.join(RESULTS_DIR, "figure1_t_eps_vs_d.png"),
        )
        plot_figure2(
            all_histories,
            d_list=args.d_list,
            f_star_map={d: float(d) for d in args.d_list},
            epsilon=args.epsilon,
            save_path=os.path.join(RESULTS_DIR, "figure2_convergence_curves.png"),
        )
        plot_figure3_normalized(
            all_histories,
            d_list=args.d_list,
            save_path=os.path.join(RESULTS_DIR, "figure3_normalized_curves.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
