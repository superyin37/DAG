# L1-惩罚版 CD-A 算法设计

**对象文件**: [`coordinate_descent/cd_A_l1.py`](../coordinate_descent/cd_A_l1.py)
**作成日**: 2026-04-25
**对应算法**: `dag_coordinate_descent_l1`（实验笔记本中记为 `cd_A_l1`）
**关联文档**: [weak_faithfulness_cd_A_noepoch_zh.md](weak_faithfulness_cd_A_noepoch_zh.md)（共享 weakfaith 屏蔽机制）

---

## 1. 目的与动机

`coordinate0.dag_coordinate_descent_l0`（以下简称 **L0 版**）使用对每条非零边
统一收取常数代价 $\lambda_0$ 的 L0 惩罚：

$$
F_{L0}(A) = f(A) + \lambda_0 \sum_{i \neq j} \mathbf{1}\{A_{ij} \neq 0\}
$$

L0 的优点是"激活/不激活"是干净的二值决策；缺点是：

1. **对边大小不敏感** —— 微小的 $A_{ij}$ 与显著的 $A_{ij}$ 付同样的代价。
2. **不连续** —— 与 GLasso、NOTEARS-L1 等基于连续松弛的工作不可直接比较。
3. **不收缩** —— 一旦激活就保留无惩罚最优值 $\delta^*$，估计值偏向 OLS 而非
   稀疏化的 shrunk 值。

**L1 版** 改用：

$$
F_{L1}(A) = f(A) + \lambda_1 \sum_{i \neq j} |A_{ij}|
$$

每条边的惩罚正比于权重幅值，自动实现 soft-thresholding 风格的收缩。
本文档描述 L1 子问题的精确闭式解（设计草案中标记为"路径 B"）以及与
现有 weakfaith 屏蔽机制的集成。

---

## 2. 子问题与目标函数

### 2.1 单坐标更新的代数

固定 $A$，对位置 $(i, j)$ 引入扰动 $\delta E_{ij}$。利用 rank-1 行列式公式与
迹的展开（与 L0 版 [`coordinate0.delta_star`](../coordinate_descent/coordinate0.py#L23-L50)
共用同一组系数）：

$$
f(A + \delta E_{ij}) - f(A)
= -2 \log(1 + \delta\alpha) + 2\delta b + \delta^2 c
$$

其中

$$
\alpha = (A^{-1})_{ji}, \quad
b = (S A)_{ij} = S_{i,:} A_{:,j}, \quad
c = S_{ii}.
$$

加上 L1 惩罚（注意调用 `update_off_diagonal_l1` 时 $A_{ij}$ 已先被置零，
所以惩罚就是 $\lambda_1 |\delta|$）：

$$
g(\delta) = -2\log(1+\delta\alpha) + 2\delta b + \delta^2 c + \lambda_1 |\delta|
$$

定义域：$1 + \delta\alpha > 0$（保证 $\det$ 不变号、log 可计算）。

### 2.2 KKT 条件

$g$ 在 $\delta \neq 0$ 处可导，$\delta = 0$ 处用次微分。设 $\sigma = \operatorname{sign}(\delta)$，
对 $\delta > 0$ 与 $\delta < 0$ 分别有：

$$
g'(\delta) = -\frac{2\alpha}{1+\delta\alpha} + 2b + 2c\delta + \lambda_1 \sigma = 0
$$

两端乘 $(1 + \delta\alpha)$ 并整理，得到与 $\sigma$ 相关的二次方程：

$$
\boxed{\;c\alpha\,\delta^2 + \big(c + \alpha b_\sigma\big)\,\delta + \big(b_\sigma - \alpha\big) = 0\;}
\quad\text{其中 } b_\sigma = b + \sigma \cdot \frac{\lambda_1}{2}.
$$

退化情形：
- $\alpha \approx 0$：方程退化为线性 $c\delta + b_\sigma = 0$，得 $\delta = -b_\sigma / c$。
- $c \approx 0$：协方差对角异常（数据近退化），算法返回 $\delta = 0$。

### 2.3 候选根的筛选

每个分支至多贡献一个**符号一致**的实数解（因为 $g$ 在 $\delta > 0$ 与
$\delta < 0$ 的开区间上都是严格凸的）。逐根筛选：

1. **判别式非负**：$D = (c+\alpha b_\sigma)^2 - 4c\alpha(b_\sigma - \alpha) \geq 0$。
2. **符号一致性**：$\operatorname{sign}(\delta) = \sigma$（候选根的实际符号
   必须与所设分支符号匹配，否则不在该分支的可行域内）。
3. **对数定义域**：$1 + \delta\alpha > 0$。

最后将通过筛选的候选 + $\delta = 0$ 一起代入 $g$，取使 $g$ 最小者作为子问题最优解。
soft-threshold dead zone（两个分支都无可行解时）等价于 $\delta = 0$ 自然胜出。

实现见 [`delta_star_l1`](../coordinate_descent/cd_A_l1.py#L66-L120)（约 50 行，含
所有边界情形）。

### 2.4 与 L0 子问题的对比

| 维度 | L0 版 `delta_star` | L1 版 `delta_star_l1` |
|---|---|---|
| 子问题最优 $\delta$ | 无惩罚最优 $\delta^*$ | $\delta_{l1} = \arg\min g(\delta)$ |
| 惩罚结构 | 接受时一次性付 $\lambda_0$ | 接受时付 $\lambda_1\|\delta\|$（与 $\delta$ 耦合）|
| 候选根数 | 1 | 至多 2 + 零点 |
| 收缩效果 | 无（"硬激活"）| 有（小 $\delta^*$ 直接被 soft-threshold 到 0）|
| $\lambda = 0$ 退化 | 直接返回 $\delta^*$ | 数值上与 `delta_star` 一致（**已验证** $\Delta f \leq 10^{-15}$）|

---

## 3. 算法结构

### 3.1 单步更新 `update_off_diagonal_l1`

与 [`coordinate0.update_off_diagonal`](../coordinate_descent/coordinate0.py#L212-L259)
完全同协议：

1. 保存旧值 $A_{ij}, A_{ji}$；将两者清零；用 Sherman–Morrison 同步 $A^{-1}$。
2. 对方向 $i \to j$：若 $A + E_{ij}$ 仍为 DAG，调用 `delta_star_l1` 计算
   $\delta_{l1}$，并求 $\Delta_{ij} = f(A) - f(A + \delta_{l1} E_{ij}) - \lambda_1 |\delta_{l1}|$。
3. 对方向 $j \to i$：同理计算 $\Delta_{ji}$。
4. 二者均为 $-\infty$（两方向都破坏 DAG）→ 保持清零。
5. 否则取 $\Delta$ 较大的方向接受；同步 SM 更新 $A^{-1}$。

**与 L0 版的关键差异**：

- L0 版的接受/拒绝由 `Δ ≥ 0` 显式判断（因为 $\delta^*$ 不是 L1 子问题最优，
  可能整体不优于 $\delta = 0$）；
- L1 版的 $\delta_{l1}$ 已经是 $\{0\} \cup \{\text{两分支根}\}$ 上的全局最小，
  $\Delta \geq 0$ **由构造保证**。所以 L1 版主循环不需要"整体不优则不动"的
  额外检查 —— 当 soft-threshold 决定不激活时，$\delta_{l1} = 0$ 自然落到 "$A$ 不变"。

### 3.2 主循环 `dag_coordinate_descent_l1`

签名与 [`cd_A_weakfaith.dag_coordinate_descent_l0_weakfaith`](../coordinate_descent/cd_A_weakfaith.py#L88-L218)
**完全平行**：

```python
dag_coordinate_descent_l1(
    S, T=100, seed=0, threshold=0.05, lambda_l1=0.2,
    return_history=False, return_graph_history=False, A_init=None,
    early_stop=False, check_every=None, tol=1e-4, patience=10, min_steps=None,
    # weakfaith hooks (set faithfulness_tau=0 to disable)
    faithfulness_tau=0.0,
    sampling_mode="preserve",
    screening="corr",
    glasso_alpha=0.01,
    combine="union",
)
```

唯一行为差异：

- 偏对角更新走 `update_off_diagonal_l1` 而非 `update_off_diagonal`。
- `history` 记录的是 **L1-惩罚目标** $f(A) + \lambda_1 \|A\|_{1, \text{off}}$
  （由 helper [`f_l1`](../coordinate_descent/cd_A_l1.py#L173-L179) 计算），
  而非纯 $f(A)$。这样可以直接观察单调下降。

### 3.3 weakfaith 屏蔽的复用

L1 版直接 import 并复用 [`cd_A_weakfaith._build_faithfulness_mask`](../coordinate_descent/cd_A_weakfaith.py#L72-L132)，
所有 screening 选项（`corr` / `pcorr` / `glasso` / 列表合并 + `combine='union'|'intersect'`）
**无需重新实现**。语义与 L0 weakfaith 版完全一致：

- `faithfulness_tau = 0` → 退化为均匀采样（与无屏蔽 L1 版 byte-equivalent）。
- `faithfulness_tau > 0` → 采样限制在 `_build_faithfulness_mask` 返回的允许对中。
- `sampling_mode='preserve'` 保持 $P(\text{diag}) = 1/d$；`'pool'` 在
  $\{d \text{ 个对角}\} \cup \{M \text{ 个允许的偏对角}\}$ 上均匀。

---

## 4. 数值正确性

[`cd_A_l1.py`](../coordinate_descent/cd_A_l1.py) 上线前通过四组验证（详见
代码仓 issue 历史中的 smoke test 输出）：

| 测试 | 内容 | 实测结果 |
|---|---|---|
| **退化一致性** | $\lambda_1 = 0$ 时 `delta_star_l1` 与 `delta_star` 同效 | $\max\|f(A+\delta_{l0}) - f(A+\delta_{l1\|\lambda=0})\| = 1.78\times 10^{-15}$ |
| **KKT 余量** | $\lambda_1 \in \{0.05, 0.2, 0.5, 1.0\}$ 上验证次梯度 KKT 条件 | $\leq 10^{-14}$ |
| **闭式 vs 网格** | 与 6001 点细网格搜索比对 30 个随机 $(i,j,\lambda)$ | gap = 0（闭式严格不差）|
| **目标单调性** | 主循环跑 2000 步，记录 $F_{L1}$ 序列 | 0 次实质上升，最大波动 $\leq 7.1\times 10^{-15}$（roundoff）|

第三条特别需要注意：scipy 的 `minimize_scalar(bracket=...)` 在某些情形下会
错过最优、错误地返回 $\delta = 0$ —— 早期我们误以为是闭式解错误，实际是
基准方法不可靠。改用细网格之后闭式解 100% 不差。

---

## 5. 调参与实践

### 5.1 $\lambda_1$ 的尺度

L1 惩罚 $\lambda_1 |\delta|$ 与 L0 惩罚 $\lambda_0$ 不在同一尺度。粗略估计：

$$
\lambda_1 \approx \lambda_0 / |\bar A|
$$

其中 $|\bar A|$ 是典型边权重幅值。我们的合成数据中 `B_scale=1.0`，
权重多在 $0.5$–$1.0$，所以 $\lambda_0 = 0.2$ 大致对应 $\lambda_1 \in [0.2, 0.4]$。
但 L1 真正的最优值常常更小（因为 soft-threshold 已经自带剪枝），实践上 $\lambda_1 \in [0.05, 0.2]$ 更常见。

[test_er_cd_A_l0_vs_l1_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_l0_vs_l1_benchmark.ipynb)
默认扫 $\lambda_1 \in \{0.05, 0.1, 0.2\}$，配合 $\lambda_0 = 0.2$ 的 L0 锚点对比。

### 5.2 后处理 threshold

L0 接受后保留 $\delta^*$，weight 多数远大于 0；后处理 `threshold = 0.05`
（[`weight_to_adjacency`](../coordinate_descent/coordinate0.py#L85-L97)）通常足够。

L1 由于 soft-threshold，会留下若干**很小但非零**的边。这些边在 L1 的
最优解里"惩罚买不掉"，但在统计意义上经常是噪声。如果发现 L1 的
`n_edges_est` 显著高于真实边数，把 threshold 抬到 0.1 或随 $\lambda_1$
线性放大都是常见做法。

### 5.3 与 weakfaith 联用

实验显示（见 [test_er_cd_A_weakfaith_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)），
`pcorr` + $\tau = 0.05$ 是 L0 上最稳的 weakfaith 配置；这一组合在 L1 上
同样适用，并且：

- 屏蔽掉的"必为零"对在 L1 下也无须 soft-threshold —— **省下的步数都用在
  真正可能贡献的边上**。
- L1 的 `Δ ≥ 0` 性质保证了即使 $\delta_{l1}$ 是 dead-zone 0，也不会
  倒回旧权重；与 weakfaith 屏蔽完全正交。

---

## 6. 路径 A 的取舍

设计阶段考虑过两条 L1 路径：

| 路径 | 思路 | 实现成本 | 精确性 |
|---|---|---|---|
| **A：proximal** | 用无惩罚 $\delta^*$ 后做 soft-threshold $\delta_{l1} \approx \operatorname{sign}(\delta^*)\max(\|\delta^*\| - \lambda_1/(2c), 0)$ | ~10 行 | 对 log-det 项是一阶近似，$\lambda_1$ 大时偏离最优 |
| **B：精确闭式**（本实现） | 解 §2.2 的 KKT 二次方程并筛选 | ~50 行 | 子问题精确最优 |

选择 B 的理由：

1. **可重复性**：精确解使 $F_{L1}$ 严格单调，便于做收敛分析与图像。
2. **理论可比性**：与文献中的 lasso 类方法直接可比，不引入额外近似误差。
3. **代价可接受**：每次 `update_off_diagonal_l1` 多评估 1–2 个候选根，
   开销 $O(1)$；主成本仍是 SM 更新与 DAG 检查，与 L0 同阶。

如果未来有 hot-loop 优化需求（例如 $T \geq 10^6$），路径 A 可以作为加速旁路
保留 —— 但当前 benchmark 规模下，路径 B 的 runtime 与 L0 在同一量级
（实测 d=12 / T=2000 下，单次 ~0.4s，与 L0 持平）。

---

## 7. API 速查

```python
from coordinate_descent.cd_A_l1 import (
    delta_star_l1,            # 闭式 L1 子问题最优 delta
    update_off_diagonal_l1,   # 单步 L1 偏对角更新
    f_l1,                     # L1-惩罚目标函数
    dag_coordinate_descent_l1,  # 主循环（含 weakfaith hooks）
)

# 1) 纯 L1，无 screening
A, G, obj = dag_coordinate_descent_l1(S, T=100000, seed=0, lambda_l1=0.1)

# 2) L1 + pcorr screening
A, G, obj = dag_coordinate_descent_l1(
    S, T=100000, seed=0, lambda_l1=0.1,
    faithfulness_tau=0.05, screening=['pcorr'],
)

# 3) L1 + 多 screen 合并（discard iff all-zero）
A, G, obj = dag_coordinate_descent_l1(
    S, T=100000, seed=0, lambda_l1=0.1,
    faithfulness_tau=0.05,
    screening=['corr', 'pcorr', 'glasso'],
    combine='union',
    glasso_alpha=0.01,
)
```

实验入口：[test_er_cd_A_l0_vs_l1_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_l0_vs_l1_benchmark.ipynb)。
