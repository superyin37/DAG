# 单边忠实性假设在 `cd_A_noepoch` 中的应用设计

**对象文件**: `coordinate_descent/coordinate0.py`
**作成日**: 2026-04-18
**对应算法**: `dag_coordinate_descent_l0`（实验笔记本中记为 `cd_A_noepoch`）

---

## 1. 目的与动机

`cd_A_noepoch` 当前在 `{0, …, d-1}²` 上均匀采样坐标 (i, j)，随后调用
`update_off_diagonal` / `update_diagonal` 执行一次坐标下降更新
（[coordinate0.py:309-315](../coordinate_descent/coordinate0.py#L309-L315)）。

该设计的代价：
- 许多采样到的坐标对 (i, j) 对应着**边际几乎不相关的变量**。这类边在真实模型中通常不会出现，但算法依然会花费一次 $O(d^2)$ 的 `update_off_diagonal` 调用（包含两次 Sherman-Morrison、两次 DAG 检测、两次 $\delta^*$ 求解）来确认它"不应加入"。
- 在高维问题（$d \geq 30$）下，这类"无效步"占绝对多数，成为每轮 $T$ 步预算中的主要浪费。

本文档引入 **fGES 中的"单边忠实性假设"**（one-edge faithfulness assumption）作为剪枝工具，在 `cd_A_noepoch` 中屏蔽掉边际不相关变量对之间的所有加边候选，使随机采样集中在可能贡献收益的坐标上。

---

## 2. 理论背景：单边忠实性假设

### 2.1 假设内容（来自 fGES 论文 [^1]）

> 若变量 $x$ 与 $y$ 的样本边际相关系数为 0，则在前向搜索的任何步骤中都**不加入**有向边 $x \to y$。

这是完整忠实性假设的一个受限版本：只对"单边（single edge）"的忠实性违例做排除，不要求对条件独立关系的全局忠实性。

### 2.2 代价：路径完全相消

若真实模型为 $A \to B \to C \to D$ 且存在 $A \to D$，且两条路径的系数**恰好互相抵消**，则 $\operatorname{corr}(A, D) = 0$。此时启用单边忠实性会导致 $A \to D$ 边永远无法进入候选集 —— 这是该假设的已知代价。

在**连续参数空间下，路径相消的参数集合测度为 0**；但有限样本估计下，近似相消可能发生在测度不可忽略的参数集合上。

### 2.3 映射到 `cd_A_noepoch`

| fGES 语境 | `cd_A_noepoch` 语境 |
|---|---|
| 前向搜索中"新增边" | `update_off_diagonal` 中对 `A[i,j]` 从 0 变为非零的更新 |
| 边际相关系数 | $S$ 归一化后的 $\lvert\operatorname{corr}(x_i, x_j)\rvert$ |
| 屏蔽条件 | 独立性判据小于阈值 $\tau$ |

### 2.4 两种独立性判据：`corr` 与 `pcorr`

本设计支持两种"$x_i$ 与 $x_j$ 近似独立"的判据，对应不同独立性概念：

| 判据 | 统计含义 | 等于 0 时的 DAG 含义 |
|---|---|---|
| $\operatorname{corr}(x_i, x_j) = 0$ | **边际独立** | 无任何路径连接 $x_i$ 和 $x_j$ |
| $\operatorname{pcorr}(x_i, x_j \mid \text{rest}) = 0$ | **给定其他所有变量的条件独立** | $(x_i, x_j)$ 不在 moral graph 中（既非父子关系，也无共同子女） |

其中偏相关系数由精度矩阵 $\Omega = S^{-1}$ 给出：

$$
\operatorname{pcorr}(x_i, x_j \mid \text{rest}) = -\frac{\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}
$$

**剪枝严格度对比**：

- `corr` 筛选只屏蔽"完全无关联"的对 —— 保留所有"任何路径可达"的对。
- `pcorr` 筛选**额外屏蔽**那些"仅通过中介变量间接相关"的对。例如 $x_A \to x_B \to x_C$ 中的 $(A, C)$：$\operatorname{corr}(A, C) \neq 0$，但 $\operatorname{pcorr}(A, C \mid B) \approx 0$。
- 因此 **`pcorr` 比 `corr` 剪枝更激进**（屏蔽集合更大 → 加速比更高）。

**对真实有向 DAG 边的漏边风险**：两种判据在理论上**等价** —— 都仅在"路径完全相消"这一测度 0 事件下会错漏 $x_i \to x_j$ 本身对应的统计量。`pcorr` 不会因为"更激进"而错漏真实有向边。

---

## 3. `cd_A_noepoch` 中的接入方式

### 3.1 为何选择"策略 B：改采样分布"

两种可能的实施方式：

- **策略 A（整步跳过）**：主循环中采样到被屏蔽的 (i, j) 则 `continue`，保留原采样分布。
- **策略 B（改采样分布）**：预先构造"允许的候选对"列表，直接从中采样。

本设计采用**策略 B**，原因：

1. `update_off_diagonal` 的内部逻辑会**无条件先清零 A[i,j] 和 A[j,i]**
   （[coordinate0.py:216](../coordinate_descent/coordinate0.py#L216)）。若采用"整步跳过"且
   被跳过的对中 `A[i,j]` 当前非零（例如 `A_init` 预设值），则需要额外逻辑来
   区分"新增边"和"重优化已存在边"，实现复杂。
2. 策略 B 让每一步都在"可能贡献收益的坐标"上进行，避免浪费 `update_off_diagonal`
   的固定成本，从**每步期望收益**角度更优。
3. 策略 B 对 `early_stop` 机制透明 —— 每步都是有效更新，节拍不变。

### 3.2 mask 构造

根据 `screening` 参数（详见 §5.1）选择统计量：

```python
if faithfulness_tau > 0:
    if screening == "corr":
        std = np.sqrt(np.diag(S))
        stat = S / np.outer(std, std)                          # 样本相关矩阵
    elif screening == "pcorr":
        Omega = np.linalg.inv(S + 1e-6 * np.eye(d))            # 正则化求逆防奇异
        d_std = np.sqrt(np.diag(Omega))
        stat = -Omega / np.outer(d_std, d_std)                 # 偏相关矩阵
    else:
        raise ValueError(f"unknown screening {screening!r}")

    forbidden = np.abs(stat) < faithfulness_tau                # 对称
    np.fill_diagonal(forbidden, False)                         # 对角线不屏蔽
    allowed_offdiag = np.argwhere(~forbidden & ~np.eye(d, dtype=bool))
    M = len(allowed_offdiag)
    if M == 0:
        raise ValueError("All off-diagonal pairs masked; tau too large.")
else:
    allowed_offdiag = None
    M = d * (d - 1)
```

说明：
- `forbidden` 对称（$|\operatorname{corr}|$ 与 $|\operatorname{pcorr}|$ 均对称），`allowed_offdiag` 自然包含 (i, j) 和 (j, i) 两个方向。
- 对角线永不屏蔽 —— 即使自相关为 1，对角线更新也必须保留（用于残差方差调节）。
- 要求 $S$ 为**协方差矩阵**（或样本 Gram 矩阵除以 $n$），保证 `np.diag(S)` 给出的是方差；`pcorr` 模式下还要求 $S$ 可逆（加 $10^{-6} I$ 防病态）。
- 阈值 $\tau$ 的取值尺度两种模式一致（都在 $[0, 1]$ 范围），可直接套用相同的扫描区间。

---

## 4. 对角线 vs off-diag 采样概率

### 4.1 基线：原算法比例

原算法每步均匀采样 (i, j)：

$$
\mathbb{P}(\text{diag}) = \frac{d}{d^2} = \frac{1}{d}, \quad
\mathbb{P}(\text{off-diag 任一对}) = \frac{1}{d^2}
$$

$d = 20$ 时 diag 占 5%，off-diag 占 95%。

### 4.2 两种模式

#### 模式 `"preserve"`（默认）—— 保持原比例

diag 采样概率仍为 $1/d$，off-diag 从允许集合中均匀采：

```python
if np.random.rand() < 1.0 / d:
    i = j = np.random.randint(d)
else:
    idx = np.random.randint(M)
    i, j = allowed_offdiag[idx]
```

**效果**：
- diag 更新频率与原算法**完全一致**。
- 每个允许的 off-diag 对被采到的概率从 $1/d^2$ 提升到 $(d-1)/(dM)$。
- **加速因子** $= d(d-1)/M$；例如 $d=20, M=40$ 时约 9.5×。

#### 模式 `"pool"` —— 池化采样

合并 $d$ 个对角位 + $M$ 个允许 off-diag 位，直接均匀采：

```python
total = d + M
r = np.random.randint(total)
if r < d:
    i = j = r
else:
    i, j = allowed_offdiag[r - d]
```

**效果**：
- $\mathbb{P}(\text{diag}) = d/(d+M)$，$\mathbb{P}(\text{off-diag 任一对}) = 1/(d+M)$。
- 实现最简；但当 $M$ 较小时 diag 被过度采样（$d=20, M=40$ 时 diag 占 33%）。

### 4.3 选择准则

| 场景 | 推荐模式 | 原因 |
|---|---|---|
| $\tau$ 较小（$M > d(d-1)/4$） | 二者差异小 | diag 比例不会被严重放大 |
| $\tau$ 较大（$M$ 接近 $d$ 量级） | `"preserve"` | 避免 diag 主导步数，拖慢结构恢复 |
| 需要与原算法可对比的收敛曲线 | `"preserve"` | history 形态最接近原版 |
| 最小化"空转步"、不在意节拍 | `"pool"` | 每步期望贡献最大 |

### 4.4 未来扩展：`"scheduled"` 模式

若实验观察到收敛呈"结构恢复 + 尺度精调"两阶段，可考虑动态调权（例如前半段 diag 概率 $1/d$，后半段放大到 $3/d$）。**不在本次首版实现范围内**，视 `"preserve"` 的实验结果决定是否加入。

---

## 5. API 设计

### 5.1 新增参数（在 `dag_coordinate_descent_l0` 中）

```python
def dag_coordinate_descent_l0(
    S, T=100, seed=0, threshold=0.05, lambda_l0=0.2,
    return_history=False, return_graph_history=False, A_init=None,
    early_stop=False, check_every=None, tol=1e-4, patience=10, min_steps=None,
    # --- 新增 ---
    faithfulness_tau: float = 0.0,          # 0 => 禁用，退化为原算法
    sampling_mode: str = "preserve",        # "preserve" | "pool"
    screening: str = "corr",                # "corr" | "pcorr"
):
```

**`screening` 参数语义**：

| 取值 | 判据 | 推荐使用场景 |
|---|---|---|
| `"corr"`（默认） | 样本相关矩阵 $\lvert\operatorname{corr}\rvert$ | 首版、快速验证、$S$ 奇异或近奇异 |
| `"pcorr"` | 偏相关矩阵 $\lvert\operatorname{pcorr}\rvert$（来自 $S^{-1}$） | 剪枝更激进、高维稀疏 DAG、$n \gg d$ |

### 5.2 向后兼容性

- 默认 `faithfulness_tau=0.0` → 完全退化到原算法，`allowed_offdiag = None`，采样逻辑不变。
- 所有现有调用（含 notebook 与 benchmark）**无需修改**即可正常运行。

### 5.3 返回值

保持现有返回形式（根据 `return_history` / `return_graph_history` 组合）。**暂不引入** `meta` 字典 —— 如需诊断信息（`M`、`τ`、`sampling_mode`），后续可作为独立 issue 处理。

---

## 6. 正确性与收敛性讨论

### 6.1 每步有效性

模式 `"preserve"` 和 `"pool"` 都**每步执行一次实质更新**（不 `continue` 跳过），因此：
- `history` 长度严格等于 `t + 1`，与原算法一致。
- `early_stop` 的 `check_every` 节拍无需修改。

### 6.2 坐标可达性

只要 $M > 0$，任意允许的 off-diag 对及任意 diag 位都有正采样概率，坐标下降的全局收敛保证仍成立（在非退化情形下收敛到局部最优）。

### 6.3 漏边风险

被屏蔽的 (i, j) 对在整个运行中**永远不会被采样到**，因此：
- `screening="corr"`：若真实模型存在 $x_i \to x_j$ 但 $\operatorname{corr}(x_i, x_j) \approx 0$（路径相消），该边会被漏掉。
- `screening="pcorr"`：若真实模型存在 $x_i \to x_j$ 但 $\operatorname{pcorr}(x_i, x_j \mid \text{rest}) \approx 0$，该边会被漏掉。这同样对应"路径相消"类的退化情形 —— 对线性高斯 SEM，$\Omega_{ij} = 0$ 当且仅当 $(i, j)$ 不在 moral graph 中，对真实有向边仅在数值相消时成立。
- **两种判据对真实有向边的漏边机制理论上等价**，都属于测度 0 事件；但在有限样本下 `pcorr` 对**小权重真实边**（接近数值边界）的屏蔽概率略高于 `corr`，因为 $\Omega$ 的估计噪声通常大于 $S$。
- 风险大小随 $\tau$ 单调递增。

### 6.4 `update_off_diagonal` 清零语义的影响

原 `update_off_diagonal` 会先清零 A[i,j] 和 A[j,i]，再决定是否加回。由于本设计采用策略 B（从允许集合中采样），被屏蔽的对**根本不会进入** `update_off_diagonal`，因此不存在"屏蔽边但被强制清零"的矛盾。

对于 $A_\text{init} \neq I$ 的情形，若 $A_\text{init}$ 中已存在边 $x_i \to x_j$ 但 $(i,j)$ 被屏蔽，该边会保留在最终解中（不被采样即不被触碰），但算法不会再有机会重优化它。使用者需要在传入 `A_init` 和 `faithfulness_tau` 时注意这一点。

---

## 7. 预期收益与代价

### 7.1 收益

- **高维加速**：在 ER 稀疏图 + 高维（$d \geq 30$）场景下：
  - `screening="corr"`：$|\operatorname{corr}|$ 矩阵稀疏度中等，$M / d(d-1)$ 通常 20-30%。每步有效率提升 3-5×。
  - `screening="pcorr"`：$|\operatorname{pcorr}|$ 对应 moral graph 稀疏度，$M / d(d-1)$ 可能低至 5-10%（moral graph 本身就是稀疏 DAG 的紧致超集）。每步有效率提升 10× 以上。
- **低维无损**：$\tau = 0$（默认）时行为与原算法完全一致，零开销。

### 7.2 代价

- **路径相消漏边**：如 §2.2 所述。$\tau$ 越大代价越高。
- **$\tau$ 调参**：需要针对问题规模和样本量做扫描。建议初始范围 $\tau \in \{0, 0.02, 0.05, 0.1\}$。
- **一次性预处理成本**：
  - `corr`：$O(d^2)$ 构造相关矩阵；
  - `pcorr`：$O(d^3)$ 求逆 $S^{-1}$，d=100 时约 10ms；
  - 相对于 $T \cdot O(d^2)$ 的主循环均可忽略。
- **pcorr 的数值要求**：`pcorr` 需要 $S$ 非奇异；$n < d$ 或多重共线性严重时可能出错。已在求逆前加 $10^{-6} I$ 正则化，但极端情况下应退回 `corr` 模式。

---

## 8. 实验计划

### 8.1 功能验证

1. **路径相消小例**（$d = 4$）：构造 $A \to B \to C \to D$ + $A \to D$ 且两路径恰好相消的数据，验证 $\tau > 0$ 时确实漏掉 $A \to D$，而 $\tau = 0$ 时能恢复。
2. **τ = 0 回归测试**：在现有 ER benchmark 数据上验证 $\tau = 0$ 时结果与原算法**逐字节一致**（相同 seed 下 `A`、`history` 完全相同）。

### 8.2 性能扫描

复用 [test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb](../experiments/notebooks/test/test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb) 的实验设定：

- $d \in \{20, 30, 50\}$
- $n = 20000$
- $\tau \in \{0, 0.02, 0.05, 0.1\}$
- `screening` $\in$ \{`"corr"`, `"pcorr"`\}
- `sampling_mode` $\in$ \{`"preserve"`, `"pool"`\}
- 每组 $\geq 5$ trials

指标：
- SHD / CPDAG-SHD / MEC-hit
- 达到 `early_stop` 所需的 $T$
- wall-clock
- $M / d(d-1)$（被保留的 off-diag 对比例，分别记录 `corr` 与 `pcorr` 两种下的值）
- 漏边率：在已知 ground truth 的 ER 图上，被 `forbidden` 标记但实际存在有向边的对数

### 8.3 与 greedy_cd_A 的横向对比

当前 [test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb](../experiments/notebooks/test/test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb) 已显示 `cd_A_noepoch` 在 SHD 上显著优于 `greedy_cd_A`（d=20 时 8.8 vs 29.4）。引入 $\tau$ 后应进一步检查：`cd_A_noepoch + weakfaith` 能否在 runtime 上追平 greedy 同时保持 SHD 优势。

---

## 9. 实施顺序

1. **先写文档**（本文件）。
2. **实现** `dag_coordinate_descent_l0` 中 `faithfulness_tau` / `sampling_mode` 参数。保留现有函数签名向后兼容。
3. **回归测试**：$\tau = 0$ 时与原算法一致性检查。
4. **新建实验 notebook**：`test_er_cd_A_weakfaith.ipynb`，按 §8 设计扫描。
5. **根据实验结果**决定：
   - 是否引入 `"scheduled"` 模式；
   - 是否需要 `meta` 返回字典；
   - 是否给 `min_steps` 默认值加入 $M$ 相关的缩放。

---

## 10. 实验结果（ER 基准）

> 实验笔记本：[`experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb`](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)
> 结果 CSV：`experiments/results/er_cd_A_weakfaith_benchmark_summary.csv`

### 10.1 实验设定

| 参数 | 值 |
|---|---|
| 图类型 | ER，度 2.0 |
| 噪声 | `gaussian_nv`，$B_{\text{scale}} = 1.0$ |
| $d$ | 20, 30, 50 |
| $n$ | 20 000 |
| 每组试验数 | 10 |
| $\lambda_{L0}$ | 0.2 |
| $T$（步数预算） | 100 000 |
| 早停 | 开启（tol=1e-4, patience=10） |
| 阈值 | 0.05 |

测试 6 个算法变体：

1. **baseline**：原 `coordinate0.dag_coordinate_descent_l0`
2. **wf_corr_tau0.02_preserve**：corr 筛选, $\tau=0.02$, preserve 采样
3. **wf_corr_tau0.05_preserve**：corr 筛选, $\tau=0.05$, preserve 采样
4. **wf_pcorr_tau0.02_preserve**：pcorr 筛选, $\tau=0.02$, preserve 采样
5. **wf_pcorr_tau0.05_preserve**：pcorr 筛选, $\tau=0.05$, preserve 采样
6. **wf_corr_tau0.05_pool**：corr 筛选, $\tau=0.05$, pool 采样

### 10.2 主要指标汇总

| 算法 | $d$ | SHD ↓ | CPDAG-SHD ↓ | MEC 匹配率 ↑ | 运行时间 (s) | 实际步数 (mean) | mask keep ratio |
|---|---|---|---|---|---|---|---|
| baseline | 20 | 9.9 | 17.4 | 10% | 3.2 | 12 096 | 1.00 |
| baseline | 30 | 26.0 | 53.2 | 0% | 14.3 | 43 152 | 1.00 |
| baseline | 50 | 32.9 | 64.9 | 0% | 50.3 | 86 593 | 1.00 |
| **wf_pcorr_tau0.05_preserve** | **20** | **2.4** | **5.3** | **60%** | 2.5 | 10 521 | 0.17 |
| **wf_pcorr_tau0.05_preserve** | **30** | **5.2** | **7.3** | **40%** | 10.3 | 32 736 | 0.12 |
| **wf_pcorr_tau0.05_preserve** | **50** | **7.5** | **12.8** | **10%** | 44.1 | 78 123 | 0.06 |
| wf_pcorr_tau0.02_preserve | 20 | 4.2 | 9.1 | 50% | 2.7 | 10 521 | 0.18 |
| wf_pcorr_tau0.02_preserve | 30 | 6.1 | 8.3 | 20% | 10.8 | 34 643 | 0.12 |
| wf_pcorr_tau0.02_preserve | 50 | 8.0 | 9.5 | 10% | 43.3 | 77 248 | 0.07 |
| wf_corr_tau0.05_preserve | 20 | 7.7 | 16.5 | 20% | 2.6 | 10 794 | 0.27 |
| wf_corr_tau0.05_preserve | 30 | 18.2 | 34.2 | 0% | 10.2 | 35 294 | 0.25 |
| wf_corr_tau0.05_preserve | 50 | 20.0 | 37.4 | 0% | 41.9 | 81 110 | 0.14 |
| wf_corr_tau0.02_preserve | 20 | 9.0 | 17.6 | 30% | 2.4 | 10 563 | 0.28 |
| wf_corr_tau0.02_preserve | 30 | 17.8 | 35.2 | 0% | 9.2 | 31 806 | 0.25 |
| wf_corr_tau0.02_preserve | 50 | 23.5 | 45.6 | 0% | 42.6 | 81 748 | 0.15 |
| wf_corr_tau0.05_pool | 20 | 11.3 | 20.4 | 10% | 1.4 | 6 426 | 0.27 |
| wf_corr_tau0.05_pool | 30 | 17.1 | 31.2 | 0% | 4.7 | 16 880 | 0.25 |
| wf_corr_tau0.05_pool | 50 | 28.1 | 48.0 | 0% | 16.1 | 32 768 | 0.14 |

### 10.3 关键发现

#### (1) pcorr 筛选大幅提升结构恢复精度

偏相关 (`pcorr`) 筛选在所有 $d$ 下均显著优于 baseline 和 corr 系列：
- $d = 50$ 时 SHD 从 32.9 (baseline) 降至 **7.5** (`pcorr_tau0.05_preserve`)，降幅 **77%**。
- CPDAG-SHD 从 64.9 降至 **12.8**，降幅 **80%**。
- MEC 匹配率在 $d = 20$ 时达到 **60%**（baseline 仅 10%）。

这证实了 §2.4 的理论分析：pcorr 筛选利用了精度矩阵的稀疏性，屏蔽了"仅通过中介间接相关"的变量对，使搜索集中在 moral graph 中的真实候选边上。

#### (2) corr 筛选改善有限但仍正向

corr 系列相对 baseline 有一定改善（$d = 50$ 时 SHD 从 32.9 降至 20.0），但远不及 pcorr。原因在于 corr 仅能屏蔽"完全无边际关联"的变量对，保留了大量间接关联对。

#### (3) pool 采样：速度最快但精度略差

`wf_corr_tau0.05_pool` 运行时间最短（$d = 50$ 时 **16.1s** vs baseline **50.3s**，加速 **3.1×**），但 SHD 与 baseline 持平（28.1 vs 32.9），劣于 preserve 模式（20.0）。pool 模式的 diag 采样比例过高（搜索空间小时 diag 占比达 1/3），导致结构恢复不充分。

符合 §4.3 的预测：当 $M$ 接近 $d$ 量级时，应使用 `preserve` 模式。

#### (4) 运行时间：preserve 模式无显著加速

preserve 模式的运行时间与 baseline 接近（$d = 50$：41-44s vs 50s），因为 preserve 下每步的 `update_off_diagonal` 成本不变，只是搜索集中度提高了。加速体现在**更少的步数即可收敛**（实际步数更少或步数相近但结果更好），而非壁钟时间的缩减。

#### (5) mask 稀疏度随 $d$ 递增

| 筛选方式 | $d=20$ | $d=30$ | $d=50$ |
|---|---|---|---|
| corr ($\tau=0.05$) | 27% | 25% | 14% |
| pcorr ($\tau=0.05$) | 17% | 12% | 6% |

pcorr 在 $d=50$ 时仅保留 **6%** 的候选对，搜索空间压缩约 **16×**。随维度增大剪枝更激进，符合稀疏 DAG 下 moral graph 边数 $O(d)$ 而全对数 $O(d^2)$ 的渐近预期。

### 10.4 结论与建议

1. **推荐默认配置**：`screening="pcorr"`, `faithfulness_tau=0.05`, `sampling_mode="preserve"`。该配置在所有测试维度下均实现最佳或接近最佳的 SHD / CPDAG-SHD，且运行时间无显著增加。
2. **追求速度时**：`screening="corr"`, `sampling_mode="pool"` 可获得 3× 加速，但精度无改善。
3. **$\tau$ 选择**：0.05 略优于 0.02（更激进的剪枝在测试范围内未导致更多漏边），但更大的 $\tau$（如 0.1）未测试，需谨慎。
4. **pcorr 的适用条件**：$n \gg d$ 时表现最佳（本实验 $n/d = 400$）。$n$ 接近 $d$ 时精度矩阵估计噪声增大，可能需退回 corr 或降低 $\tau$。

---

## 11. 参考

[^1]: Ramsey et al., "A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images." International Journal of Data Science and Analytics, 2017.
[^2]: 实验笔记本 [`test_er_cd_A_weakfaith_benchmark.ipynb`](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)，2026-04-18 运行。
