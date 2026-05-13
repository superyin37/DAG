# CD_A / CD_A_weakfaith 改进方向汇总

> 目标：精度（CPDAG-SHD / Skeleton P/R）超过 PC, FGES, NOTEARS, GOLEM-NV-l1, CALM 这套 baseline。

---

## 1. 背景与现状诊断

### 1.1 当前 benchmark 基线（50-node ER1, n=20000, noise_ratio=16, no standardization, causaldag-SHD）

| 算法 | CPDAG-SHD | Skel-P | Skel-R | n_edges_est (true=50) | runtime/trial |
|---|---|---|---|---|---|
| PC                | **10.4** | 0.99 | 0.98 | 65 | ~4s |
| FGES              | **13.0** | 0.95 | 1.00 | 66 | ~4s |
| NOTEARS           | **12.4** | 0.93 | 0.99 | 54 | ~520s |
| CALM              | 43.8 | 0.97 | 0.53 | 28 | ~140s |
| **cd_A_weakfaith**| 15.4 | 0.95 | 0.92 | 49 | ~66s |
| **cd_A**          | 62.6 | 0.69 | 0.79 | 57 | ~68s |

要击穿 PC/FGES (10–13)，cd_A_weakfaith 需要再降 ~3–5 个 SHD；cd_A 缺口更大 (~50)。

### 1.2 失分模式

- **cd_A**：precision=0.69 → ~30 条假边；纯随机采样浪费迭代
- **cd_A_weakfaith**：precision/recall 都高，但 CPDAG-SHD 显著大于有向 SHD → **方向错配**贡献了大半误差（v-structure 错认 / 单边方向反）

### 1.3 算法核心结构（供下文引用）

`coordinate0.dag_coordinate_descent_l0` 主循环：

```python
for t in range(T):
    i, j = np.random.choice(d, 2, replace=True)        # 均匀随机采样
    if i == j:
        A = update_diagonal(A, S, i, ...)              # 对角元更新（封闭解）
    else:
        A = update_off_diagonal(A, S, i, j, λ, ...)   # 非对角元贪心 + DAG check
    history.append(f(A, S))
```

cd_A_weakfaith 在此基础上加了一阶 faithfulness screening：把 |corr| 或 |pcorr| < τ 的 (i,j) 从采样池中剔除，节省迭代。

---

## 2. 改进方向（共 30 项，按 8 大类组织）

每项标注：**收益**（高/中/低 = 预期 CPDAG-SHD 降幅）+ **改动量**（小/中/大 = LOC 估算）+ **理由**。

---

### 2.1 评分函数 (Score function)

cd_A 现在用 `f(A,S) = LS_loss + lambda_l0 * #edges`，`lambda_l0=0.2` 是手调死数。

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **S1** | BIC 替代纯 L0 | 高 | 中 | `loss + (log n / 2) · #params`。penalty 自动随 n 调；与 FGES 的评分一致；λ 不再是超参 |
| **S2** | 非等方差似然 (GOLEM-NV) | 中 | 中 | 当前是等方差 LS。改成 `-log\|det(I-A)\| + 0.5 Σ log σ_j² + 0.5 Σ resid_j²/σ_j²`，对 noise_ratio=16 的非等方差数据更合身 |
| **S3** | λ 路径 warm-start | 中 | 小 | 跑 λ ∈ [大, 小] 序列，每个 warm-start 上一个解；通常比单 λ 解稳定 |
| **S4** | CV / AIC 选 λ | 中 | 小 | 把 λ 从手调超参变成数据自适应；2-fold CV 估计泛化误差 |
| **S5** | L0 + 小 L2 (elastic net) | 低 | 小 | L2 防止权重爆掉；对稠密图 / 高噪声有用。L0 不变，加个 ridge term |
| **S6** | 数据相关 λ scaling | 低 | 小 | λ 根据 `np.diag(S)` 缩放（每个变量自己的方差），不再全局统一 |

---

### 2.2 搜索策略 (Search strategy) ⭐ 核心

均匀随机采样 (i,j) 从 d²=2500 个坐标里抽，大部分坐标在已收敛后是无价值的。

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **R1** | **重要性采样** | **高** | 中 | P((i,j)) ∝ \|gain_{ij}\| 的某个粗略估计；可以每 K 步重计算一次 gain，每步按其分布采样 |
| **R2** | Greedy + 随机混合调度 | **高** | 小 | 项目已有 `cd_greedy_A.py`。前 X 步走 greedy（每步选最大 gain 坐标），之后切随机或保持 greedy；试 cosine/linear 调度 |
| **R3** | Active set | 中 | 中 | 维护 "近期被改动 + 它们邻居" 的活跃集合；在活跃集内采样；周期性 refresh |
| **R4** | Epoch 化 + 早终 | 中 | 小 | 每个 epoch 把 d²/2 个非对角坐标都过一遍 (顺序可洗牌)；连续 K 个 epoch 改善 < tol 即停 |
| **R5** | **Block coordinate descent** | **高** | 大 | 每次更新一行（一个节点的所有入边）；给定父集合可解析最优；类似 NOTEARS 的 inner solve 但在 binary 上 |
| **R6** | Swap / reverse moves | **高** | 中 | 当前只能 add/remove；显式加 reverse(i→j ↔ j→i)、3-cycle swap (i→j→k 换成 k→j→i) 等跳出局部最优的算子 |
| **R7** | Tabu search | 中 | 中 | 维护近 K 步访问过的 (i,j) tabu list 避免反复振荡 |
| **R8** | Simulated annealing | 中 | 中 | 以 P=exp(-Δf/T) 接受劣化的更新；T 缓慢降温；增强全局探索 |
| **R9** | 平行多步采样 + 选最优 | 中 | 中 | 每步从 K 个候选 (i,j) 选 gain 最大的；探索 vs 计算开销 trade-off |

---

### 2.3 Skeleton / 初始化

cd_A 默认 `A_init = I`，等价于"空图"开始。weakfaith 的 screening 只用 0/d-2 阶 CI（marginal corr / pcorr）。

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **I1** | **PC-style 多阶 CI 做 skeleton** | **高** | 中 | weakfaith 升级：除 corr/pcorr 外，加 1-2 阶 CI 测试（条件 1-2 个变量）；接近 PC skeleton 阶段的判别力 |
| **I2** | **Warm-start from PC** | **高** | 小 | 用 PC 输出（CPDAG）做 A_init：定向边按方向，未定向边按一个临时方向。10s PC + 30s cd_A refine 仍比 NOTEARS 快得多 |
| **I3** | Warm-start from FGES | **高** | 小 | 同上，用 FGES 输出；FGES 已经是高质量 CPDAG |
| **I4** | Warm-start from GOLEM | 中 | 小 | GOLEM 的连续解 thresholded 做 A_init；接受 GOLEM 的"形状"再用 L0 重打方向 |
| **I5** | Adaptive faithfulness τ | 中 | 中 | 当前 τ=0.05 硬阈值；改成 (a) 噪声水平估计的函数，(b) BH-FDR 控制 false discovery |
| **I6** | 多重 screening 投票 | 中 | 小 | corr / pcorr / glasso / shrunk-cov 各出一份候选，多数票决定 candidate pool |
| **I7** | Topological order 先验 | 中 | 中 | 从数据估一个粗略的 topological order（如按节点方差排序的 varsortability heuristic），偏向那个方向开始搜 |

---

### 2.4 Post-processing refinement

CD 收敛后再做一轮局部修正，常常能再降几个 SHD。

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **P1** | **Hill-climbing on edges** | **高** | 中 | CD 跑完后，对每条边尝试 add/remove/reverse，按 BIC 接受 / 拒绝；类似 GES backward phase |
| **P2** | **Meek's 4 rules** | 中 | 中 | 拿到 skeleton + v-structures 后用 Meek 规则补全方向；CPDAG-SHD 直接降，特别针对 cd_A_weakfaith 的"方向错配" |
| **P3** | 边显著性筛除 | 中 | 小 | 每条估出的边再做一次 partial-correlation test (条件其余父节点)，p > α 的剪掉 |
| **P4** | Stability selection | 中 | 中 | bootstrap K 次，只保留出现 ≥ k/K 次的边；通用降假阳工具 |
| **P5** | DAG → CPDAG 投影 | 中 | 小 | 输出统一转 CPDAG（projection），与 paper Table 5/7 对齐；纯实现细节 |

---

### 2.5 集成 / 多重启

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **E1** | Multi-restart + BIC 选优 | 中 | 小 | 跑 K=10 不同 seed，按 BIC 选最优；项目已有 `test_cd_A_multi_restart` 探索过，**做正式集成** |
| **E2** | **Diversified multi-init** | **高** | 小 | K 个起点不是随机种子，而是从 (I, PC out, FGES out, GOLEM out, varsortability order) 各跑一次；保留 BIC 最低 |
| **E3** | Bagging on bootstrap | 中 | 中 | 每次重抽样数据跑一遍，按 edge frequency 投票；标准 bagging |
| **E4** | **多算法投票集成** | **高** | 小 | cd_A_weakfaith / PC / FGES 各出一份 CPDAG，按 majority 投票合成；几乎稳赢任一单算法（对偶有方差，但平均更好） |
| **E5** | Stacking (元模型) | 中 | 大 | 训一个小 meta-classifier 决定每条边是否保留，特征是各 base 算法的 score / confidence |

---

### 2.6 数据预处理

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **D1** | Shrinkage covariance (Ledoit-Wolf) | 中 | 小 | 替代 sample cov，小 n / 高 d 上更稳定；`sklearn.covariance.LedoitWolf` 一行 |
| **D2** | GraphLasso precision matrix | 中 | 小 | 先估稀疏精度矩阵，再从中导出 cd_A 输入 S |
| **D3** | Robust covariance | 低 | 小 | trimmed / rank-based cov，对噪声异质性更稳定 |
| **D4** | 标准化模式 toggle | 低 | 小 | 把"是否标准化"作为算法显式参数；目前混在外部，对比时易出错 |

---

### 2.7 算法理论侧

| ID | 改进 | 收益 | 改动 | 详细 |
|---|---|---|---|---|
| **T1** | KKT-style optimality certificate | 中 | 中 | 跑完检查每个非对角 (i,j) 的 gain 是否 ≤ 0；否则强制更新最大 gain 那个并继续 |
| **T2** | **CPDAG-level search** | **高** | 大 | 直接在 Markov equivalence class 里搜（类似 GES）；避免在等价 DAG 之间反复横跳 |
| **T3** | Online topological order | ⚡ | 中 | 维护拓扑序列，避免每步重算 acyclicity |
| **T4** | Score-equivalent moves | 中 | 中 | 显式枚举与当前 DAG score 等价的"等分图"，跳到边数最少的那个（regularization 自然收益） |

---

### 2.8 实现 / 计算

不影响精度，但能让你跑更多 trials，间接帮调参。

| ID | 改进 | 加速 | 改动 |
|---|---|---|---|
| **C1** | Numba JIT 内层循环 | 5–10× | 中 |
| **C2** | Cython / C 扩展 | 10–30× | 大 |
| **C3** | 批量 Sherman-Morrison 更新 A_inv | 2–3× | 中 |
| **C4** | GPU (PyTorch) | d > 200 才显著 | 大 |
| **C5** | 共享 dataset cache（避免每 trial 重生成） | 1.x× | 小 |

---

## 3. 优先级建议

按"性价比"（预期收益 / 改动量）排序前 6：

| 阶段 | ID | 改进 | 预期效果 |
|---|---|---|---|
| **第一波（一周内全做完）** | I2 | PC warm-start | 50-ER1 cpdag_shd 估计降到 8 以下 |
|  | E4 | 多算法投票（cd_A_wf + PC + FGES） | 几乎稳赢任一单算法 |
|  | P1 | Hill-climbing post-refine | 主要降"方向错配"，CPDAG-SHD 再降几个 |
| **第二波（核心算法增强）** | R2 | Greedy + 随机混合调度 | 节省 50-70% 迭代 |
|  | I1 | PC-style 多阶 CI skeleton | precision 显著提升 |
|  | S1 | BIC 替代纯 L0 | 评分更可靠，λ 不再调 |
| **第三波（理论 / 性能）** | R5 | Block coordinate descent | 收敛更快，跳出更好 |
|  | R6 | Swap / reverse moves | 跳出局部最优 |
|  | C1 | Numba | 跑更多 trials 不头疼 |

---

## 4. 推荐组合方案 (一个具体的 pipeline 提案)

如果只能选一个组合 push 到 SOTA：

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Skeleton + 初定向                                    │
│   PC (Tetrad, FisherZ, alpha=0.01) → CPDAG_pc                 │
│                                                                │
│ Stage 2: Refine (cd_A_weakfaith on skeleton)                  │
│   - candidate pool = PC skeleton                              │
│   - A_init = PC orientation (warm-start, I2)                  │
│   - score = BIC (S1)                                          │
│   - schedule: 5K greedy + 50K 随机 (R2)                        │
│   - 加 reverse moves (R6)                                     │
│                                                                │
│ Stage 3: Post-refine                                          │
│   - Meek's rules 补全方向 (P2)                                 │
│   - Hill-climbing add/remove/reverse, BIC 接受 (P1)           │
│                                                                │
│ Stage 4: Multi-init 选优                                       │
│   - K=5 个不同 init (I, PC, FGES, GOLEM out, varsort order)   │
│   - 选 BIC 最低 (E2)                                          │
└──────────────────────────────────────────────────────────────┘
```

**预期**：50-ER1 cpdag-shd ≤ 5（即比 PC/FGES 都好），cost ~30s/trial（PC 5s + cd refine 20s + post 5s），仍比 NOTEARS 快一个数量级。

---

## 5. 风险点 / 待解决问题

1. **过拟合 baseline 的风险**：当前优先级是按"在 50-ER1 noise_ratio=16 上击败基线"算的。改 graph_type / sample size / noise_ratio 后，最优组合可能变。**建议每个改进至少在 (50-ER1, 50-ER4, 100-ER1) × (ratio=4, 16) 上验证**。

2. **集成方案的"作弊嫌疑"**：E4 的多算法投票本质上是把别人的劳动成果拿来用，论文化的时候要论证 cd_A_weakfaith 的独特贡献是什么（比如：它在 ensemble 里贡献了什么独有的、PC/FGES 抓不到的边？）。

3. **CPDAG-SHD 的解读陷阱**：cd_A 输出 DAG，转 CPDAG 后某些边变 undirected → CPDAG-SHD 可能莫名升或降。看 SHD/CPDAG-SHD/方向错误数三个指标，避免被单一指标误导。

4. **超参 λ / τ / threshold 的耦合**：S1, S6, I5 几个超参互相影响，单独调 vs 联合调结果可能不同。建议用 CV grid search 而不是逐个手调。

5. **跑得快的 PC/FGES 让 cd_A_weakfaith 显得"贵而不优"**：哪怕 cd_A_weakfaith 把精度做到第一，PC/FGES 4s 一次的速度让它在工程实用性上仍占下风。**论文化时要找到 cd_A_weakfaith 独有的优势场景**：比如非线性 SEM？mixed data type？小样本下的 CI test 不可靠？

6. **复现 paper 数字的优先级**：在大改算法前，**先把 calm_subproblem_iter 调回 40000、把 noise_scale generation 对齐 paper**（见 `D:\tmp\CALM-inspect` 与 `calm_dataset.py` 的差异），保证 baseline 数字本身可信，避免"在错的对照上优化"。

---

## 6. 实验计划骨架（实现阶段用）

每个改进项做 ablation 时建议遵守：

1. **基准固定**：cd_A_weakfaith default (T=100K, λ=0.2, τ=0.05, pcorr screening, preserve mode)
2. **逐项启用**：一次只改一个 knob，记录 (CPDAG-SHD ± std, runtime)
3. **多 seed 平均**：至少 5 trials，最好 10
4. **多 setting 验证**：(50-ER1, 50-ER4, 100-ER1) × (ratio=4, 16) 6 个 cell 都跑
5. **结果落到 CSV**：复用 `experiments/results/` 下的命名约定，标 `tag` 区分

---

## 7. 已经在项目里探索过的相关工作

避免重复造轮子，先看：

- `coordinate_descent/cd_greedy_A.py` — greedy 变种（R2 直接用得上）
- `coordinate_descent/cd_A_weakfaith.py` — 当前 weakfaith 实现
- `coordinate_descent/cd_A_l1.py` — L1 版本
- `experiments/notebooks/test/test_cd_A_multi_restart.ipynb` — multi-restart (E1)
- `experiments/notebooks/test/test_cd_A_early_stop.ipynb` — 早停研究
- `experiments/notebooks/test/test_cd_A_score_shd_relation.ipynb` — score 与 SHD 关系
- `experiments/notebooks/test/test_cd_A_variants_shd_vs_tier.ipynb` — 变体对比
- `experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb` — weakfaith benchmark
- `docs/weak_faithfulness_cd_A_noepoch_zh.md` — weakfaith 设计文档

---

*文档版本：v1。日后补充实验结果时直接在对应小节追加。*
