# SyntheticDataset 与 CALM 数据生成方式对比

本文档说明项目中两套线性 SEM 合成数据生成方式的流程和差异：

- `synthetic_dataset.py`：项目原有生成器，改自 NOTEARS 工具代码。
- `calm_dataset.py`：本项目新增的 CALM 兼容生成器，核心逻辑来自 `https://github.com/kaifeng-jin/CALM/blob/main/simulate_data.py`。

两者都生成形如

```text
X_j = X_pa(j) @ W_pa(j),j + noise_j
```

的线性结构方程数据，但图生成、边数参数、随机数接口和非等方差高斯噪声的参数语义不同。

## 1. SyntheticDataset 流程

入口：

```python
from synthetic_dataset import SyntheticDataset

dataset = SyntheticDataset(
    n=20000,
    d=50,
    graph_type="ER",
    degree=4,
    noise_type="gaussian_nv",
    B_scale=1.0,
    seed=1,
    noise_ratio=2.0,
)
X = dataset.X
W = dataset.B
G = dataset.B_bin
```

整体流程：

1. 用 `simulate_random_dag()` 生成二值 DAG。
2. 用 `simulate_weight()` 给每条边采样权重。
3. 用 `simulate_linear_sem()` 按拓扑序生成观测数据。

### 1.1 图生成

ER 图：

```text
p = degree / (d - 1)
G_und ~ Erdos-Renyi(d, p)
B_bin = lower_triangle(G_und)
B_bin = random_permutation(B_bin)
```

含义：

- `degree` 是期望平均度数的控制量。
- 实际边数是随机的，不固定。
- 依赖 `networkx`。

SF 图：

```text
m = round(degree / 2)
G = Barabasi(d, m, directed=True)
```

### 1.2 权重生成

默认权重范围由 `B_scale` 缩放：

```text
W_ij ~ Uniform(-2 * B_scale, -0.5 * B_scale)
   or Uniform( 0.5 * B_scale,  2 * B_scale)
```

只有 `B_bin_ij = 1` 的位置会保留权重。

### 1.3 噪声生成

`noise_type` 可选：

- `gaussian_ev`
- `gaussian_nv`
- `exponential`
- `gumbel`

其中 `gaussian_nv` 是本项目原有定义：

```text
sigma_j ~ Uniform(1, 2) * noise_ratio
noise_j ~ Normal(0, sigma_j^2)
```

所以 `noise_ratio` 是标准差的整体乘法因子。例如：

```text
noise_ratio = 16
sigma_j in [16, 32]
variance_j in [256, 1024]
```

`dataset.N` 保存的是按拓扑序生成的每个节点噪声标准差。

## 2. CALM 兼容生成器流程

入口：

```python
from calm_dataset import CalmDataset

dataset = CalmDataset(
    n=20000,
    d=50,
    graph_type="ER",
    degree=1.0,
    sem_type="gauss",
    noise_ratio=16.0,
    noise_scale_mode="variance",
    seed=1,
)
X = dataset.X
W = dataset.B
G = dataset.B_bin
noise_scale = dataset.noise_scale
```

也可以直接使用 CALM 风格函数：

```python
from calm_dataset import (
    set_random_seed,
    simulate_dag,
    simulate_parameter,
    simulate_linear_sem,
    make_gaussian_nv_noise_scale,
)

set_random_seed(1)
B_bin = simulate_dag(d=50, s0=50, graph_type="ER")
W = simulate_parameter(B_bin)
noise_scale = make_gaussian_nv_noise_scale(50, 16.0, mode="variance")
X = simulate_linear_sem(W, 20000, sem_type="gauss", noise_scale=noise_scale)
```

整体流程：

1. 用 `simulate_dag()` 生成二值 DAG。
2. 用 `simulate_parameter()` 给每条边采样权重。
3. 构造 `noise_scale`。
4. 用 `simulate_linear_sem()` 按拓扑序生成数据。

### 2.1 图生成

ER 图：

```text
G_und ~ Erdos-Renyi(d, m=s0)
B = lower_triangle(random_permutation(G_und))
B = random_permutation(B)
```

在 `CalmDataset` 包装类中：

```text
s0 = round(degree * d)
```

含义：

- CALM 原函数接收的是 `s0`，即目标边数。
- `CalmDataset` 为了和现有实验参数接近，提供 `degree`，内部换算成 `s0`。
- 对 ER 图，`igraph.Graph.Erdos_Renyi(n=d, m=s0)` 会生成固定边数。
- 依赖 `igraph`。

CALM 还支持：

- `SF`
- `BP`

### 2.2 权重生成

CALM 原始默认范围：

```text
W_ij ~ Uniform(-2, -0.5)
   or Uniform( 0.5,  2)
```

本项目的 `CalmDataset` 额外提供 `b_scale`，默认 `1.0`，用于和 `SyntheticDataset` 的 `B_scale` 行为对齐：

```text
W_ij ~ Uniform(-2 * b_scale, -0.5 * b_scale)
   or Uniform( 0.5 * b_scale,  2 * b_scale)
```

### 2.3 噪声生成

CALM 原始 `simulate_linear_sem()` 没有 `gaussian_nv` 这个分支。它使用：

```python
simulate_linear_sem(W, n, sem_type="gauss", noise_scale=noise_scale)
```

其中 `noise_scale[j]` 直接作为第 `j` 个节点高斯噪声的标准差：

```text
noise_j ~ Normal(0, noise_scale_j^2)
```

为了表达 CALM 论文里常见的 Gaussian-NV 设定，本项目新增：

```python
make_gaussian_nv_noise_scale(d, noise_ratio, mode="variance")
```

默认 `mode="variance"`：

```text
variance_j ~ Uniform(1, noise_ratio)
sigma_j = sqrt(variance_j)
noise_j ~ Normal(0, sigma_j^2)
```

并且当 `d >= 2` 时，会强制两个节点分别取端点 `1` 和 `noise_ratio`，保证本次 trial 覆盖完整噪声区间。

例如：

```text
noise_ratio = 16
variance_j in [1, 16]
sigma_j in [1, 4]
```

如果使用 `mode="std"`：

```text
sigma_j ~ Uniform(1, noise_ratio)
variance_j in [1, noise_ratio^2]
```

## 3. 核心差异

| 维度 | `SyntheticDataset` | CALM 兼容生成器 |
|---|---|---|
| 文件 | `synthetic_dataset.py` | `calm_dataset.py` |
| 原始来源 | NOTEARS 风格工具代码 | CALM `simulate_data.py` |
| 图库 | `networkx` 为主 | `igraph` |
| ER 参数 | `p = degree / (d - 1)` | `s0 = round(degree * d)` |
| ER 边数 | 随机边数 | 固定 `s0` 条无向边再定向 |
| 支持图类型 | `ER`, `SF` | `ER`, `SF`, `BP` |
| 权重范围 | `B_scale * [-2,-0.5] U [0.5,2]` | 默认同 CALM；包装类支持 `b_scale` |
| 随机数接口 | 独立 `np.random.RandomState(seed)` | CALM 原始全局 `np.random.seed(seed)` |
| 高斯等方差 | `gaussian_ev` | `sem_type="gauss"` + 标量 `noise_scale` |
| 高斯非等方差 | `gaussian_nv` 内置 | `sem_type="gauss"` + 向量 `noise_scale` |
| `noise_ratio=16` 的默认含义 | 标准差乘到 `[16,32]` | 方差范围 `[1,16]`，标准差 `[1,4]` |

## 4. gaussian_nv / Gaussian-NV 的关键区别

最容易混淆的是 `gaussian_nv`。

### SyntheticDataset

```text
sigma_j ~ Uniform(1, 2) * noise_ratio
```

`noise_ratio` 是标准差倍数。

### CALM 兼容默认

```text
variance_j ~ Uniform(1, noise_ratio)
sigma_j = sqrt(variance_j)
```

`noise_ratio` 是最大方差。

因此相同的 `noise_ratio` 数值不能直接横向比较。尤其是较大值时，`SyntheticDataset` 的噪声会远大于 CALM 默认解释。

## 5. 实验中如何选择

如果目标是沿用项目旧实验或旧 notebook 的设置，使用：

```python
from synthetic_dataset import SyntheticDataset
```

如果目标是复现 CALM 风格数据，尤其是 Gaussian-NV，使用：

```python
from calm_dataset import CalmDataset
```

或直接调用：

```python
noise_scale = make_gaussian_nv_noise_scale(d, noise_ratio, mode="variance")
X = simulate_linear_sem(W, n, sem_type="gauss", noise_scale=noise_scale)
```

现有 NOTEARS 噪声实验也可以通过参数选择：

```powershell
python experiments/scripts/run_notears_noise_ratio_benchmark.py `
  --data-generator synthetic `
  --noise-type gaussian_nv `
  --noise-ratios 0.5,1.0,2.0
```

```powershell
python experiments/scripts/run_notears_noise_ratio_benchmark.py `
  --data-generator calm `
  --noise-type gaussian_nv `
  --noise-ratios 16 `
  --noise-scale-mode variance
```

独立 CALM benchmark 仍然使用 CALM 兼容路径：

```powershell
python experiments/scripts/run_notears_calm_data_benchmark.py `
  --noise-ratios 16 `
  --noise-scale-mode variance
```

## 6. 建议

- 报告实验结果时明确写出 `data_generator=synthetic` 或 `data_generator=calm`。
- 对 Gaussian-NV 实验，额外报告 `noise_scale_mode`。
- 不要把 `SyntheticDataset(noise_ratio=16)` 与 CALM Gaussian-NV `noise_ratio=16` 当成同等噪声强度。
- 如果要比较两种生成器本身的影响，除了生成器外应尽量固定 `d`, `n`, 权重范围、算法超参数和评估指标。
