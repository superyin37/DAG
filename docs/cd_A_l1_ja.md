# L1 罰則版 CD-A アルゴリズム設計

**対象ファイル**: [`coordinate_descent/cd_A_l1.py`](../coordinate_descent/cd_A_l1.py)
**作成日**: 2026-04-25
**対応アルゴリズム**: `dag_coordinate_descent_l1`（実験ノートブック中では `cd_A_l1` と表記）
**関連ドキュメント**: [weak_faithfulness_cd_A_noepoch_zh.md](weak_faithfulness_cd_A_noepoch_zh.md)（weakfaith マスク機構を共有）

---

## 1. 目的と動機

`coordinate0.dag_coordinate_descent_l0`（以下 **L0 版**）は、各非ゼロ辺に対し
一律の定数コスト $\lambda_0$ を課す L0 罰則を用いている：

$$
F_{L0}(A) = f(A) + \lambda_0 \sum_{i \neq j} \mathbf{1}\{A_{ij} \neq 0\}
$$

L0 の利点は「活性化／非活性化」が綺麗な二値判定になる点。一方、欠点は次の三つ：

1. **辺の大きさを区別しない** —— 微小な $A_{ij}$ も顕著な $A_{ij}$ も
   同じコストを支払う。
2. **不連続** —— GLasso、NOTEARS-L1 など連続緩和に基づく既存研究と
   直接比較できない。
3. **収縮しない** —— 一度活性化されると無罰則最適値 $\delta^*$ をそのまま
   保持し、推定値が OLS 寄りになり、稀薄化された shrunk 値ではない。

**L1 版** ではこれを次の形に置き換える：

$$
F_{L1}(A) = f(A) + \lambda_1 \sum_{i \neq j} |A_{ij}|
$$

各辺の罰則が重みの絶対値に比例するため、soft-thresholding 風の収縮が
自動的に実現される。本ドキュメントでは L1 部分問題の厳密閉形式解
（設計案で「経路 B」とした方）と、既存の weakfaith マスク機構との
統合方法を述べる。

---

## 2. 部分問題と目的関数

### 2.1 単座標更新の代数

$A$ を固定し、位置 $(i, j)$ に摂動 $\delta E_{ij}$ を加える。rank-1 行列式公式と
トレース展開を使う（L0 版 [`coordinate0.delta_star`](../coordinate_descent/coordinate0.py#L23-L50)
と同一の係数を共有）：

$$
f(A + \delta E_{ij}) - f(A)
= -2 \log(1 + \delta\alpha) + 2\delta b + \delta^2 c
$$

ここで

$$
\alpha = (A^{-1})_{ji}, \quad
b = (S A)_{ij} = S_{i,:} A_{:,j}, \quad
c = S_{ii}.
$$

L1 罰則を加える（`update_off_diagonal_l1` 呼び出し時に $A_{ij}$ は
事前にゼロ化されているため、罰則は $\lambda_1 |\delta|$）：

$$
g(\delta) = -2\log(1+\delta\alpha) + 2\delta b + \delta^2 c + \lambda_1 |\delta|
$$

定義域：$1 + \delta\alpha > 0$（$\det$ の符号反転を防ぎ、log を計算可能に保つ）。

### 2.2 KKT 条件

$g$ は $\delta \neq 0$ で微分可能、$\delta = 0$ では劣微分を用いる。
$\sigma = \operatorname{sign}(\delta)$ とおき、$\delta > 0$ と $\delta < 0$ それぞれで：

$$
g'(\delta) = -\frac{2\alpha}{1+\delta\alpha} + 2b + 2c\delta + \lambda_1 \sigma = 0
$$

両辺に $(1 + \delta\alpha)$ を掛けて整理すると、$\sigma$ に依存する二次方程式：

$$
\boxed{\;c\alpha\,\delta^2 + \big(c + \alpha b_\sigma\big)\,\delta + \big(b_\sigma - \alpha\big) = 0\;}
\quad\text{ただし } b_\sigma = b + \sigma \cdot \frac{\lambda_1}{2}.
$$

退化ケース：
- $\alpha \approx 0$：方程式が線形 $c\delta + b_\sigma = 0$ に退化し、
  $\delta = -b_\sigma / c$。
- $c \approx 0$：共分散の対角が異常（データが退化に近い）。
  アルゴリズムは $\delta = 0$ を返す。

### 2.3 候補根のフィルタリング

各分岐は**符号一致**な実数解を高々 1 個しか寄与しない（$g$ は
$\delta > 0$ と $\delta < 0$ の開区間それぞれで狭義凸であるため）。
逐次フィルタリング：

1. **判別式が非負**：$D = (c+\alpha b_\sigma)^2 - 4c\alpha(b_\sigma - \alpha) \geq 0$。
2. **符号一致**：$\operatorname{sign}(\delta) = \sigma$（候補根の実際の符号が
   想定された分岐符号と一致する必要あり。さもなくばその分岐の許容領域外）。
3. **対数の定義域**：$1 + \delta\alpha > 0$。

最終的に、フィルタを通過した候補と $\delta = 0$ をすべて $g$ に代入し、
$g$ を最小にする値を部分問題の最適解とする。両分岐とも実行可能解を
持たない場合（soft-threshold dead zone）には $\delta = 0$ が自然に勝ち、
等価な結果が得られる。

実装は [`delta_star_l1`](../coordinate_descent/cd_A_l1.py#L66-L120)（約 50 行、
全境界ケースを含む）。

### 2.4 L0 部分問題との比較

| 観点 | L0 版 `delta_star` | L1 版 `delta_star_l1` |
|---|---|---|
| 部分問題最適 $\delta$ | 無罰則最適 $\delta^*$ | $\delta_{l1} = \arg\min g(\delta)$ |
| 罰則構造 | 採択時に一括で $\lambda_0$ を支払う | 採択時に $\lambda_1\|\delta\|$（$\delta$ と結合）|
| 候補根の数 | 1 | 高々 2 個 + 零点 |
| 収縮効果 | なし（「ハード活性化」）| あり（小さな $\delta^*$ は soft-threshold で 0 に）|
| $\lambda = 0$ への退化 | $\delta^*$ をそのまま返す | 数値的に `delta_star` と一致（**検証済** $\Delta f \leq 10^{-15}$）|

---

## 3. アルゴリズム構造

### 3.1 単ステップ更新 `update_off_diagonal_l1`

[`coordinate0.update_off_diagonal`](../coordinate_descent/coordinate0.py#L212-L259)
と完全に同じプロトコル：

1. 旧値 $A_{ij}, A_{ji}$ を保存し、両者をゼロ化。Sherman–Morrison で
   $A^{-1}$ を同期更新。
2. 方向 $i \to j$ について：$A + E_{ij}$ が DAG ならば `delta_star_l1` を呼んで
   $\delta_{l1}$ を計算し、$\Delta_{ij} = f(A) - f(A + \delta_{l1} E_{ij}) - \lambda_1 |\delta_{l1}|$
   を求める。
3. 方向 $j \to i$ も同様に $\Delta_{ji}$ を計算。
4. 両者とも $-\infty$（両方向とも DAG 制約を破る）の場合 → ゼロ化のまま。
5. それ以外は $\Delta$ の大きい方向を採択し、SM で $A^{-1}$ を同期更新。

**L0 版との重要な差異**：

- L0 版は採択／棄却を `Δ ≥ 0` で明示判定（$\delta^*$ は L1 部分問題の
  最適解ではないため、全体として $\delta = 0$ より劣る可能性がある）。
- L1 版の $\delta_{l1}$ は $\{0\} \cup \{\text{両分岐の根}\}$ 上で既に
  全域最小であるため、$\Delta \geq 0$ は**構成上保証される**。したがって
  L1 版のメインループには「全体として劣るなら更新しない」という
  追加チェックは不要 —— soft-threshold が非活性化を選ぶ場合は
  $\delta_{l1} = 0$ となり、自然に「$A$ が変化しない」結果になる。

### 3.2 メインループ `dag_coordinate_descent_l1`

シグネチャは [`cd_A_weakfaith.dag_coordinate_descent_l0_weakfaith`](../coordinate_descent/cd_A_weakfaith.py#L88-L218)
と**完全に並行**：

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

挙動上の唯一の違い：

- 偏対角更新が `update_off_diagonal_l1` を呼ぶ（`update_off_diagonal` ではなく）。
- `history` が記録するのは **L1 罰則付き目的関数** $f(A) + \lambda_1 \|A\|_{1, \text{off}}$
  （ヘルパー [`f_l1`](../coordinate_descent/cd_A_l1.py#L173-L179) で計算）。
  これにより単調減少を直接観察できる。

### 3.3 weakfaith マスクの再利用

L1 版は [`cd_A_weakfaith._build_faithfulness_mask`](../coordinate_descent/cd_A_weakfaith.py#L72-L132)
を直接 import して再利用する。すべての screening オプション
（`corr` / `pcorr` / `glasso` / リスト合成 + `combine='union'|'intersect'`）は
**再実装不要**。意味は L0 weakfaith 版と完全に一致：

- `faithfulness_tau = 0` → 一様サンプリングに退化（無マスク L1 版と
  byte-equivalent）。
- `faithfulness_tau > 0` → サンプリングを `_build_faithfulness_mask` が
  返す許可ペアに限定。
- `sampling_mode='preserve'` は $P(\text{diag}) = 1/d$ を保つ；
  `'pool'` は $\{d \text{ 個の対角}\} \cup \{M \text{ 個の許可された偏対角}\}$ 上で一様。

---

## 4. 数値的正当性

[`cd_A_l1.py`](../coordinate_descent/cd_A_l1.py) は公開前に以下 4 組の
検証を通過している（リポジトリの issue 履歴に smoke test の出力あり）：

| テスト | 内容 | 実測結果 |
|---|---|---|
| **退化一致性** | $\lambda_1 = 0$ のとき `delta_star_l1` が `delta_star` と等価 | $\max\|f(A+\delta_{l0}) - f(A+\delta_{l1\|\lambda=0})\| = 1.78\times 10^{-15}$ |
| **KKT 余量** | $\lambda_1 \in \{0.05, 0.2, 0.5, 1.0\}$ で劣勾配 KKT を検証 | $\leq 10^{-14}$ |
| **閉形式 vs グリッド** | 6001 点の細かいグリッド探索と 30 個のランダム $(i,j,\lambda)$ で比較 | gap = 0（閉形式が厳密に劣らない）|
| **目的関数の単調性** | メインループを 2000 ステップ走らせ、$F_{L1}$ 系列を記録 | 実質的な上昇 0 回、最大変動 $\leq 7.1\times 10^{-15}$（roundoff）|

特に 3 つ目は注意が必要：scipy の `minimize_scalar(bracket=...)` は
状況によって最適点を逃して誤って $\delta = 0$ を返すことがある ——
当初これを閉形式解の誤りと誤認したが、実は基準手法の不備だった。
細かいグリッドに切り替えてからは閉形式解が 100% 劣らないことを確認。

---

## 5. パラメータ調整と実用上の注意

### 5.1 $\lambda_1$ のスケール

L1 罰則 $\lambda_1 |\delta|$ と L0 罰則 $\lambda_0$ は同じスケールではない。
おおまかな見積もり：

$$
\lambda_1 \approx \lambda_0 / |\bar A|
$$

ここで $|\bar A|$ は典型的な辺重みの絶対値。我々の合成データでは
`B_scale=1.0` で重みは大体 $0.5$–$1.0$ なので、$\lambda_0 = 0.2$ は
おおよそ $\lambda_1 \in [0.2, 0.4]$ に対応する。ただし L1 の真の最適値は
これより小さくなることが多い（soft-threshold 自体が剪枝効果を持つため）。
実用上は $\lambda_1 \in [0.05, 0.2]$ が一般的。

[test_er_cd_A_l0_vs_l1_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_l0_vs_l1_benchmark.ipynb)
ではデフォルトで $\lambda_1 \in \{0.05, 0.1, 0.2\}$ をスキャンし、
$\lambda_0 = 0.2$ の L0 をアンカーとして比較する設定にしている。

### 5.2 後処理 threshold

L0 は採択後に $\delta^*$ をそのまま保持するため、重みは多くが 0 から
十分離れている。後処理 `threshold = 0.05`
（[`weight_to_adjacency`](../coordinate_descent/coordinate0.py#L85-L97)）で
通常は問題ない。

L1 は soft-threshold の性質上、**小さいが非ゼロ**な辺をいくつか残す。
これらは L1 の最適解においては「罰則で消し切れない」辺だが、
統計的にはノイズであることが多い。L1 の `n_edges_est` が真の辺数を
著しく上回る場合は、threshold を 0.1 まで上げる、あるいは $\lambda_1$ に
比例させて拡大するのが定石。

### 5.3 weakfaith との併用

実験（[test_er_cd_A_weakfaith_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)）
では `pcorr` + $\tau = 0.05$ が L0 上で最も安定した weakfaith 設定だった。
この組み合わせは L1 でも同様に有効で、さらに：

- マスクで弾かれた「必ず 0 の辺」は L1 でも soft-threshold する必要が
  ない —— **節約されたステップを真に貢献し得る辺に集中できる**。
- L1 の $\Delta \geq 0$ という性質により、$\delta_{l1}$ が dead-zone の 0 でも
  古い重みに戻ることはない；weakfaith マスクと完全に直交する。

---

## 6. 経路 A の取捨選択

設計段階では L1 の実装方針として 2 つの経路を検討した：

| 経路 | 考え方 | 実装コスト | 厳密性 |
|---|---|---|---|
| **A：proximal** | 無罰則 $\delta^*$ を計算した後 soft-threshold $\delta_{l1} \approx \operatorname{sign}(\delta^*)\max(\|\delta^*\| - \lambda_1/(2c), 0)$ | ~10 行 | log-det 項を一次近似するため、$\lambda_1$ が大きいとき最適から乖離 |
| **B：厳密閉形式**（本実装） | §2.2 の KKT 二次方程式を解いてフィルタリング | ~50 行 | 部分問題の厳密最適 |

経路 B を選んだ理由：

1. **再現性**：厳密解により $F_{L1}$ が厳密に単調になり、収束解析や
   グラフ作成が容易。
2. **理論的比較可能性**：lasso 系の文献手法と直接比較でき、追加の
   近似誤差を持ち込まない。
3. **コストが許容範囲**：1 回の `update_off_diagonal_l1` あたり候補根の
   評価が 1–2 個増える程度で、$O(1)$ の追加コスト。主コストは依然として
   SM 更新と DAG 検査で、L0 と同オーダー。

将来的に hot-loop 最適化が必要になった場合（例：$T \geq 10^6$）には、
経路 A を高速化バイパスとして残す余地がある。ただし現在のベンチマーク
規模では、経路 B のランタイムは L0 と同オーダー（実測：d=12 / T=2000 で
1 回約 0.4 秒、L0 と同等）。

---

## 7. API クイックリファレンス

```python
from coordinate_descent.cd_A_l1 import (
    delta_star_l1,            # L1 部分問題の閉形式最適 delta
    update_off_diagonal_l1,   # L1 偏対角の単ステップ更新
    f_l1,                     # L1 罰則付き目的関数
    dag_coordinate_descent_l1,  # メインループ（weakfaith フック付き）
)

# 1) 純粋な L1（screening なし）
A, G, obj = dag_coordinate_descent_l1(S, T=100000, seed=0, lambda_l1=0.1)

# 2) L1 + pcorr screening
A, G, obj = dag_coordinate_descent_l1(
    S, T=100000, seed=0, lambda_l1=0.1,
    faithfulness_tau=0.05, screening=['pcorr'],
)

# 3) L1 + 複数 screen の合成（all-zero のときのみ discard）
A, G, obj = dag_coordinate_descent_l1(
    S, T=100000, seed=0, lambda_l1=0.1,
    faithfulness_tau=0.05,
    screening=['corr', 'pcorr', 'glasso'],
    combine='union',
    glasso_alpha=0.01,
)
```

実験エントリポイント：[test_er_cd_A_l0_vs_l1_benchmark.ipynb](../experiments/notebooks/test/test_er_cd_A_l0_vs_l1_benchmark.ipynb)。
