# 貪欲座標降下法による DAG 学習（A-定式化）

## 1. 問題設定

### 1.1 構造方程式モデル

$d$ 次元の観測データ $X \in \mathbb{R}^d$ が線形構造方程式モデル（SEM）に従うとする：

$$X = A^{-1} N, \quad N \sim \mathcal{N}(0, I)$$

ここで $A$ は下三角成分に DAG の辺の重みを持つ行列である。このとき $X$ の精度行列は $\Omega = A^T A$ となる。

### 1.2 目的関数

$n$ 個の観測から得られる標本共分散行列 $S = \frac{1}{n} X X^T$ のもとで、負の対数尤度（定数を除く）は

$$f(A) = -2 \log \det A + \operatorname{tr}(A^T S A)$$

で与えられる。エッジの疎性を制御する $\ell_0$ 正則化項 $\lambda \cdot \|B\|_0$ を加えた目的関数

$$\min_{A \in \mathrm{DAG}} \; f(A) + \lambda \|A\|_{\mathrm{off},0}$$

を最小化することが目標である。

### 1.3 DAG 制約

行列 $A$ の非対角成分 $A_{ij} \neq 0$ は辺 $i \to j$ を表す。$A$ が DAG に対応するとは、対応する有向グラフが有向閉路を含まないことである。

---

## 2. 既存手法との比較

本アルゴリズムは `coordinate0.py` の**ランダム座標降下法**を**貪欲選択**に置き換えたものである。

| 手法 | 座標選択 | 1ステップのコスト |
|------|----------|------------------|
| ランダム CD（T-step） | 一様ランダム | $O(d^2)$ |
| ランダム CD（epoch） | ランダム順列 | $O(d^4)$/epoch |
| **貪欲 CD（T-step）** | 勾配最大ペア | $O(d^2)$ |
| **貪欲 CD（epoch）** | 勾配降順ソート | $O(d^4)$/epoch |

更新ステップ自体は同一であり、違いは**どのペアを選ぶか**のみである。

---

## 3. アルゴリズムの概要

### 3.1 T-step 版（`dag_greedy_A`）

各ステップ $t = 0, 1, \ldots, T-1$ で：

1. **勾配の計算**
$$G^{(t)} = \nabla_A f(A^{(t)}) = -2(A^{(t)})^{-T} + 2SA^{(t)}$$

2. **ペアの選択**（DAG 実行可能性条件のもとで）
$$
(i_t, j_t) \in \arg\max_{i < j} \max\!\left\{|G^{(t)}_{ij}|,\; |G^{(t)}_{ji}|\right\}
$$

3. **ゼロ化**
$$A^{(t+1/2)} \leftarrow A^{(t)}, \quad A^{(t+1/2)}_{i_t j_t} = A^{(t+1/2)}_{j_t i_t} = 0$$

4. **最適方向の選択と更新**

$$
\delta^*_{ij} = \arg\min_\delta f\!\left(A^{(t+1/2)} + \delta E_{ij}\right), \quad
\Delta_{ij} = f(A^{(t+1/2)}) - f(A^{(t+1/2)} + \delta^*_{ij} E_{ij}) - \lambda
$$

両方向について $\Delta_{ij}$, $\Delta_{ji}$ を計算し，より大きい方向に更新する。

### 3.2 Epoch 版（`dag_greedy_A_epoch`）

各 epoch で以下の 2 ブロックを実行する：

- **Block 1（構造ブロック）**：全 $\binom{d}{2}$ ペアを勾配スコア降順に並べ替え，順番に `update_off_diagonal_greedy` を適用。
- **Block 2（スケールブロック）**：全対角成分 $A_{ii}$ を最適化。

---

## 4. 数学的導出

### 4.1 座標の閉形式最適解

$A^{(t+1/2)} + \delta E_{ij}$（$i \neq j$）に沿った 1 次元最小化の解は：

$$c = S_{ii}, \quad b = (SA)_{ij}, \quad \alpha = (A^{-1})_{ji}$$

$$D = (c + \alpha b)^2 - 4\alpha c(b - \alpha), \quad
\delta^* = \frac{2(b - \alpha)}{-(c + \alpha b) - \sqrt{D}}$$

これは $O(d)$ で計算できる（$A^{-1}$ がキャッシュされている場合）。

### 4.2 目的関数の増分更新

$A[i,j] \mathrel{+}= \delta$（ランク-1 摂動 $A \leftarrow A + \delta e_i e_j^T$）に対して：

**行列式項**（行列式補題より）：
$$\Delta(-2\log\det A) = -2\log(1 + \delta \alpha), \quad \alpha = A^{-1}[j,i]$$

**トレース項**（$S$ が対称であることを利用）：
$$\Delta\operatorname{tr}(A^T S A) = 2\delta (SA)_{ij} + \delta^2 S_{ii}$$

ただし $(SA)_{ij}$ は，キャッシュ済みの $G$ から
$$
(SA)_{ij} = \frac{G_{ij} + 2A^{-1}[j,i]}{2}
$$
と $O(1)$ で回収できる。

**合計**：$\Delta f = -2\log(1 + \delta\alpha) + 2\delta(SA)_{ij} + \delta^2 S_{ii}$

この式により，`f(A, S)` を $O(d^2)$ で再計算することなく，改善量 $\Delta = -\Delta f - \lambda$ を **$O(1)$** で評価できる。

### 4.3 Sherman–Morrison による逆行列の更新

$A \leftarrow A + \delta e_i e_j^T$ のとき：

$$A^{-1}_{\mathrm{new}} = A^{-1} + c \cdot (A^{-1}[:,i])(A^{-1}[j,:]), \quad c = \frac{-\delta}{1 + \delta A^{-1}[j,i]}$$

更新コスト：**$O(d^2)$**（外積計算）。

### 4.4 勾配行列の増分更新

$G = -2A^{-T} + 2SA$ の変化量：

**Term 1**（$-2A^{-T}$ の変化，SM 更新の転置）：
$$\Delta G_{\mathrm{term1}} = -2c \cdot (A^{-1}[j,:])^T (A^{-1}[:,i])^T \quad \text{（ランク-1 外積，}O(d^2)\text{）}$$

**Term 2**（$2SA$ の変化，$S$ 対称より第 $j$ 列のみ変化）：
$$\Delta G_{\mathrm{term2}}[:,j] = 2\delta \cdot S[:,i] \quad \text{（列更新，}O(d)\text{）}$$

これにより $G$ を毎ステップ $O(d^3)$ で再計算する必要がなくなり，**$O(d^2)$** での増分更新が可能になる。

---

## 5. キャッシュ管理

アルゴリズムは以下の 5 つの状態を常に同期して維持する：

| キャッシュ | 内容 | 更新コスト |
|-----------|------|-----------|
| `A` | 現在の行列 | $O(1)$ |
| `A_inv` | $A^{-1}$，SM 更新 | $O(d^2)$ |
| `G` | 勾配 $\nabla_A f$，ランク-1 更新 | $O(d^2)$ |
| `f_val` | $f(A)$，解析公式 | $O(1)$ |
| `adj` | 有向辺の隣接リスト | $O(1)$ |

これらはすべて `_rank1_update` 1 回の呼び出しで同時に更新される。SM の分母が数値的に不安定（$|1 + \delta\alpha| < 10^{-15}$）になった場合は，フォールバックとして `_recompute_all` により $O(d^3)$ で完全再計算を行う。

Epoch 版では各 epoch 開始時に数値的安定性のため `A_inv`，`G`，`f_val`，`adj` をすべて再計算する。

---

## 6. DAG 実行可能性の判定

### 6.1 従来手法（NOTEARS）の問題点

既存コード（`coordinate0.py`）が使用している NOTEARS 制約：

$$h(W) = \operatorname{tr}(\exp(W \circ W)) - d = 0$$

は行列指数関数の計算を要し，**$O(d^3)$** のコストがかかる。これは連続最適化（勾配法）での微分可能な正則化のために設計されたものであり，2値の判定には過剰なコストである。

### 6.2 新手法：Kahn のトポロジカルソート

辺 $i \to j$ を**一時的に追加**したうえで，グラフ全体に対して Kahn のアルゴリズムを実行する：

```
is_dag_kahn(adj):
    各ノードの入次数を計算
    入次数 0 のノードをキューに追加
    while キューが空でない:
        ノード u をキューから取り出し，処理済みカウントをインクリメント
        u の隣接ノード v の入次数を 1 減らし，0 になればキューに追加
    return 処理済みカウント == d   # 全ノード処理できれば DAG

can_add_edge(adj, i, j):
    adj[i] に j を追加
    result = is_dag_kahn(adj)
    adj[i] から j を削除
    return result
```

不変量に一切依存せず，グラフの現在の状態に関わらず正しく動作する。

計算量：**$O(d + E)$**（$E$ は現在の辺数）。

| 手法 | 計算量 | 不変量依存 | 備考 |
|------|--------|-----------|------|
| NOTEARS expm | $O(d^3)$ | 不要 | 連続最適化向け設計 |
| DFS 到達可能性 | $O(d + E)$ | **必要** | j の可達部分グラフのみ探索 |
| **Kahn トポロジカルソート** | $O(d + E)$ | **不要** | グラフ全体を走査 |

疎なグラフ（$E = O(d)$）では $O(d)$ まで改善される。

---

## 7. 計算複雑度

### 7.1 T-step 版（1ステップあたり）

| 操作 | 複雑度 | 説明 |
|------|--------|------|
| 勾配取得 | $O(1)$ | `G` はキャッシュ済み |
| スコア計算 | $O(d^2)$ | `max(\|G\|, \|G^T\|)` の上三角 |
| ペア選択（argmax × $k$） | $O(k d^2)$ | 不可行ペアをマスクして繰り返し argmax，通常 $k \approx 1$ |
| 実行可能性チェック × $k$ | $O(k(d+E))$ | Kahn，$O(k d^2)$ に比べ無視できる |
| ゼロ化（rank-1 × 2） | $O(d^2)$ | SM + G 更新 |
| $\delta^*$ 計算 | $O(d)$ | 閉形式 |
| $\Delta$ 評価 | $O(1)$ | 増分公式 |
| 辺の追加（rank-1） | $O(d^2)$ | SM + G 更新 |
| **合計** | $\mathbf{O(k d^2)}$ | $k=1$ 時は $O(d^2)$，ソート不要 |

初期化（1回のみ）：$A^{-1}$，$G$，$f$ の計算で $O(d^3)$。

### 7.2 Epoch 版（1 epoch あたり）

| 操作 | 複雑度 | 説明 |
|------|--------|------|
| 精確再計算 | $O(d^3)$ | epoch 開始時のみ |
| スコア計算・ソート | $O(d^2 \log d)$ | $O(d^4)$ に比べ無視できる |
| 構造ブロック（全ペア更新） | $O(d^4)$ | 主要項 |
| スケールブロック（対角更新） | $O(d^3)$ | |
| **合計** | $\mathbf{O(d^4)}$ | ランダム epoch CD と同一 |

Epoch 版の漸近複雑度はランダム CD と変わらないが，**1 epoch あたりに扱う情報の質が高い**（勾配の大きいペアを先に処理）ため，収束が速くなることが期待される。

---

## 8. 実装上の注意点

### 8.1 SM フォールバック

`_rank1_update` は SM の分母 $|1 + \delta\alpha| < 10^{-15}$ を検出した場合，何も変更せずに `False` を返す。呼び出し元は：
1. `A` を直接更新
2. `adj` を手動更新
3. `_recompute_all` で全キャッシュを再計算

という手順でフォールバックする。このケースは実用上ほぼ発生しない。

### 8.2 対角成分の更新

対角成分 $A_{ii}$ は DAG 制約と無関係であり，実行可能性チェックを必要としない。座標の大域最小解を求めるため，いったん $A_{ii} = 0.3$ にリセットしてから $\delta^*$ を適用する（`coordinate0.py` の `update_diagonal` と同一の手順）。

### 8.3 勾配スコアと $\ell_0$ 正則化

ペア選択は勾配の絶対値のみに基づく（$\lambda$ を含まない）。$\lambda$ は改善量 $\Delta = -\Delta f - \lambda$ の評価時にのみ使用される。したがって，勾配が大きくても $|\Delta f| < \lambda$ であれば辺は追加されない。

### 8.4 数値安定性

- `np.log1p(x)` を使用（$|x| \ll 1$ での精度向上）
- Epoch 版では epoch 開始時に `A_inv`，`G`，`f_val` を完全再計算（SM の累積誤差を排除）
- T-step 版では累積誤差が懸念される場合，`_recompute_all` を定期的に呼び出すことを推奨

---

## 9. 関連関数一覧

```
cd_greedy_A.py
├── _build_adjacency(A, threshold)           隣接リストの構築
├── _is_dag_kahn(adj)                        Kahn DAG 判定        O(d+E)
├── _can_add_edge(adj, i, j)                 辺追加の実行可能性   O(d+E)
├── _rank1_update(A, A_inv, G, S, i, j, δ,  ランク-1 同期更新    O(d²)
│       f_state, adj)
├── _recompute_all(A, S, A_inv, G, f_state)  フォールバック再計算 O(d³)
├── _pair_scores(G)                          スコア行列の計算      O(d²)
├── update_off_diagonal_greedy(...)          非対角更新            O(d²+d+E)
├── update_diagonal_greedy(...)              対角更新              O(d²)
├── dag_greedy_A(S, T, ...)                  T-step 版メイン関数
└── dag_greedy_A_epoch(S, n_epochs, ...)     Epoch 版メイン関数
```
