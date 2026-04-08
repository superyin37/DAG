# `dag_greedy_A` のバグと有向座標降下法による修正

**対象ファイル**: `coordinate_descent/cd_greedy_A.py`
**作成日**: 2026-03-30

---

## 1. 問題の発見

`dag_greedy_A`（T-step Greedy 座標降下法）を実行すると、スコア関数
$$f(A) = -2\log\det A + \operatorname{tr}(A^\top S A)$$
が初回更新以降まったく減少しないという現象が確認された。

```
step    pair      Delta_ij      Delta_ji     actual_df
-------------------------------------------------------
   0   (2,3)       74.9322       41.1258    -74.932229
   1   (2,3)       74.9322       41.1258      0.000000
   2   (2,3)       74.9322       41.1258      0.000000
   ...（以下同じ）
```

---

## 2. 根本原因

### 2.1 旧アルゴリズムの更新方式（pair-based reset）

`dag_greedy_A` の各ステップは以下の操作を行う。

1. **スコア計算**：無向ペア $(i,j)$ のスコアを $\max(|G_{ij}|,\, |G_{ji}|)$ で定義
2. **ペア選択**：スコアが最大のペアを選ぶ
3. **リセット更新**：$A_{ij}$ と $A_{ji}$ の両方をゼロにクリアし、より改善の大きい方向に辺を追加

### 2.2 デッドロックの発生メカニズム

step 0 でペア $(2,3)$ が選ばれ $A_{2,3}$ が最適化されると：

- 更新後：$G_{2,3} = 0$（方向 $2 \to 3$ の偏微分がゼロ）
- しかし：$G_{3,2} = 115.83$（逆方向の勾配は依然として大きい）

```
Before step 0:  G[2,3] = 118.36,  G[3,2] = 118.36
After  step 0:  G[2,3] =   0.00,  G[3,2] = 115.83
```

スコア $\max(|G_{2,3}|, |G_{3,2}|) = 115.83$ は全ペア中で依然最大のため、step 1 以降も $(2,3)$ が選ばれ続ける。しかし：

$$\text{update} = \underbrace{A_{2,3} \leftarrow 0}_{\text{ゼロクリア}} \to \underbrace{A_{2,3} \leftarrow \delta^*}_{\text{同じ値を再設定}}$$

すなわち毎ステップが no-op となり、$\Delta f = 0$ が続く。

### 2.3 設計上の不整合

| | 旧アルゴリズム |
|---|---|
| 選択基準 | 無向ペアのスコア $\max(\|G_{ij}\|, \|G_{ji}\|)$ |
| 更新演算子 | ペアをリセット後、両方向を比較して勝者を追加 |

「スコアが高い」は「その有向座標に改善余地がある」を意味しない。$G_{ji}$ が大きくても、方向 $j \to i$ が DAG 上実行不可能であれば更新はゼロになる。

---

## 3. 修正方針：有向座標降下法（Directed Coordinate Descent）

### 3.1 アイデア

ペア単位での reset をやめ、**1つの有向座標 $A_{ij}$ を直接更新**する。

$$
(i^*, j^*) = \operatorname{argmax}_{i \neq j} |G_{ij}|
$$

この座標に対して 1 次元最適化を行う：

$$
A_{i^*j^*} \;\leftarrow\; A_{i^*j^*} + \delta^*, \qquad
\delta^* = \operatorname{argmin}_\delta\, f(A + \delta\, e_{i^*} e_{j^*}^\top)
$$

更新後は $G_{i^*j^*} = 0$（1 次元最適値での偏微分がゼロ）になるため、同一座標が再選択されることはない。

### 3.2 DAG 制約の分類

| $A_{ij}$ の現在値 | 操作内容 | DAG チェック |
|---|---|---|
| $A_{ij} \neq 0$（辺 $j \to i$ が既存）| 値のみ更新、グラフ構造は不変 | 不要 |
| $A_{ij} = 0$（新規辺を追加）| $\delta^* > 0$ の場合のみ辺を追加 | 必要（Kahn's algorithm）|

2-cycle の防止は DAG チェックが自動的に担う。$A_{ji} \neq 0$（辺 $i \to j$ が既存）の場合、$A_{ij} \neq 0$（辺 $j \to i$）を追加しようとすると `_can_add_edge` が `False` を返す。

### 3.3 $\lambda_{\ell_0}$ ペナルティの扱い

| シナリオ | ペナルティ |
|---|---|
| 新規辺の追加（$0 \to \text{nonzero}$）| $\Delta = -\Delta f - \lambda_{\ell_0}$ |
| 既存辺の値更新（$\text{nonzero} \to \text{nonzero}$）| $\Delta = -\Delta f$（辺数不変） |
| 辺の削除（$\delta^*$ が自然にゼロへ）| `adj` は `_rank1_update` が自動更新 |

---

## 4. 実装

`coordinate_descent/cd_greedy_A.py` に以下の 2 関数を追加した。

### 4.1 `update_directed_coordinate`

```python
def update_directed_coordinate(A, S, i, j, lambda_l0, A_inv, G, f_state, adj):
    """単一有向座標 A[i,j] を 1 次元最適化する。"""
    edge_exists = (j in adj[i])
    if not edge_exists and not _can_add_edge(adj, i, j):
        return False                          # DAG 制約違反

    delta = delta_star(A, S, i, j, A_inv=A_inv)
    alpha = A_inv[j, i]
    sa_ij = (G[i, j] + 2.0 * alpha) / 2.0
    df = (-2.0 * np.log1p(delta * alpha)
          + 2.0 * delta * sa_ij
          + delta ** 2 * S[i, i])
    penalty = lambda_l0 if not edge_exists else 0.0
    if -df - penalty <= 0.0:
        return False                          # 改善なし

    _rank1_update(A, A_inv, G, S, i, j, delta, f_state, adj)
    return True
```

`_rank1_update` が A、$A^{-1}$（Sherman–Morrison）、G、f_state、adj をすべて $O(d^2)$ でインプレース更新する。

### 4.2 `dag_greedy_A_directed`

```python
def dag_greedy_A_directed(S, T=100, ...):
    # 初期化（O(d³)）
    A, A_inv, G, f_state, adj = ...

    # d*(d-1) 個の有向座標を候補とする
    ii_all, jj_all = np.where(~np.eye(d, dtype=bool))

    for _ in range(T):
        scores = np.abs(G[ii_all, jj_all])   # O(d²)

        # argmax → DAG チェック → マスクして次候補へ
        selected = greedy_scan(scores, adj)

        if selected:
            update_directed_coordinate(A, S, *selected, ...)

        history.append(f_state[0])
```

---

## 5. 実験結果

同一データ（$d=4$, $n=5000$, seed=41）で T=300 ステップを実行した結果。

| アルゴリズム | start | end（T=300）| 挙動 |
|---|---|---|---|
| `dag_greedy_A`（旧） | 62.97 | 62.97 | **完全に停滞**（no-op ループ）|
| `dag_greedy_A_directed`（新） | 137.90 → 62.97 | 58.72 | **毎ステップ単調減少** |
| `dag_coordinate_descent_l0` | 119.83 | 9.14 | 単調収束（対角更新あり）|

```
directed 最初の 5 ステップ:
  step  0: f=62.965  (step 0 では 137.9→62.97 へ -74.93 改善)
  step  1: f=60.086  df=-2.879
  step  2: f=59.656  df=-0.430
  step  3: f=59.031  df=-0.625
  step  4: f=58.899  df=-0.133
```

### 注意事項

- `dag_greedy_A_directed` は**対角成分を更新しない**（T-step 旧版と同様）。
  対角成分まで収束させるには epoch-based の `dag_greedy_A_epoch` を使うか、
  対角座標を候補 $d^2$ 個に含める拡張が必要。
- 辺の「方向転換」（$i \to j$ から $j \to i$ へのフリップ）は 2 ステップに分かれる。
  epoch-based 版では 1 epoch 内で暗黙に処理される。

---

## 6. 旧アルゴリズム（`dag_greedy_A`）の暫定修正について

`dag_greedy_A` に対しては「ペア選択後に zeroing してから実際の $\Delta$ を計算し、改善なければスキップ」という暫定パッチも施したが、これは pair-based の根本的な設計問題を回避するものに過ぎない（$O(d^3)$ の追加逆行列計算が毎ステップ発生する）。

**推奨**：T-step greedy には `dag_greedy_A_directed` を使用すること。

---

## 7. APIリファレンス

```python
from coordinate_descent.cd_greedy_A import dag_greedy_A_directed

A, G_bin, score = dag_greedy_A_directed(S, T=1000, lambda_l0=0.0, threshold=0.05)
A, G_bin, score, history = dag_greedy_A_directed(S, T=1000, return_history=True)
```

戻り値は `dag_greedy_A` と同一シグネチャ。既存コードとの互換性を維持している。
