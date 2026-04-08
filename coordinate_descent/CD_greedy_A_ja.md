# Greedy Coordinate Descent for A-formulation

## 1. 背景設定

### 1.1 A-formulation

$d$ 次元の確率変数ベクトル $X \in \mathbb{R}^d$ を

$$
X = A^{-1} N, \qquad N \sim \mathcal{N}(0, I)
$$

で表す。ここで $A$ は上手く並べ替えると三角化できる行列であり，非対角成分の有向構造が DAG を与える。

サンプル共分散

$$
S = \frac{1}{n} X^\top X
$$

に対して，本実装が最小化する滑らかな目的関数は

$$
f(A) = -2 \log \det A + \operatorname{tr}(A^\top S A)
$$

である。さらに非対角成分の疎性を促すために，必要に応じて

$$
F_\lambda(A) = f(A) + \lambda_{\ell_0} \|A\|_{\mathrm{off},0}
$$

を考える。

### 1.2 DAG 制約

非対角成分 $A_{ij} \neq 0$ は有向辺 $i \to j$ を表す。したがって更新後の非対角部分が閉路を含まないことが必要である。本実装では隣接リスト `adj` を持ち，辺の追加時だけ Kahn のトポロジカルソートで DAG 性を判定する。

---

## 2. 実装の全体像

`cd_greedy_A.py` には次の 3 系統が実装されている。

| 関数 | 役割 | 特徴 |
|---|---|---|
| `dag_greedy_A` | T-step の主実装 | 非対角と対角を同じ候補集合で greedy に選ぶ |
| `dag_greedy_A_directed` | T-step の off-diagonal のみ版 | 対角更新を含まない |
| `dag_greedy_A_epoch` | epoch ベース版 | 旧来の pair-based 構造ブロック + 対角ブロック |

現行の中心実装は `dag_greedy_A` と `dag_greedy_A_directed` であり，どちらも共有関数 `_dag_greedy_A_impl(...)` を用いている。

---

## 3. T-step 版アルゴリズム

### 3.1 初期化

初期行列は

$$
A^{(0)} = I_d
$$

または引数 `A_init` で与えられる。初期化時に次のキャッシュを一括で構成する。

- `A_inv = A^{-1}`
- `G = -2A^{-T} + 2SA`
- `f_state = [f(A)]`
- `adj`: 非対角非零成分から作る隣接リスト

### 3.2 1 ステップの流れ

`dag_greedy_A` の各ステップは次の通りである。

1. 現在の勾配

$$
G = \nabla_A f(A) = -2A^{-T} + 2SA
$$

を用いて候補スコアを作る。

2. `include_diagonal=True` のとき

- 対角候補: $|G_{ii}|$
- 非対角候補: $|G_{ij}|,\ i \neq j$

を 1 本のベクトル `scores` にまとめる。

3. `scores` の最大要素を 1 つ取り出し，その候補が実際に改善を与えるかを判定する。

4. 改善しない候補は `scores[idx] = -1` として淘汰し，次に大きい候補を再び調べる。

5. 最初に受理された候補だけを実際に更新する。

6. どの候補も受理されなければそこで打ち切る。

このため，1 ステップの理論計算量は「候補を何個棄却したか」に依存する。

### 3.3 `dag_greedy_A` と `dag_greedy_A_directed` の違い

- `dag_greedy_A`
  - 対角候補と非対角候補を同一ループ内で扱う
  - 対角更新も greedy に選ばれる
- `dag_greedy_A_directed`
  - 非対角候補のみを対象とする
  - 対角更新は行わない

---

## 4. 非対角更新

### 4.1 1 次元最適化

座標 $(i,j)$ に対して

$$
A \leftarrow A + \delta E_{ij}
$$

を考えると，$f(A)$ を最小化する $\delta^\*$ は `coordinate0.delta_star(...)` で計算される。

必要な量は

$$
c = S_{ii}, \qquad b = (SA)_{ij}, \qquad \alpha = (A^{-1})_{ji}
$$

であり，

$$
D = (c + \alpha b)^2 - 4 \alpha c (b - \alpha)
$$

$$
\delta^\* =
\frac{2(b-\alpha)}{-(c+\alpha b)-\sqrt{D}}
$$

を用いる。`A_inv` がキャッシュされていれば $\alpha$ は $O(1)$ で取り出せる。

### 4.2 目的関数差分

更新 $A[i,j] \mathrel{+}= \delta$ に対する $f$ の差分は

$$
\Delta f
=
-2 \log(1+\delta \alpha)
+ 2 \delta (SA)_{ij}
+ \delta^2 S_{ii}
$$

である。実装では

$$
(SA)_{ij} = \frac{G_{ij} + 2(A^{-1})_{ji}}{2}
$$

を使って，`G` と `A_inv` から $O(1)$ で取り出している。

### 4.3 受理条件

非対角候補 $(i,j)$ に対する受理判定は `_directed_coordinate_gain(...)` で行う。

- すでに辺 $i \to j$ が存在する場合

$$
\text{gain}_{ij} = -\Delta f
$$

- まだ辺が存在しない場合

$$
\text{gain}_{ij} = -\Delta f - \lambda_{\ell_0}
$$

`gain > 0` のときのみ更新を受理する。

### 4.4 DAG 判定

辺 $i \to j$ がまだ存在しないときだけ，`_can_add_edge(adj, i, j)` を呼ぶ。

手順は単純で，

1. `adj[i]` に一時的に `j` を追加
2. `_is_dag_kahn(adj)` で DAG 性を判定
3. 元に戻す

という流れである。計算量は $O(d+E)$，ここで $E$ は現在の辺数である。

---

## 5. 対角更新

対角成分 $A_{ii}$ は DAG 制約と無関係なので，別の 1 次元最適化を使う。

`_diagonal_coordinate_gain(...)` は `coordinate0.update_diagonal(...)` と同じ考え方を採用している。

1. まず

$$
A_{ii} \leftarrow 0.3
$$

へ一旦リセットする。

2. そのリセット後の状態から，再び 1 次元最適化で最良の追加量 `final_delta` を計算する。

3. リセット分と追加分の合計改善量

$$
\text{gain}_{ii} = -(\Delta f_{\text{reset}} + \Delta f_{\text{final}})
$$

が正なら，2 回の rank-1 update を適用する。

対角更新には $\ell_0$ ペナルティは課していない。

---

## 6. rank-1 更新とキャッシュ

本実装の要点は，更新ごとに `A_inv`・`G`・`f_state`・`adj` を同期的に保つ点にある。

### 6.1 Sherman-Morrison

$$
A_{\text{new}} = A + \delta e_i e_j^\top
$$

に対して，

$$
A_{\text{new}}^{-1}
=
A^{-1}
-
\frac{\delta}{1+\delta (A^{-1})_{ji}}
\,
A^{-1}[:,i]\,
A^{-1}[j,:]
$$

を用いる。これは `_rank1_update(...)` に実装されている。

### 6.2 勾配キャッシュの更新

`G = -2A^{-T} + 2SA` なので，

- 逆行列項は outer product で $O(d^2)$
- $2SA$ 項は第 $j$ 列だけを $O(d)$ で更新

できる。したがって 1 回の受理更新は基本的に $O(d^2)$ で済む。

### 6.3 目的関数のキャッシュ

`f_state[0]` は差分公式を使って in-place に更新する。これにより各ステップで `f(A,S)` を $O(d^2)$ で再評価する必要がない。

### 6.4 フォールバック

Sherman-Morrison の分母

$$
1 + \delta (A^{-1})_{ji}
$$

が極端に小さいときは数値的に不安定なので，`_rank1_update(...)` は `False` を返す。その場合 `_recompute_all(...)` により

- `A_inv`
- `G`
- `f_state`

を厳密に再計算する。

---

## 7. Epoch 版

`dag_greedy_A_epoch(...)` は T-step 版とは別系統であり，旧来の pair-based 更新を 1 epoch ごとにまとめて実行する。

1 epoch は次の 2 ブロックからなる。

1. 構造ブロック
   - 上三角の全 pair $(i,j)$ を `max(|G_{ij}|, |G_{ji}|)` で並べ替える
   - 各 pair に対して `update_off_diagonal_greedy(...)` を適用する

2. スケールブロック
   - すべての対角成分に `update_diagonal_greedy(...)` を適用する

`update_off_diagonal_greedy(...)` は directed 版とは異なり，pair $(i,j)$ を選んだら

- まず $A_{ij}, A_{ji}$ をともに 0 に戻し
- その後で $i \to j$ と $j \to i$ のどちらが良いかを比べ
- 良い方向だけを入れ直す

という pair-based の処理になっている。

---

## 8. 計算量の見方

### 8.1 T-step 版

候補数を

$$
m =
\begin{cases}
d + d(d-1) & \text{対角を含む場合} \\
d(d-1) & \text{非対角のみの場合}
\end{cases}
$$

とする。また 1 ステップ内で実際に調べた候補数を $k$ とする。

| 処理 | 計算量 | 備考 |
|---|---|---|
| スコアベクトル構築 | $O(d^2)$ | `abs(G)` の抽出 |
| 候補の探索 | $O(k d^2)$ | 現行実装は `argmax` を繰り返す |
| DAG 判定 | $O(k(d+E))$ | 新規辺候補に対してのみ |
| 受理更新 | $O(d^2)$ | rank-1 update |

したがって 1 ステップの支配項は概ね

$$
O(k d^2)
$$

である。$k$ が小さければ 1 ステップは $O(d^2)$ に近く，$k$ が大きいと候補探索のコストが支配的になる。

### 8.2 Epoch 版

epoch 版は 1 epoch で全 pair を走査するため，

$$
O(d^4)
$$

スケールになる。こちらは T-step 版よりも重いが，1 epoch の中で網羅的に座標を更新する。

---

## 9. 実装上の注意

### 9.1 `dag_greedy_A` の意味

`dag_greedy_A` は「現在の勾配絶対値が大きい候補から順に，実際に受理可能なものを探す」アルゴリズムである。したがって純粋な `argmax |G_{ij}|` だけで更新するわけではなく，途中で複数候補が淘汰されることがある。

### 9.2 `lambda_l0`

`lambda_l0` は非対角成分の新規追加時にだけ効く。すでに存在する辺の再推定や，対角成分の更新にはペナルティを課していない。

### 9.3 `threshold`

返り値の `G_binary = weight_to_adjacency(A, threshold)` は，最終的な重み行列 `A` を閾値処理して二値グラフに変換したものである。学習中の `adj` では `1e-8` 程度の閾値を使い，出力時の可視化・評価には `threshold` を用いる。

---

## 10. 関数対応表

```text
cd_greedy_A.py
├── _build_adjacency(A, threshold)          非零非対角から隣接リストを構成
├── _is_dag_kahn(adj)                       Kahn 法による DAG 判定
├── _can_add_edge(adj, i, j)                辺 i->j を追加可能か判定
├── _rank1_update(...)                      A, A_inv, G, f_state, adj の同時更新
├── _recompute_all(...)                     失敗時の厳密再計算
├── _incremental_df(...)                    1 座標更新の Δf
├── _directed_coordinate_gain(...)          非対角候補の gain 計算
├── _diagonal_coordinate_gain(...)          対角候補の gain 計算
├── update_directed_coordinate(...)         非対角 1 座標更新
├── update_diagonal_greedy(...)             対角 1 座標更新
├── _dag_greedy_A_impl(...)                 T-step 版の共有本体
├── dag_greedy_A(...)                       対角込み T-step greedy
├── dag_greedy_A_directed(...)              非対角のみ T-step greedy
└── dag_greedy_A_epoch(...)                 epoch ベース greedy
```

---

## 11. まとめ

現行の `greedy_cd_A` 実装は，

- `A_inv`・`G`・`f_state` をキャッシュし
- 受理された更新は rank-1 で $O(d^2)$ に反映し
- DAG 制約は隣接リスト + Kahn 法で処理し
- T-step 版では勾配絶対値の大きい候補から順に受理可能性を調べる

という構成になっている。

したがって本実装の性質は，

- 受理更新自体は軽い
- どれだけ候補を棄却するかが T-step 版の実行時間を左右する
- epoch 版はより網羅的だが計算量は重い

という 3 点に要約できる。
