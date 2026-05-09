# 単辺忠実性仮定の `cd_A_noepoch` への適用設計

**対象ファイル**: `coordinate_descent/coordinate0.py`
**作成日**: 2026-04-18
**対応アルゴリズム**: `dag_coordinate_descent_l0`（実験ノートブックでは `cd_A_noepoch` と記載）

---

## 1. 目的と動機

`cd_A_noepoch` は現在 `{0, …, d-1}²` 上で座標 (i, j) を一様にサンプリングし、
`update_off_diagonal` / `update_diagonal` を呼び出して一回の座標降下更新を実行する
（[coordinate0.py:309-315](../coordinate_descent/coordinate0.py#L309-L315)）。

この設計のコスト：
- サンプリングされた座標対 (i, j) の多くは、**周辺相関がほぼゼロの変数**に対応している。これらの辺は真のモデルには通常存在しないが、アルゴリズムは「追加すべきでない」ことを確認するために $O(d^2)$ の `update_off_diagonal` 呼び出し（Sherman-Morrison 更新 2 回、DAG 判定 2 回、$\delta^*$ 求解 2 回を含む）を費やす。
- 高次元問題（$d \geq 30$）では、この種の「無効ステップ」が大多数を占め、各ラウンド $T$ ステップ予算の主な浪費となる。

本文書は **fGES の「単辺忠実性仮定」**（one-edge faithfulness assumption）を枝刈りツールとして導入し、`cd_A_noepoch` において周辺無相関な変数対間のすべての辺追加候補をマスクすることで、ランダムサンプリングを収益に寄与しうる座標に集中させる。

---

## 2. 理論的背景：単辺忠実性仮定

### 2.1 仮定の内容（fGES 論文 [^1] より）

> 変数 $x$ と $y$ の標本周辺相関係数が 0 であれば、前向き探索のいかなるステップでも有向辺 $x \to y$ を**追加しない**。

これは完全忠実性仮定の制限版であり、「単辺（single edge）」の忠実性違反のみを排除し、条件付き独立関係の大域的忠実性は要求しない。

### 2.2 コスト：パス完全相殺

真のモデルが $A \to B \to C \to D$ かつ $A \to D$ を含み、二つのパスの係数が**ちょうど相殺**する場合、$\operatorname{corr}(A, D) = 0$ となる。このとき単辺忠実性を有効にすると、$A \to D$ 辺は候補集に入れない —— これが本仮定の既知のコストである。

**連続パラメータ空間ではパス相殺のパラメータ集合の測度は 0** だが、有限標本推定下では近似的な相殺が無視できない測度のパラメータ集合で生じうる。

### 2.3 `cd_A_noepoch` へのマッピング

| fGES の文脈 | `cd_A_noepoch` の文脈 |
|---|---|
| 前向き探索での「辺の追加」 | `update_off_diagonal` における `A[i,j]` の 0 から非零への更新 |
| 周辺相関係数 | $S$ を正規化した $\lvert\operatorname{corr}(x_i, x_j)\rvert$ |
| マスク条件 | 独立性指標が閾値 $\tau$ 未満 |

### 2.4 二種類の独立性指標：`corr` と `pcorr`

本設計は「$x_i$ と $x_j$ が近似的に独立」を判定する二種類の指標をサポートし、それぞれ異なる独立性概念に対応する：

| 指標 | 統計的意味 | = 0 のときの DAG 上の意味 |
|---|---|---|
| $\operatorname{corr}(x_i, x_j) = 0$ | **周辺独立** | $x_i$ と $x_j$ を結ぶパスが一切存在しない |
| $\operatorname{pcorr}(x_i, x_j \mid \text{rest}) = 0$ | **他の全変数を所与とした条件付き独立** | $(x_i, x_j)$ が moral graph 上に存在しない（親子関係でなく、共通の子もない） |

偏相関係数は精度行列 $\Omega = S^{-1}$ から得られる：

$$
\operatorname{pcorr}(x_i, x_j \mid \text{rest}) = -\frac{\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}
$$

**枝刈り強度の比較**：

- `corr` スクリーニングは「完全に無関連」な対のみをマスク —— 「何らかのパスで到達可能」な対はすべて保持する。
- `pcorr` スクリーニングは**さらに**「中間変数を介してのみ間接的に関連」する対もマスクする。例えば $x_A \to x_B \to x_C$ における $(A, C)$：$\operatorname{corr}(A, C) \neq 0$ だが $\operatorname{pcorr}(A, C \mid B) \approx 0$。
- したがって **`pcorr` は `corr` より枝刈りが積極的**（マスク集合が大きい → 高速化率が高い）。

**真の有向 DAG 辺の漏れリスク**：二種類の指標は理論上**等価** —— いずれも「パス完全相殺」という測度 0 の事象下でのみ $x_i \to x_j$ 自体に対応する統計量を見逃す。`pcorr` は「より積極的」であるがゆえに真の有向辺を見逃すわけではない。

---

## 3. `cd_A_noepoch` への組み込み方式

### 3.1 「方策 B：サンプリング分布の変更」を選択した理由

実装方法として二つの候補がある：

- **方策 A（ステップ全体のスキップ）**：メインループでマスクされた (i, j) がサンプリングされたら `continue` し、元のサンプリング分布を保持する。
- **方策 B（サンプリング分布の変更）**：事前に「許可候補対」リストを構築し、そこから直接サンプリングする。

本設計は**方策 B** を採用する。理由：

1. `update_off_diagonal` の内部ロジックは **A[i,j] と A[j,i] を無条件にまず零にする**
   （[coordinate0.py:216](../coordinate_descent/coordinate0.py#L216)）。「ステップ全体のスキップ」を
   採用した場合、スキップされた対の `A[i,j]` が現在非零（例えば `A_init` の設定値）であれば、
   「辺の新規追加」と「既存辺の再最適化」を区別する追加ロジックが必要となり実装が複雑化する。
2. 方策 B は各ステップで「収益に寄与しうる座標」上で更新を行い、`update_off_diagonal` の
   固定コストの浪費を避ける。**ステップあたりの期待収益**の観点でより優れている。
3. 方策 B は `early_stop` 機構に対して透過的 —— 各ステップが有効な更新であり、タイミングが変わらない。

### 3.2 マスクの構築

`screening` パラメータ（§5.1 参照）に基づいて統計量を選択する：

```python
if faithfulness_tau > 0:
    if screening == "corr":
        std = np.sqrt(np.diag(S))
        stat = S / np.outer(std, std)                          # 標本相関行列
    elif screening == "pcorr":
        Omega = np.linalg.inv(S + 1e-6 * np.eye(d))            # 正則化逆行列（特異防止）
        d_std = np.sqrt(np.diag(Omega))
        stat = -Omega / np.outer(d_std, d_std)                 # 偏相関行列
    else:
        raise ValueError(f"unknown screening {screening!r}")

    forbidden = np.abs(stat) < faithfulness_tau                # 対称
    np.fill_diagonal(forbidden, False)                         # 対角線はマスクしない
    allowed_offdiag = np.argwhere(~forbidden & ~np.eye(d, dtype=bool))
    M = len(allowed_offdiag)
    if M == 0:
        raise ValueError("All off-diagonal pairs masked; tau too large.")
else:
    allowed_offdiag = None
    M = d * (d - 1)
```

説明：
- `forbidden` は対称（$|\operatorname{corr}|$ と $|\operatorname{pcorr}|$ はいずれも対称）であるため、`allowed_offdiag` は自然に (i, j) と (j, i) の両方向を含む。
- 対角線は決してマスクしない —— 自己相関が 1 であっても、対角線の更新は保持する必要がある（残差分散の調整に使用）。
- $S$ が**共分散行列**（または標本 Gram 行列を $n$ で割ったもの）であることを要求し、`np.diag(S)` が分散を与えることを保証する。`pcorr` モードではさらに $S$ が可逆であることを要求する（$10^{-6} I$ を加えて病的条件を防止）。
- 閾値 $\tau$ のスケールは両モードで一致（いずれも $[0, 1]$ 範囲）しており、同じスキャン区間をそのまま使える。

---

## 4. 対角線 vs off-diag サンプリング確率

### 4.1 ベースライン：元アルゴリズムの比率

元アルゴリズムは各ステップで (i, j) を一様にサンプリングする：

$$
\mathbb{P}(\text{diag}) = \frac{d}{d^2} = \frac{1}{d}, \quad
\mathbb{P}(\text{off-diag の任意の対}) = \frac{1}{d^2}
$$

$d = 20$ のとき diag は 5%、off-diag は 95% を占める。

### 4.2 二つのモード

#### モード `"preserve"`（デフォルト）—— 元の比率を維持

diag サンプリング確率は $1/d$ のまま、off-diag は許可集合から一様にサンプリング：

```python
if np.random.rand() < 1.0 / d:
    i = j = np.random.randint(d)
else:
    idx = np.random.randint(M)
    i, j = allowed_offdiag[idx]
```

**効果**：
- diag 更新頻度は元アルゴリズムと**完全に同一**。
- 各許可 off-diag 対がサンプリングされる確率は $1/d^2$ から $(d-1)/(dM)$ に上昇する。
- **高速化係数** $= d(d-1)/M$；例えば $d=20, M=40$ のとき約 9.5×。

#### モード `"pool"` —— プール型サンプリング

$d$ 個の対角位置と $M$ 個の許可 off-diag 位置を統合し、直接一様にサンプリングする：

```python
total = d + M
r = np.random.randint(total)
if r < d:
    i = j = r
else:
    i, j = allowed_offdiag[r - d]
```

**効果**：
- $\mathbb{P}(\text{diag}) = d/(d+M)$、$\mathbb{P}(\text{off-diag の任意の対}) = 1/(d+M)$。
- 実装が最も簡潔だが、$M$ が小さいとき diag が過剰にサンプリングされる（$d=20, M=40$ のとき diag が 33% を占める）。

### 4.3 選択基準

| シナリオ | 推奨モード | 理由 |
|---|---|---|
| $\tau$ が小さい（$M > d(d-1)/4$） | 両者の差は小さい | diag 比率が大幅に拡大されない |
| $\tau$ が大きい（$M$ が $d$ のオーダーに近い） | `"preserve"` | diag がステップ数を支配し、構造復元が遅延するのを防ぐ |
| 元アルゴリズムと比較可能な収束曲線が必要 | `"preserve"` | history の形状が元版に最も近い |
| 「空回り」ステップの最小化、タイミングは不問 | `"pool"` | ステップあたりの期待寄与が最大 |

### 4.4 将来の拡張：`"scheduled"` モード

収束が「構造復元 + スケール微調整」の二段階を示す場合、動的な重み調整（例：前半は diag 確率 $1/d$、後半は $3/d$ に拡大）を検討できる。**本初版の実装範囲には含まれない**。`"preserve"` の実験結果に基づいて導入を判断する。

---

## 5. API 設計

### 5.1 新規パラメータ（`dag_coordinate_descent_l0` 内）

```python
def dag_coordinate_descent_l0(
    S, T=100, seed=0, threshold=0.05, lambda_l0=0.2,
    return_history=False, return_graph_history=False, A_init=None,
    early_stop=False, check_every=None, tol=1e-4, patience=10, min_steps=None,
    # --- 新規追加 ---
    faithfulness_tau: float = 0.0,          # 0 => 無効化、元アルゴリズムに退化
    sampling_mode: str = "preserve",        # "preserve" | "pool"
    screening: str = "corr",                # "corr" | "pcorr"
):
```

**`screening` パラメータの意味**：

| 値 | 指標 | 推奨使用シナリオ |
|---|---|---|
| `"corr"`（デフォルト） | 標本相関行列 $\lvert\operatorname{corr}\rvert$ | 初版・迅速検証・$S$ が特異または近特異 |
| `"pcorr"` | 偏相関行列 $\lvert\operatorname{pcorr}\rvert$（$S^{-1}$ から算出） | より積極的な枝刈り・高次元疎 DAG・$n \gg d$ |

### 5.2 後方互換性

- デフォルト `faithfulness_tau=0.0` → 元アルゴリズムに完全退化し、`allowed_offdiag = None`、サンプリングロジックは変更なし。
- すべての既存呼び出し（ノートブックおよびベンチマークを含む）は**修正不要**で正常に動作する。

### 5.3 戻り値

既存の戻り値形式を維持する（`return_history` / `return_graph_history` の組み合わせに従う）。**`meta` 辞書は当面導入しない** —— 診断情報（`M`、`τ`、`sampling_mode`）が必要な場合は、後続の独立 issue で対応する。

---

## 6. 正確性と収束性の議論

### 6.1 各ステップの有効性

モード `"preserve"` と `"pool"` はいずれも**各ステップで一回の実質的な更新を実行する**（`continue` でスキップしない）。したがって：
- `history` の長さは厳密に `t + 1` であり、元アルゴリズムと一致する。
- `early_stop` の `check_every` タイミングの修正は不要。

### 6.2 座標の到達可能性

$M > 0$ である限り、任意の許可 off-diag 対および任意の diag 位置が正のサンプリング確率を持つため、座標降下法の大域的収束保証は維持される（非退化ケースで局所最適に収束）。

### 6.3 辺の漏れリスク

マスクされた (i, j) 対は実行全体を通じて**一度もサンプリングされない**。したがって：
- `screening="corr"`：真のモデルに $x_i \to x_j$ が存在するが $\operatorname{corr}(x_i, x_j) \approx 0$（パス相殺）の場合、その辺は漏れる。
- `screening="pcorr"`：真のモデルに $x_i \to x_j$ が存在するが $\operatorname{pcorr}(x_i, x_j \mid \text{rest}) \approx 0$ の場合、その辺は漏れる。これも「パス相殺」型の退化状況に対応する —— 線形ガウス SEM では $\Omega_{ij} = 0$ は $(i, j)$ が moral graph にないことと同値であり、真の有向辺では数値的相殺時にのみ成立する。
- **二種類の指標の真の有向辺に対する漏れメカニズムは理論上等価**であり、いずれも測度 0 の事象に属する。ただし有限標本下では `pcorr` は**小重みの真の辺**（数値境界に近い）に対するマスク確率が `corr` よりわずかに高い。$\Omega$ の推定ノイズが通常 $S$ より大きいためである。
- リスクの大きさは $\tau$ に対して単調増加する。

### 6.4 `update_off_diagonal` のゼロクリア意味論への影響

元の `update_off_diagonal` は A[i,j] と A[j,i] をまずゼロにクリアしてから、再追加するかを判定する。本設計は方策 B（許可集合からサンプリング）を採用するため、マスクされた対は `update_off_diagonal` に**そもそも入らない**。したがって「マスクした辺が強制的にゼロクリアされる」矛盾は存在しない。

$A_\text{init} \neq I$ の場合、$A_\text{init}$ に辺 $x_i \to x_j$ が存在するが $(i,j)$ がマスクされているとき、その辺は最終解に保持される（サンプリングされないため触れられない）が、アルゴリズムがそれを再最適化する機会はない。`A_init` と `faithfulness_tau` を同時に指定する際はこの点に注意が必要である。

---

## 7. 期待される収益とコスト

### 7.1 収益

- **高次元での高速化**：ER 疎グラフ + 高次元（$d \geq 30$）シナリオ下：
  - `screening="corr"`：$|\operatorname{corr}|$ 行列のスパース度は中程度、$M / d(d-1)$ は通常 20-30%。ステップあたり有効率が 3-5× 向上。
  - `screening="pcorr"`：$|\operatorname{pcorr}|$ は moral graph のスパース度に対応し、$M / d(d-1)$ は 5-10% まで低下しうる（moral graph 自体が疎 DAG の緊密な上位集合）。ステップあたり有効率が 10× 以上向上。
- **低次元での無損失**：$\tau = 0$（デフォルト）のとき、動作は元アルゴリズムと完全に同一であり、オーバーヘッドはゼロ。

### 7.2 コスト

- **パス相殺による辺の漏れ**：§2.2 で述べた通り。$\tau$ が大きいほどコストが増大。
- **$\tau$ のチューニング**：問題規模とサンプルサイズに応じたスキャンが必要。推奨初期範囲：$\tau \in \{0, 0.02, 0.05, 0.1\}$。
- **一回限りの前処理コスト**：
  - `corr`：$O(d^2)$ で相関行列を構築。
  - `pcorr`：$O(d^3)$ で $S^{-1}$ を計算。d=100 で約 10ms。
  - $T \cdot O(d^2)$ のメインループに比べていずれも無視できる。
- **pcorr の数値要件**：`pcorr` は $S$ が非特異であることを要求する。$n < d$ または多重共線性が深刻な場合はエラーが発生しうる。逆行列計算前に $10^{-6} I$ を加えて正則化しているが、極端なケースでは `corr` モードに戻すべきである。

---

## 8. 実験計画

### 8.1 機能検証

1. **パス相殺の小例**（$d = 4$）：$A \to B \to C \to D$ + $A \to D$ で二パスがちょうど相殺するデータを構築し、$\tau > 0$ のとき $A \to D$ が確かに漏れ、$\tau = 0$ のときは復元されることを検証。
2. **τ = 0 回帰テスト**：既存 ER ベンチマークデータで $\tau = 0$ の結果が元アルゴリズムと**バイト単位で一致**する（同一 seed で `A`、`history` が完全同一）ことを検証。

### 8.2 性能スキャン

[test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb](../experiments/notebooks/test/test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb) の実験設定を再利用する：

- $d \in \{20, 30, 50\}$
- $n = 20000$
- $\tau \in \{0, 0.02, 0.05, 0.1\}$
- `screening` $\in$ \{`"corr"`, `"pcorr"`\}
- `sampling_mode` $\in$ \{`"preserve"`, `"pool"`\}
- 各組 $\geq 5$ trials

指標：
- SHD / CPDAG-SHD / MEC 一致率
- `early_stop` に到達するまでに要した $T$
- 壁時計時間
- $M / d(d-1)$（保持された off-diag 対の比率、`corr` と `pcorr` それぞれの値を記録）
- 漏辺率：既知の ground truth の ER グラフ上で、`forbidden` でマスクされたが実際に有向辺が存在する対の数

### 8.3 greedy_cd_A との横断比較

[test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb](../experiments/notebooks/test/test_er_cd_A_vs_greedy_cd_A_noepoch.ipynb) で `cd_A_noepoch` が SHD で `greedy_cd_A` を大幅に上回ること（d=20 で 8.8 vs 29.4）が既に示されている。$\tau$ 導入後は、`cd_A_noepoch + weakfaith` が runtime で greedy に追いつきつつ SHD の優位性を維持できるかをさらに確認すべきである。

---

## 9. 実施順序

1. **まず文書を作成**（本ファイル）。
2. `dag_coordinate_descent_l0` に `faithfulness_tau` / `sampling_mode` パラメータを**実装**する。既存の関数シグネチャの後方互換性を保持。
3. **回帰テスト**：$\tau = 0$ での元アルゴリズムとの一致性確認。
4. **新規実験ノートブック**を作成：`test_er_cd_A_weakfaith.ipynb`、§8 の設計に従ってスキャン。
5. **実験結果に基づき**以下を判断：
   - `"scheduled"` モードの導入要否。
   - `meta` 戻り辞書の必要性。
   - `min_steps` のデフォルト値に $M$ 関連のスケーリングを加えるか。

---

## 10. 実験結果（ER ベンチマーク）

> 実験ノートブック：[`experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb`](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)
> 結果 CSV：`experiments/results/er_cd_A_weakfaith_benchmark_summary.csv`

### 10.1 実験設定

| パラメータ | 値 |
|---|---|
| グラフタイプ | ER、次数 2.0 |
| ノイズ | `gaussian_nv`、$B_{\text{scale}} = 1.0$ |
| $d$ | 20, 30, 50 |
| $n$ | 20 000 |
| 各組の試行数 | 10 |
| $\lambda_{L0}$ | 0.2 |
| $T$（ステップ予算） | 100 000 |
| 早期停止 | 有効（tol=1e-4, patience=10） |
| 閾値 | 0.05 |

テスト対象の 6 つのアルゴリズムバリアント：

1. **baseline**：元の `coordinate0.dag_coordinate_descent_l0`
2. **wf_corr_tau0.02_preserve**：corr スクリーニング, $\tau=0.02$, preserve サンプリング
3. **wf_corr_tau0.05_preserve**：corr スクリーニング, $\tau=0.05$, preserve サンプリング
4. **wf_pcorr_tau0.02_preserve**：pcorr スクリーニング, $\tau=0.02$, preserve サンプリング
5. **wf_pcorr_tau0.05_preserve**：pcorr スクリーニング, $\tau=0.05$, preserve サンプリング
6. **wf_corr_tau0.05_pool**：corr スクリーニング, $\tau=0.05$, pool サンプリング

### 10.2 主要指標まとめ

| アルゴリズム | $d$ | SHD ↓ | CPDAG-SHD ↓ | MEC 一致率 ↑ | 実行時間 (s) | 実際ステップ数 (mean) | mask keep ratio |
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

### 10.3 主な発見

#### (1) pcorr スクリーニングが構造復元精度を大幅に向上

偏相関（`pcorr`）スクリーニングは、すべての $d$ で baseline および corr 系列を大幅に上回った：
- $d = 50$ で SHD が 32.9 (baseline) から **7.5** (`pcorr_tau0.05_preserve`) に低下。低下率 **77%**。
- CPDAG-SHD が 64.9 から **12.8** に低下。低下率 **80%**。
- MEC 一致率は $d = 20$ で **60%** に到達（baseline はわずか 10%）。

これは §2.4 の理論分析を裏付ける：pcorr スクリーニングは精度行列のスパース性を活用し、「仲介変数を介してのみ間接相関する」変数対をマスクすることで、moral graph 内の真の候補辺に探索を集中させる。

#### (2) corr スクリーニングの改善は限定的だが正の方向

corr 系列は baseline に対して一定の改善がある（$d = 50$ で SHD が 32.9 → 20.0）が、pcorr には遠く及ばない。corr は「周辺関連が完全にない」変数対しかマスクできず、大量の間接関連対が残るためである。

#### (3) pool サンプリング：最速だが精度はやや劣る

`wf_corr_tau0.05_pool` の実行時間が最短（$d = 50$ で **16.1s** vs baseline **50.3s**、**3.1×** 高速化）だが、SHD は baseline と同程度（28.1 vs 32.9）であり、preserve モード（20.0）より劣る。pool モードでは diag サンプリング比率が高すぎ（探索空間が小さいとき diag 占有率が 1/3 に達する）、構造復元が不十分となる。

§4.3 の予測と一致する：$M$ が $d$ のオーダーに近いときは `preserve` モードを使用すべきである。

#### (4) 実行時間：preserve モードでは顕著な高速化なし

preserve モードの実行時間は baseline と近い（$d = 50$：41-44s vs 50s）。preserve では各ステップの `update_off_diagonal` コストが不変であり、探索の集中度のみが向上しているためである。高速化は**より少ないステップ数で収束する**（実際のステップ数が少ない、またはステップ数は同程度だが結果がより良い）という形で現れ、壁時計時間の短縮としては現れない。

#### (5) マスクスパース度が $d$ とともに増大

| スクリーニング方式 | $d=20$ | $d=30$ | $d=50$ |
|---|---|---|---|
| corr ($\tau=0.05$) | 27% | 25% | 14% |
| pcorr ($\tau=0.05$) | 17% | 12% | 6% |

pcorr は $d=50$ でわずか **6%** の候補対のみを保持し、探索空間を約 **16×** 圧縮する。次元の増大とともに枝刈りがより積極的になり、疎 DAG 下での moral graph 辺数 $O(d)$ vs 全対数 $O(d^2)$ の漸近的予測と一致する。

### 10.4 結論と推奨

1. **推奨デフォルト構成**：`screening="pcorr"`, `faithfulness_tau=0.05`, `sampling_mode="preserve"`。この構成はテストしたすべての次元で最良またはほぼ最良の SHD / CPDAG-SHD を達成し、実行時間の顕著な増加もない。
2. **速度を優先する場合**：`screening="corr"`, `sampling_mode="pool"` で 3× の高速化が得られるが、精度の改善はない。
3. **$\tau$ の選択**：0.05 は 0.02 よりわずかに優れる（テスト範囲内ではより積極的な枝刈りが辺の漏れ増加を招いていない）が、より大きな $\tau$（例：0.1）は未テストであり注意が必要。
4. **pcorr の適用条件**：$n \gg d$ のとき最も良好に機能する（本実験では $n/d = 400$）。$n$ が $d$ に近い場合、精度行列の推定ノイズが増大するため、corr に戻すか $\tau$ を下げる必要があるかもしれない。

---

## 11. 参考文献

[^1]: Ramsey et al., "A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images." International Journal of Data Science and Analytics, 2017.
[^2]: 実験ノートブック [`test_er_cd_A_weakfaith_benchmark.ipynb`](../experiments/notebooks/test/test_er_cd_A_weakfaith_benchmark.ipynb)、2026-04-18 実行。
