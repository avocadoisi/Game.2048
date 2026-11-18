# issue の分類（area / kind）

issue は **`area:`（系統 = 何の話か）と `kind:`（種別 = 何をしたか）の 2 軸**で分類する。
1 issue に**両方を 1 つずつ**付ける。

134 件まで溜まった時点で、分類はタイトル先頭の接頭辞（`探索:` `学習:` `NN学習:` …）だけに頼っていた。
接頭辞は 39 通りに散らばり、しかも「系統」と「種別」が混ざっている（`計測:` は種別、`学習:` は系統）。
ラベルに分けることで、「学習側で実装だけしたもの」「探索側で測り直したもの」を引ける。

## area: 系統

| ラベル | 何を指すか | 代表 |
| --- | --- | --- |
| `area:solver` | 探索と評価関数。`ai/solver/` の expectimax・方策・n-tuple・枝刈り・探索の高速化 | #267 #171 #266 |
| `area:nn` | NN の価値関数そのもの。重み・学習則・蒸留・推論・容量（`ai/solver/nn.zig`） | #233 #262 #277 |
| `area:eval` | 計測基盤と判定手順。`pairw` / `ranking` / `bench` / `evalab` の土台、条件の公平化、分散対策 | #210 #168 #235 |
| `area:core` | ゲームロジック `2048.core`（move / merge / `row_table`） | #160 #71 |
| `area:web` | wasm・ブラウザ UI・配信・HTTP サーバー（`src/wasm.zig` / `src/web/` / Pages） | #154 #145 #149 |
| `area:docs` | `docs/` / `README` / skill | #124 #53 |

## kind: 種別

| ラベル | 何を指すか | 例 |
| --- | --- | --- |
| `kind:explore` | 未知の効果を測って採否を出す。条件を変えた新しい仮説 | #233（h1=128 で +57,424 → 採用） |
| `kind:implement` | コードやノブを足す・既定を変える・重複を潰す（挙動不変のリファクタも含む） | #277（shaping を opt-in で追加） |
| `kind:measure` | 既にある構成を測り直す・数値を確定する（「再測定」「評価し直す」） | #257（phase_h2/h3 の再測定） |
| `kind:fix` | 壊れているものを直す（ビルド・テスト・計測の誤り・デプロイ） | #138 #149 |
| `kind:docs` | ドキュメントを書く・直す | #124 |

## 迷ったときの規則

1. **コードが増えたら `implement`、数値と採否だけなら `explore`。** 同じ issue で両方やったときは
   主たる成果物で選ぶ。ノブがリポジトリに残った → `implement`、ノブごと捨てた → `explore`。
2. **「測り直す」「再測定」は `measure`。** 条件を変えた新しい仮説は `explore`。
3. 学習の効果測定は `area:nn`、探索のノブは `area:solver`、**計測の仕組みそのもの**（pairw の実装、
   ranking の列、判定基準）は `area:eval`。
4. `src/core.zig` を触るなら `area:core`。wasm サイズや同梱形式の都合で触るなら `area:web`。
5. n-tuple は「学習器」ではなく「探索側の評価関数」として `area:solver`（NN の重み学習は `area:nn`）。

## 検索

```bash
gh issue list --state all -l area:nn -l kind:implement    # 学習側で実装したもの（複数 -l は AND）
gh issue list --state all -l area:eval --search "pairw"   # 計測基盤のうち pairw の話
gh issue list --state open -l area:solver                 # 開いている探索系
gh issue list --state all -l kind:measure                 # 測り直しの履歴
```

## 新しい issue

```bash
gh issue create -t "学習: ..." -b "..." -l area:nn,kind:explore
```

タイトルの接頭辞（`学習:` など）は履歴として残している。新規では付けても付けなくてよい
（系統はラベルが持つ）。issue 本文の `## Why` / `## Do` はそのまま。

## ラベルの定義と一括付与

ラベルの色・説明は `scripts/label-issues.py` の `LABELS`、issue 番号 → ラベルの対応表は
`scripts/issue-labels.tsv`（#282 で過去分を埋め戻した記録。**新規 issue を追記する必要はない**）。

```bash
python3 scripts/label-issues.py            # dry-run。何が変わるかだけ表示
python3 scripts/label-issues.py --apply    # ラベル作成 + 付与。冪等（2 回流しても差分なし）
```

`--apply` は不足しているラベルを作ってから付与する。どのラベルも付いていない issue が残ると
終了コード 1 で知らせる。

## 運用

- GitHub の既定ラベル（`bug` `documentation` `enhancement` `question` など）は**使わない**。
  `kind:fix` / `kind:docs` が対応する。過去に付いていた 6 件（#111〜#117）には残っているが、新規には付けない。
- 分類を変えたくなったら、`gh issue edit <n> --remove-label ... --add-label ...` して
  `scripts/issue-labels.tsv` も直す（次回の `--apply` で戻らないように）。

## 進行タグ `train:more`

`area:` / `kind:` とは別に、**追加学習を回す価値がある**ことを示す進行タグ。
4 時間の学習予算では ADOPT に届かなかったが、未完走で明確には負けておらず、上向きのときだけ付ける
（条件は [docs/measurement.md](measurement.md) の「学習の時間予算」）。
付いていない issue には追加学習を回さない。ADOPT したら外して close する。

```bash
gh issue edit <n> --add-label train:more
gh issue list -l train:more --state open
```
