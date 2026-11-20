# issue の分類（area / kind）

issue は **`area:`（系統 = 何の話か）と `kind:`（種別 = 何をしたか）の 2 軸**で分類し、1 issue に**両方を 1 つずつ**付ける。
タイトル接頭辞（`探索:` `学習:` …）だけでは 39 通りに散らばり、系統と種別が混ざる（`計測:` は種別、`学習:` は系統）ため、
ラベルに分けて「学習側で実装だけしたもの」「探索側で測り直したもの」を引けるようにした。

## area: 系統

| ラベル | 何を指すか | 代表 |
| --- | --- | --- |
| `area:solver` | 探索と評価関数。`ai/solver/` の expectimax・方策・n-tuple・枝刈り・高速化 | #267 #171 #266 |
| `area:nn` | NN の価値関数そのもの。重み・学習則・蒸留・推論・容量（`ai/solver/nn.zig`） | #233 #262 #277 |
| `area:eval` | 計測基盤と判定手順。`pairw` / `ranking` / `bench` / `evalab` の土台、条件の公平化、分散対策 | #210 #168 #235 |
| `area:core` | ゲームロジック `2048.core`（move / merge / `row_table`） | #160 #71 |
| `area:web` | wasm・ブラウザ UI・配信・HTTP サーバー（`src/wasm.zig` / `src/web/` / Pages） | #154 #145 #149 |
| `area:docs` | `docs/` / `README` / skill | #124 #53 |

## kind: 種別

| ラベル | 何を指すか | 例 |
| --- | --- | --- |
| `kind:explore` | 未知の効果を測って採否を出す。条件を変えた新しい仮説 | #233（h1=128 で +57,424 → 採用） |
| `kind:implement` | コードやノブを足す・既定を変える・重複を潰す（挙動不変のリファクタを含む） | #277（shaping を opt-in で追加） |
| `kind:measure` | 既にある構成を測り直す・数値を確定する | #257（phase_h2/h3 の再測定） |
| `kind:fix` | 壊れているものを直す（ビルド・テスト・計測の誤り・デプロイ） | #138 #149 |
| `kind:docs` | ドキュメントを書く・直す | #124 |

## 迷ったときの規則

1. **コードが増えたら `implement`、数値と採否だけなら `explore`。** 両方やったときは主たる成果物で選ぶ
   （ノブがリポジトリに残った → `implement`、ノブごと捨てた → `explore`）。
2. **「測り直す」「再測定」は `measure`。** 条件を変えた新しい仮説は `explore`。
3. 学習の効果測定は `area:nn`、探索のノブは `area:solver`、**計測の仕組みそのもの**（pairw の実装、ranking の列、
   判定基準）は `area:eval`。
4. `src/core.zig` を触るなら `area:core`。wasm サイズや同梱形式の都合で触るなら `area:web`。
5. n-tuple は「学習器」ではなく「探索側の評価関数」なので `area:solver`（NN の重み学習は `area:nn`）。

## 検索と作成

```bash
gh issue list --state all -l area:nn -l kind:implement    # 複数 -l は AND
gh issue list --state all -l area:eval --search "pairw"
gh issue list --state open -l area:solver
gh issue create -t "学習: ..." -b "..." -l area:nn,kind:explore
```

タイトルの接頭辞は履歴として残しているだけで、新規では付けても付けなくてよい（系統はラベルが持つ）。
issue 本文の `## Why` / `## Do` はそのまま。GitHub の既定ラベル（`bug` `documentation` など）は使わず、
`kind:fix` / `kind:docs` で表す（過去の #111〜#117 に残っているだけ）。

ラベルの色・説明は `scripts/label-issues.py` の `LABELS`、issue 番号 → ラベルの対応表は
`scripts/issue-labels.tsv`（#282 で過去分を埋め戻した記録。新規 issue を追記する必要はない）。

```bash
python3 scripts/label-issues.py            # dry-run。何が変わるかだけ表示
python3 scripts/label-issues.py --apply    # ラベル作成 + 付与。冪等（2 回流しても差分なし）
```

`--apply` は不足しているラベルを作ってから付与し、どのラベルも付いていない issue が残ると終了コード 1 で知らせる。
分類を変えたくなったら `gh issue edit <n> --remove-label ... --add-label ...` して `scripts/issue-labels.tsv` も直す
（次回の `--apply` で戻らないように）。

## 進行タグ `train:more` / `train:wait`

`area:` / `kind:` とは別の進行状態で、どちらも付かなければ追加学習もラン待ちもしていない。

| タグ | 意味 | 外すとき |
| --- | --- | --- |
| `train:more` | **追加学習を回す価値がある**。4 時間の学習予算で ADOPT に届かなかったが、未完走で明確には負けておらず上向き | ADOPT したら外して close |
| `train:wait` | **ラン待ち**（学習・計測・profile の実行中、または CPU の順番待ち） | 採否と数値を issue に記録したら close と同時に必ず外す |

`train:more` の条件は [docs/measurement.md](measurement.md) の「学習の時間予算」。付いていない issue には追加学習を回さない。
学習も計測も 1 本ずつしか回せないので、`train:wait` はランの依頼を積んだ時点で付け、待ちの理由（実行中 / 順番待ち /
4 時間上限で打ち切り）を issue のコメントに書く。`gh issue close` はラベルを外さないので `gh issue edit` で先に外す。
両方付いてもよい（= いまラン待ちで、終わったら追加学習の判断）。新しい候補の回数上限は下記 close 上限を参照する。

```bash
gh issue edit <n> --add-label train:more
gh issue list -l train:more --state open
```

## 進行タグ `train:wait`

**ラン待ち**（学習・計測・profile の実行中、または CPU の順番待ち）を示す進行タグ。
学習も計測も 1 本ずつしか回せないので、issue は「コードは入ったが、数値を取るランがまだ終わっていない」
間このタグを持つ。待ちの理由（実行中 / 順番待ち / 4 時間上限で打ち切り）は issue のコメントに書く。

- ランの依頼を積んだ時点で付ける。
- ランが終わって採否と数値を issue に記録したら、**close のときに必ず外す**。
  `gh issue close` はラベルを外さないので、`gh issue edit` で先に外す。
- `train:more` は「4 時間では決着せず、追加学習で伸びる見込み」の意味。`train:wait` は
  「いまラン待ち」の意味で、採否が決まったら close と同時に消える。両方付いてもよい。

### 候補1件のclose上限（#336）

新しい学習候補の予算は、[docs/concept.md](concept.md) と [docs/measurement.md](measurement.md) で固定する。
**候補1件につき学習2回・最終評価1セット**を使っても採用条件を満たさない場合、両方の進行タグを外して close する。

- 1局の時間超過は、部分 score ≥100,000 または 8192 tile のときだけ、延長1回を検討できる。
  未定の再挑戦・別seedでの繰り返しを `train:more` という理由で積まない。
- 明らかに発散・失敗・許容できない減速のときは、枠を使い切る前に no-go で閉じてよい。
  一方で完了しない時間超過を負け・進歩として記録せず、`no-go` か `inconclusive` と書く。
- issue の本文・コメントに、実行コマンド、各1局の使用上限、`+xtime` の有無、完走した学習/評価の数、採否を残す。
- close はbotによる自動処理ではなく、上記の記録を書いてから行う。

```bash
gh issue edit <n> --add-label train:wait
gh issue edit <n> --remove-label train:wait && gh issue close <n>
```
