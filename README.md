# Game.2048

ブラウザで遊ぶ: **https://avocadoisi.github.io/Game.2048/** （wasm、solver は `nn3` 既定で自動再生、`nn2` も選べる。速度も選べる）

Zig で書いた 2048。ゲームロジックを `2048.core` として切り出し、CLI / HTTP サーバー / WASM / solver AI が同じコアを使う。

- Zig **0.17**（`std.Io` / `std.process.Init` 系 API）
- `zig build test` で全ユニットテストが通ること
- 乱数は注入式で、seed 固定なら同じゲーム・同じ solver の手順が再現する

## クイックスタート

```bash
nix develop                      # Zig ツールチェイン (flake.nix)
zig build test
zig build run                    # 端末で遊ぶ
zig build run -- serve 8080      # http://127.0.0.1:8080/ でブラウザ UI
zig build run -- help
```

- `play`: `w/a/s/d` または `up/left/down/right` で移動、`n` でリセット、`q` で終了
- `serve`: 矢印キー/WASD で操作。`GET /api/state`、`POST /api/new`、`POST /api/move?dir=left`、
  `GET /api/solve?depth=2&algo=nn2`、`GET /api/ranking`。solver の自動再生と速度選択ができる
- 静的配信: `bash scripts/deploy-pages.sh`（ビルド → スモークテスト → `gh-pages` へ push）。
  手元で見るだけなら `zig build wasm -Doptimize=ReleaseFast` のあと `python3 -m http.server 8000 -d zig-out/web`。
  静的版の方針（既定 `nn3`）と自動再生の速度（既定は最速）は select で選べる。選べる方針は
  ビルドに同梱したものだけで、その一覧は `wasm.policies` が決める（下記）

## サイズ

| 何 | 大きさ | 中身 |
| --- | ---: | --- |
| `game2048.wasm` | 685 KiB | gzip 264 KiB。ゲーム + 同梱 solver（`wasm.policies`、既定は `nn2` と `nn3`）+ NN 重み |
| 初期メモリ | 2.44 MiB | `memory.grow` なし。スタック 1 MiB、静的データ 1.4 MiB（うち TT 640 KiB） |
| NN 重み | 148 KB | `@embedFile`。実行時に外部ファイルを取らない |
| 行テーブル | 512 KiB | `src/core.zig` の `row_table`（1 行のマージを引く 65,536 エントリ）。wasm の静的データの大半 |
| n-tuple 表 | 64.6 MiB | wasm には入れない（native の API 専用） |

ブラウザへ配るのは HTML と wasm の 2 ファイルだけ。NN 系は重み込みでこの中に収まるので、
サーバーも外部アセットも無しで動く。実測（ReleaseFast、V8 上の wasm、1 手あたり）は `nn2` が
9〜14 ms、`nn3` が序盤 0.3〜0.4 s / 盤面が詰まる終盤 4〜80 ms。`nn3` は 1 手ごとにメイン
スレッドを掴むので、遅くして眺めたいときだけ速度を選ぶ（既定は遅延なし）。

## 同梱する方針（`wasm.policies`）

ブラウザに配る wasm に入れる solver は**コードではなく設定ファイル** `wasm.policies` が決める
（1 行 1 名、`#` 以降はコメント）。ページの方針 select はモジュールの `algo_count()` /
`algo_name()` / `algo_score()` / `algo_speed()` から組み立てるので、同梱していない方針は
選べないし、UI とモジュールがずれることもない。select の下には実装の説明ではなく**測った平均
スコアと 1 手の時間**を出す（`policy_info` の `score` と `speed`）。未知の名前・重複・n-tuple 表が要る方針（`ntuple`）はビルドエラー。
ネイティブの CLI / HTTP サーバーはこのファイルを見ず、13 方針すべてを持つ。

既定の同梱は 2 つ:

| 方針 | 平均スコア | 1 手（V8） | ページに出る文 |
| --- | ---: | ---: | --- |
| `nn2` | 82,772 | 9〜14 ms | `nn2: 平均 82,772 点 / 1 手 9〜14 ms` |
| `nn3` | 116,205 | 4〜400 ms | `nn3: 平均 116,205 点 / 1 手 4〜400 ms` |

（スコアは native の ranking と同条件の実測。`nn3` は既定の ranking に入れていない重い方針なので
`#152` の計測値。1 手の ms はブラウザ（V8 on wasm）の実測で、native の `zig build ranking` の
ms/move とは別物（wasm は 3〜8 倍遅い）。測っていない方針は `speed` が空で、ページはスコアだけを
出す。実装の説明は `policy_info.description` にあり、CLI と README はそちらを使う。）

`nn3` は序盤ほど高い（3 手探索の枝が多い）。盤面が詰まると `nn2` と変わらない速さになるので、
自動再生の既定は `nn3`、速度は遅延なしにしてある。

`scripts/wasm-smoke.mjs`（`deploy-pages.sh` が必ず走らせる）が、この設定とビルドのずれを止める:
モジュールの一覧が `wasm.policies` と名前も順序も一致すること、各エントリに説明があること、
ページが `algo_count()` から一覧を作っていること、ページの既定（`DEFAULT_POLICY`）が同梱一覧に
含まれること、そして各方針の `solve` が 5 秒以内に返ること（返らない方針はハングとして落とす）。

## コア API

`src/core.zig`。UI も solver もここだけを使う。

- `GameState.gameStart(rng)` / `GameState.move(dir, rng)` / `moveUp/Left/Down/Right`
- `core.mergeLine(line)`: 1 行のスライド + マージ（1 ターン 1 回マージ）
- `spawnRandomTile` / `spawnAt` / `isGameOver` / `core.Direction.parse("up")`
- `core.Tile = u32`（`u16` だと 65536 のマージで溢れる）

`move` は「1 行を取り出す → `mergeLine` → 書き戻す → 変化したときだけ spawn」で全方向を統一している。

## Solver

`ai/solver/`。`2048.core` の公開 API だけを使い、core は solver を知らない。

| policy | 中身 |
|---|---|
| `expectimax`（既定）/ `expectimax_tt` / `expectimax_id` | 手作り評価関数 + expectimax。`_tt` は置換表、`_id` は確率枝刈り付き反復深化（時間予算依存で非決定） |
| `greedy` / `random` | 比較用 |
| `nn` / `nn2` / `nn3` | TD(0) で学習した MLP 価値関数。1-ply + spawn / 2-ply / 3-ply（`nn3` は 25 倍遅い） |
| `nnmcts` / `nnbeam` | NN を葉にした打ち切りロールアウト UCT / ビーム探索（`ranking` には未採用） |
| `ntuple` | n-tuple 表（7 パターン、10M ゲーム学習）の 1-ply。`ai/solver/ntuple_wide_q16.bin`（64.6 MiB）を遅延ロードするので API 専用（WASM 非対応、`ranking` 未採用） |
| `mcts` / `beam` / `ga` | 探索と GA 重みの比較用 |

評価関数は空きマス数・蛇行順の違反ペナルティ・同値隣接ペア（merge）に、monotonic / smoothness / tile_sum を足せる（`solver.Weights`）。

```bash
zig build solve -Doptimize=ReleaseFast -- 5 20 2 7000 expectimax_tt   # runs games depth seed policy
zig build ranking -Doptimize=ReleaseFast -- 5 20 1000                 # 全ポリシー比較 (JSON, latency/throughput 付き)
zig build ranking -Doptimize=ReleaseFast -- 5 20 1000 nn,nn2          # 末尾引数でポリシーを絞る
```

現状のおおよその強さ（ReleaseFast、評価 seed 9000、100 ゲーム。行ごとに条件が違うので数字は目安）:

`localhost:8080` のUIに出す比較表は `zig build ranking` の JSON をそのまま読む。
表を更新するときは、生成して `src/web/ranking.json` に置き直し、ビルドし直す
（`src/server.zig` が `@embedFile` しているので、置き換えただけでは古いまま）。

```bash
zig build ranking -Doptimize=ReleaseFast -- 5 20 1000 > src/web/ranking.json
```

表は avg score と一緒に速度（games/s と 1 手あたり ms）も出す。速度は同じ実行・同じマシン
の相対値なので、他の負荷が載っていると絶対値は動く（順位は動きにくい）。

| 手法 | avg score | 到達 | 速度 (games/s) |
|---|---:|---:|---:|
| `nn2`（committed weights, 100 ゲーム） | 90,358 | 8192 | 0.42 |
| `nn`（committed weights, 1-ply + spawn, 100 ゲーム） | 69,717 | 8192 | 4.8 |
| `expectimax_id`（2 ms/手, 100 ゲーム） | 75,202 | 8192 | — |
| `expectimax_tt`（depth 3, 200 ゲーム） | 52,736 | 8192 | 1.1 |
| `ntuple` 1-ply（wide 表 10M ゲーム） | 20,014 | 2048 | — |
| `expectimax`（depth 2, 既定, 100 ゲーム） | 34,000 | 4096 | 15 |

速度は `zig build ranking` の同じ実行から出した 1 秒あたりのゲーム数（スコアの行とは条件が違う）。
`nn2` は 1 手 3.2 ms で `expectimax` の 17 倍遅いが、スコアは 2.4 倍。

`nn` / `ntuple` の数値は手作り評価と違い、学習ゲーム数に強く依存する
（n-tuple の 2-ply は wide 表 10M で 51,951、TD の `nn` は 100k がピークで 400k まで回すと 73,007 に落ちる）。

数字は条件で動く。系譜・却下した案・A/B の履歴は [docs/solver-history.md](docs/solver-history.md)、
評価関数の比較基盤は [docs/evalab.md](docs/evalab.md) にある。

### 学習・実験ランナー

```bash
zig build evalab -Doptimize=ReleaseFast -- "<spec>;<spec>" 5 40 2 4000   # 評価関数の variant 比較
zig build tune   -Doptimize=ReleaseFast                                  # CMA-ES で重みを調整
zig build probe                                                          # 学習型評価の盤面選好を測る
zig build nn     -Doptimize=ReleaseFast -- 20000 100 2000 0.003 - base 30 /tmp/n.bin 1 par
zig build ntuple -Doptimize=ReleaseFast -- 200 20 2000 0.01 base /tmp/t.bin 1
zig build nn-quant -- ai/solver/nn_weights.bin /tmp/nn_weights_q.bin  # f32 -> int16 の変換と一致率 (#170)
zig build ga-rl  -- 16 15 8 2 1 500 100                                  # population generations games depth seed holdout_seed holdout_games
```

引数の意味は各ランナーの usage（引数なしで実行）に出す。variant の実行では重みを `/tmp` などリポジトリ外へ出す。
学習 variant（`nn` の `sym` / `nstep` / `death` など、`ntuple` の `light` / `wide`）の一覧と結果は
[docs/solver-history.md](docs/solver-history.md) を参照。

## ソース構成

| パス | 役割 |
|---|---|
| `src/core.zig` | ゲームロジック（純粋・RNG 注入） |
| `src/cli.zig` / `src/render.zig` / `src/main.zig` | 端末プレイとサブコマンドの振り分け |
| `src/server.zig` / `src/web/` | HTTP サーバー、JSON API、ブラウザ UI |
| `src/wasm.zig` | WASM の C ABI（ゲーム + solver、`@embedFile` で NN 重みを同梱） |
| `wasm.policies` | ブラウザ版 wasm に同梱する方針の一覧（ビルド時設定）。編集すると wasm が再ビルドされる |
| `ai/solver/solver.zig` | expectimax / 評価関数 / 反復深化 |
| `ai/solver/bench.zig` / `ranking.zig` / `parallel.zig` | ゲーム実行・集計・並列化 |
| `ai/solver/nn.zig` / `ntuple.zig` | TD(0) の価値関数（MLP / パターン表） |
| `ai/solver/evalab.zig` / `tune.zig` / `probe.zig` / `ga_rl.zig` | 評価関数の比較・調整・分析 |
| `docs/` / `scripts/` | 実験記録、ベンチ追記、WASM スモークテスト |

## 開発方針

- Core first: ロジックを先に固め、上位層は薄く保つ
- TDD: Red → Green → Refactor。`zig build test` が常に緑
- Deterministic: seed 固定で再現可能にする（時間予算を使う `expectimax_id` だけは例外）
- 変更は issue → worktree → PR。1 行の docs 修正は main へ直接 commit。手順は global の `dev-workflow` skill（`/skill:dev-workflow`）

## ライセンス

MIT（`LICENSE`）。タイルの配色は Gabriele Cirulli の 2048（MIT）由来で、著作権表示は `NOTICE` に置いている。
