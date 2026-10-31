# Game.2048

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
- 静的配信: `zig build wasm -Doptimize=ReleaseFast` → `zig-out/web/`（`index.html` + `game2048.wasm`、ReleaseFast で gzip 後 280 KiB）。
  確認は `node scripts/wasm-smoke.mjs`、表示は `python3 -m http.server 8000 -d zig-out/web`

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
