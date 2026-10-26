# Game.2048

Zig で 2048 を実装し、将来的に solver AI・WASM 展開まで見据えたプロジェクト。

## コンセプト

このリポジトリは **「2048 のゲームロジックを再利用可能なコアとして切り出す」** ことを最優先にする。
UI や実行環境に依存しない `2048.core` を中心に据えることで、次を同じロジックで展開できる状態を作る。

- ネイティブ実行 (まずは CLI/terminal)
- AI solver (探索・評価関数・方策比較)
- WASM 版 (ブラウザで遊ぶ・静的配信)

## 目標アーキテクチャ

`2048.core` は純粋ロジック層として設計する。

- `Board` と `GameState` のデータ構造
- `move` / `merge` / `spawn` / `game over` 判定
- 乱数注入ポイント (テスト容易性と再現性のため)
- シリアライズ可能な状態表現 (AI 学習/評価で使いやすく)

上位層は薄く保つ。

- `app/cli`: 入出力と描画のみ
- `ai/solver`: `2048.core` を読むだけで行動を決定
- `src/wasm.zig`: JS バインディング (`zig build wasm`) で `2048.core` を呼ぶ

## Core API

`src/core.zig` の公開 API。

- `core.GameState.gameStart(rng)` / `core.gameStart(rng)`: 初期盤面 (タイル 2 つ)
- `GameState.move(direction, rng)` と `moveUp/Left/Down/Right`: 移動・マージ・生成
- `core.mergeLine(line)`: 1 行のスライド + マージ (1 ターン 1 回マージ)
- `GameState.spawnRandomTile(rng)` / `spawnAt(pos, value)`: タイル生成
- `GameState.isGameOver()`: 終局判定
- `core.Direction.parse("up")`: 文字列から方向へ変換

`move` の内部は方向ごとに以下で統一している。

1. ボードから 1 行 (ベクトル) を取り出す (`cellAt` で向きを吸収)
2. `mergeLine` でスライド + マージする
3. 結果を盤面へ書き戻す
4. 盤面が変化した場合のみ生成 (spawn) を行う

タイル値は `core.Tile = u32`。`u16` だと 65536 のマージで溢れるため。

## 開発方針 (TDD)

TDD を基本サイクルにする。

1. Red: 仕様をテストで先に書く
2. Green: 最小実装で通す
3. Refactor: 構造を整える

特に以下をユニットテストで固定する (現在 `zig build test` で全件通過)。

- 各方向のスライドとマージ規則 (`moveUp/Down/Left/Right`)
- 1 ターン 1 回マージ制約、連鎖マージの禁止
- 盤面変化がないときの挙動 (生成しない・RNG を呼ばない)
- 乱数シード固定時の再現性
- 終局判定
- `spawnAt` の境界・占有・値の検証
- CLI のコマンド解析・描画、サーバーのルーティング

## Nix ベースの開発環境

Ghostty のように、**Nix で再現可能な開発環境** を前提にする。

- `nix develop` で Zig ツールチェインを取得
- ローカル環境差分に依存しないビルド/テスト
- CI でも同じ定義を使う

### セットアップ

```bash
nix develop
zig version      # 0.17.x を使う
zig build test
```

`flake.nix` は `devShell` を提供し、`zig` と (利用可能であれば) `zls` を使える。

> Zig **0.17** でビルドする。`build.zig` と `src/` は 0.17 の API (`std.Io`, `std.process.Init`) に合わせている。

## 起動方法

```bash
zig build run                 # 端末で遊ぶ (play, 既定)
zig build run -- serve 8080   # HTTP サーバー + ブラウザ UI (既定ポート 8080)
zig build run -- help         # 使い方
```

- `play`: `w/a/s/d` または `up/left/down/right` で移動、`n` でリセット、`q` で終了
- `serve`: `http://127.0.0.1:8080/` を開くと矢印キー/WASD で遊べる (日本語 IME が有効なままでも操作できる)
  - `GET /api/state`, `POST /api/new`, `POST /api/move?dir=up|left|down|right`
  - `GET /api/ranking`: ポリシーの比較結果 (`src/web/ranking.json`) を返す
  - `GET /api/solve?depth=N&algo=<policy>`: solver の次の一手を返す (盤面は変えない)。`algo` は `random` / `greedy` / `expectimax` (既定) / `mcts` / `beam` / `expectimax_tt` / `ga` / `nn` / `nn2` / `nnmcts` / `nnbeam`。`depth` の既定は 2。レスポンスは `{"algo","depth","dir","ranking"}` で、`dir` は合法手なしのとき `null`、`ranking` は `expectimax` / `greedy` のときだけ各手の評価が入り、それ以外は空配列
  - ブラウザの「Solver: 1 step」「Solver: auto」で solver が手を打つ様子を確認できる
  - 「speed」で auto の再生速度を選べる (600 / 300 / 150 / 50 ms / max = 待ち時間なし)。auto 中に変えると即座に反映される
  - 例: `curl -X POST "localhost:8080/api/move?dir=left"`
  - 127.0.0.1 のみで待ち受ける。1 リクエスト 1 接続 (ローカル試作用)

## 静的配信 (WASM)

`zig build wasm -Doptimize=ReleaseFast` で、ゲームと solver を WebAssembly にまとめ、サーバー無しで遊べる静的サイトを `zig-out/web/` に出力する。HTTP サーバー (`zig build run -- serve`) は従来どおり動く。

- `zig-out/web/index.html`: 静的 UI (`src/web/index.static.html` をコピー)。`/api/*` を一切使わず、`game2048.wasm` を直接呼ぶ
- `zig-out/web/game2048.wasm`: ゲーム + solver。`nn` の重みは `@embedFile` で埋め込み済み
- ABI は `src/wasm.zig` 冒頭のコメント参照: `init(seed_lo, seed_hi)` / `move(dir)` / `solve(algo, depth)` / `board_ptr()` / `score()` / `best_tile()` / `algo_count()` / `algo_name(i)`

サイズ (ReleaseFast): raw 753,911 B (736 KiB)、`gzip -9` 後 286,245 B (280 KiB)。

```bash
zig build wasm -Doptimize=ReleaseFast
node scripts/wasm-smoke.mjs                      # init/move/solve のスモークテスト (失敗時は非 0)
python3 -m http.server 8000 -d zig-out/web       # http://localhost:8000/ で遊ぶ
```

`game2048.wasm` は `application/wasm` で配信される必要がある (python の `http.server` と GitHub Pages は対応済み)。GitHub Pages などに公開する場合は `zig-out/web/` の中身 (`index.html` と `game2048.wasm`) をそのまま置く。

静的版 UI は埋め込み UI (`src/web/index.html`) とは別ファイルにしてある。サーバー版は `/api/ranking` の表を出すため、同じファイルで wasm と `/api/` を切り替えるより単純なため。

## Solver (AI)

`ai/solver/` に置き、`2048.core` の公開 API (`slide` / `move` / `isGameOver` など) だけを使う。core は solver を知らない。

- `solver.bestMove(state, depth)`: expectimax で最良手を返す (`null` は合法手なし)
- `solver.evaluate(board)`: 空きマス数 + 蛇行順 (1..16、行ごとに左→右、右→左、左上が最大) に沿った大小関係の違反ペナルティ + 隣接同値ペア数 × merge bonus (既定 10)
- `solver.evaluateState(state)`: `evaluate` に「down しか合法手がない盤面」の大きなペナルティを加えた葉の評価
- `solver.playGame(rng, depth)`: 1 ゲーム分プレイして結果を返す

検証 (ReleaseFast, seed 1, 100 games):

```bash
zig build solve -Doptimize=ReleaseFast -- 1 100 2 1   # runs games/run depth seed = 100 games, depth 2, seed 1
```

| 版 | depth | avg score | avg moves | best tile | 1024 到達 |
|---|---|---|---|---|---|
| 角ボーナス版 (seed 1) | 1 | 3,673 | 292 | 512 | 0 / 100 |
| 角ボーナス版 (seed 1) | 2 | 7,606 | 512 | 1024 | 13 / 100 |
| 行順 + forced-down 版 (seed 1) | 1 | 2,843 | 236 | 512 | — |
| 行順 + forced-down 版 (seed 1) | 2 | 18,596 | 1,032 | 4096 | 最大 4096 2 / 100、2048 38 / 100 |
| 角ボーナス版 (seed 2) | 2 | 7,866 | 524 | 1024 | — |
| 行順 + forced-down 版 (seed 2) | 2 | 19,812 | 1,087 | 4096 | — |
| 左上最大 + forced-down 版 (seed 1) | 1 | 3,343 | 265 | 1024 | — |
| 左上最大 + forced-down 版 (seed 1) | 2 | 19,198 | 1,069 | 4096 | — |
| 左上最大 + forced-down 版 (seed 2) | 2 | 19,352 | 1,067 | 4096 | — |
| 左上最大 + 順序ペナルティ x30 (seed 1) | 2 | 23,204 | 1,247 | 4096 | — |
| 左上最大 + 順序ペナルティ x30 (seed 2) | 2 | 21,840 | 1,177 | 4096 | — |
| 蛇行順 + 順序ペナルティ x30 (seed 1) | 2 | 26,716 | 1,408 | 4096 | — |
| 蛇行順 + 順序ペナルティ x30 (seed 2) | 2 | 24,935 | 1,334 | 4096 | — |

蛇行順は行順と同じ重み (x30、空きマス 10) のまま変更したもの。重みは行順向けに調整した値で、蛇行順向けの再調整はしていない。

順序ペナルティの重み (空きマス 10 固定、100 games、depth 2、seed 1 / seed 2 の平均):
x3 は約 10,500、x10 は 19,275、x30 は 22,522 (採用)、x50 は 20,308、x100 は 21,051。空きマス重みを 5 / 30 に変えると 20,897 / 19,980 で、10 が最良。

左上最大版は、初版で「逆向きは弱い」と出た計測（5,421 / 5,106）が、先頭タイルを誤って罰する不具合によるものだったため、修正後に再計測した値です。

行順評価では depth 1 が角ボーナス版より弱くなります (depth 2 以上で強い)。depth 3 は未計測 (角ボーナス版は 20 games で約 23 秒)。

ポリシー比較・NN・GA のランナー:

```bash
zig build ranking -Doptimize=ReleaseFast -- 5 20 1000   # runs games/run seed (全ポリシーを同じ seed で比較、JSON 出力)
zig build nn -Doptimize=ReleaseFast -- 0 100 2000 0.003 # train_games eval_games seed lr (train_games=0 は保存済み重みを読み込む)
zig build ntuple -- 200 20 2000 0.01 base /tmp/t.bin 1  # train_games eval_games seed lr variant table_path train_seed
zig build ga-rl -- 16 15 8 2 1 500 100                  # population generations games depth seed holdout_seed holdout_games
```

`expectimax` 系に加えて、GA の重みを使う `ga` と TD 学習した MLP 価値関数の `nn` (1-ply + 出現1回) / `nn2` (2-ply)、実験中の `nnmcts` (8 手で打ち切ったランダムロールアウトを NN の盤面価値で評価する UCT) と `nnbeam` (NN 評価のビーム探索: 幅8・3手先・各ノードで spawn を2回サンプリング) も選べる (`/api/solve?algo=nnmcts` のように指定)。`nnmcts` は seed 1000 の 10 ゲームで 20,137 (5.4 ms/手)、`nnbeam` は同 seed で 23,875 といずれも `nn` (68,772) に遠く及ばないため `ranking.json` には入れていない (`docs/solver-history.md` の "Truncated-rollout MCTS with the NN leaf: nnmcts (#82)" と "NN-value beam with sampled spawns (#83)")。n-tuple は学習・評価用のランナーのみで、`algo` には未接続 (light 表 + 深さ2探索で seed 2000 の 20 ゲーム平均 25,760、expectimax 深さ2 の 21,840 を上回る。`docs/solver-history.md` の "n-tuple: deep search, light tables, resume (#73)")。

## ソース構成

| ファイル | 役割 |
|---|---|
| `src/core.zig` | ゲームロジック (純粋・RNG 注入) |
| `src/render.zig` | 盤面のテキスト描画 |
| `src/cli.zig` | 端末プレイのコマンド解析とループ |
| `src/server.zig` | HTTP サーバー、JSON API、埋め込み UI |
| `src/wasm.zig` | WASM 版の C ABI (ゲーム + solver を JS から呼ぶ) |
| `src/web/index.html` | ブラウザ UI (サーバーに埋め込み) |
| `src/web/index.static.html` | 静的配信用のブラウザ UI (wasm のみ、`zig build wasm` が `zig-out/web/index.html` にコピー) |
| `src/main.zig` | サブコマンドの振り分け |
| `ai/solver/solver.zig` | expectimax solver と評価関数 (core のみ依存) |
| `ai/solver/main.zig` | `zig build solve` の検証用ランナー |
| `ai/solver/bench.zig` | solver のベンチマーク実行と集計 |
| `ai/solver/parallel.zig` | ゲーム単位の並列実行 (決定性は seed ごとに保持) |
| `ai/solver/ranking.zig` | `zig build ranking` (全ポリシー比較、JSON 出力) |
| `ai/solver/nn.zig` | TD(0) で学習する小さな MLP 価値関数 |
| `ai/solver/nn_main.zig` | `zig build nn` の学習・評価ランナー |
| `ai/solver/ntuple.zig` | TD(0) で学習する n-tuple 価値関数 (パターン表) |
| `ai/solver/ntuple_main.zig` | `zig build ntuple` の学習・評価ランナー |
| `ai/solver/ga_rl.zig` | `zig build ga-rl` (GA による評価重みの進化) |
| `docs/solver-history.md` | solver の試行錯誤・計測・却下した案の記録 |
| `scripts/record-bench.sh` | ベンチマーク出力を `docs/solver-history.md` に追記するスクリプト |
| `scripts/wasm-smoke.mjs` | WASM ビルドのスモークテスト (node、依存なし) |

## マイルストーン

1. ✅ `2048.core` の最小実装 + テスト基盤
2. ✅ CLI でプレイ可能にする / ✅ ローカル HTTP サーバーで試せる
3. ✅ solver AI の PoC (expectimax, 手作り評価関数) / ✅ 軽量かつ高スコアの方式 (TD 学習の MLP を `nn` ポリシーとして採用。ranking seed 1000 で 64,113 と 1 位) / ⬜ n-tuple の採用 (ランナーのみ実装済み #70。light 表 + 深さ2 探索で expectimax 深さ2 超えまで確認 #73、`algo` 接続は未着手)
4. ✅ WASM ターゲット対応 (静的配信)

## このプロジェクトで大事にすること

- Core first: ロジックを最初に固める
- Deterministic: テスト可能で再現可能にする
- Portable: CLI / AI / WASM へ横展開しやすくする
