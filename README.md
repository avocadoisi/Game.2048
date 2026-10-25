# Game.2048

Zig で 2048 を実装し、将来的に solver AI・WASM 展開まで見据えたプロジェクト。

## コンセプト

このリポジトリは **「2048 のゲームロジックを再利用可能なコアとして切り出す」** ことを最優先にする。
UI や実行環境に依存しない `2048.core` を中心に据えることで、次を同じロジックで展開できる状態を作る。

- ネイティブ実行 (まずは CLI/terminal)
- AI solver (探索・評価関数・方策比較)
- 将来的な WASM 版 (Web UI への接続)

## 目標アーキテクチャ

`2048.core` は純粋ロジック層として設計する。

- `Board` と `GameState` のデータ構造
- `move` / `merge` / `spawn` / `game over` 判定
- 乱数注入ポイント (テスト容易性と再現性のため)
- シリアライズ可能な状態表現 (AI 学習/評価で使いやすく)

上位層は薄く保つ。

- `app/cli`: 入出力と描画のみ
- `ai/solver`: `2048.core` を読むだけで行動を決定
- `wasm` (future): JS バインディングで `2048.core` を呼ぶ

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
- `serve`: `http://127.0.0.1:8080/` を開くと矢印キー/WASD で遊べる
  - `GET /api/state`, `POST /api/new`, `POST /api/move?dir=up|left|down|right`
  - `GET /api/solve?depth=N`: solver の次の一手を返す (盤面は変えない。`{"dir":"left"}`、合法手なしは `null`)
  - ブラウザの「Solver: 1 step」「Solver: auto」で solver が手を打つ様子を確認できる
  - 「speed」で auto の再生速度を選べる (600 / 300 / 150 / 50 ms / max = 待ち時間なし)。auto 中に変えると即座に反映される
  - 例: `curl -X POST "localhost:8080/api/move?dir=left"`
  - 127.0.0.1 のみで待ち受ける。1 リクエスト 1 接続 (ローカル試作用)

## Solver (AI)

`ai/solver/` に置き、`2048.core` の公開 API (`slide` / `move` / `isGameOver` など) だけを使う。core は solver を知らない。

- `solver.bestMove(state, depth)`: expectimax で最良手を返す (`null` は合法手なし)
- `solver.evaluate(board)`: 空きマス数 + 「左上から行順 (1..16) に大きいタイルが並ぶ」大小関係の違反ペナルティ (左上が最大)
- `solver.evaluateState(state)`: `evaluate` に「down しか合法手がない盤面」の大きなペナルティを加えた葉の評価
- `solver.playGame(rng, depth)`: 1 ゲーム分プレイして結果を返す

検証 (ReleaseFast, seed 1, 100 games):

```bash
zig build solve -Doptimize=ReleaseFast -- 100 2 1   # games depth seed
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

左上最大版は、初版で「逆向きは弱い」と出た計測（5,421 / 5,106）が、先頭タイルを誤って罰する不具合によるものだったため、修正後に再計測した値です。

行順評価では depth 1 が角ボーナス版より弱くなります (depth 2 以上で強い)。depth 3 は未計測 (角ボーナス版は 20 games で約 23 秒)。

## ソース構成

| ファイル | 役割 |
|---|---|
| `src/core.zig` | ゲームロジック (純粋・RNG 注入) |
| `src/render.zig` | 盤面のテキスト描画 |
| `src/cli.zig` | 端末プレイのコマンド解析とループ |
| `src/server.zig` | HTTP サーバー、JSON API、埋め込み UI |
| `src/web/index.html` | ブラウザ UI (サーバーに埋め込み) |
| `src/main.zig` | サブコマンドの振り分け |
| `ai/solver/solver.zig` | expectimax solver と評価関数 (core のみ依存) |
| `ai/solver/main.zig` | `zig build solve` の検証用ランナー |

## マイルストーン

1. ✅ `2048.core` の最小実装 + テスト基盤
2. ✅ CLI でプレイ可能にする / ✅ ローカル HTTP サーバーで試せる
3. ✅ solver AI の PoC (expectimax, 手作り評価関数) / ⬜ 軽量かつ高スコアの方式の検討 (n-tuple など)
4. WASM ターゲット対応

## このプロジェクトで大事にすること

- Core first: ロジックを最初に固める
- Deterministic: テスト可能で再現可能にする
- Portable: CLI / AI / WASM へ横展開しやすくする
