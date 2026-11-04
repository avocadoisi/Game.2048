# 候補 net の判定と計測（#207 / L1）

同梱 net（`ai/solver/nn_weights.bin`）を差し替えてよいかを、毎回同じ数字で決めるための手順。
候補は「同梱 net と同じ 2-ply の打ち方」で比べるので、強さの差だけが残る。

## 判定基準

1. **`pairw`（A = 同梱 net、B = 候補、20 局、seed 9000）で `mean diff - 2*stderr > 0`。**
   `mean diff` は B − A の平均、`stderr` はその標準誤差。20 局だと stderr は 6,000〜10,000 点で、
   採用に必要な差はおおよそ **+15,000 点**（+2σ）。1 seed / 20 局で足りないと感じたら `--games 40` と
   seed を増やす。σ はゲーム数でしか縮まない（seed を増やすと spawn の分散が減る）。
2. **非ペア表（seed 9000 と 9100、`.deep` 20 局）で悪化しない。**
   `avg_score` と `reach8192` の両方を見る。ペア比較は「同じ spawn でどちらが伸びるか」しか見ないので、
   分布の裾（8192 到達）が落ちていないかをここで確認する。ペアと非ペアが食い違ったら **ペアを採る**
   （非ペアは spawn 運で ±10,000 点動く）。
3. **ms/move が悪化しない。** 学習側の改善（#204 / L3）と混ぜないため、評価のコストは候補ごとに記録する。

3 つとも通ったときだけ `ai/solver/nn_weights.bin` を差し替え、`docs/solver-history.md` に
「何をどこまで回して、どの数字で採ったか」を残す。

## 手順（1 コマンド）

```sh
scripts/eval-net.sh /tmp/cand.bin                 # 20 局 seed 9000、同梱 net と比較
scripts/eval-net.sh /tmp/cand.bin --games 40      # σ を縮めたいとき
scripts/eval-net.sh /tmp/cand.bin --shipped /tmp/base.bin
```

出力は 3 ブロック:

```
== pairw (A = ai/solver/nn_weights.bin, B = /tmp/cand.bin, 20 games, seed 9000) ==
pair A vs B: games=20 mean diff=1183.8 ± 9370.3 (stderr) better=9 worse=11 elapsed=254.8 s

== non-paired 2-ply table ==
net      seed  avg_score reach8192  ms/move
shipped  9000    79657.0      4/20     0.43
shipped  9100    91348.0      5/20     0.44
cand     9000    93920.0      6/20     0.44
cand     9100    85581.8      5/20     0.41

VERDICT: REJECT (mean diff 1183.8 ± 9370.3, B better/worse 9/11; rule: mean diff - 2*stderr > 0)
```

`VERDICT` は 1 の条件だけを見た機械判定。2 と 3 は表を読んで判断する。

## 生のコマンド（スクリプトの中身）

```sh
# ペア比較（A = weights_path、B = `-` の後ろのパス、同じ spawn tape）
zig build nn -Doptimize=ReleaseFast -- 0 20 9000 0.003 1 pairw 30 ai/solver/nn_weights.bin 1 - /tmp/cand.bin

# 非ペア表（rows=0,2 で 1-ply と 2-ply だけ。3-ply 行は 1 本 20 分なので測らない）
zig build nn -Doptimize=ReleaseFast -- 0 20 9000 0.003 1 resume 30 /tmp/cand.bin 1 - - - 0,2

# 学習そのもの（`base` から 20k ゲーム、par の sqrt 則 = 既定）
zig build nn -Doptimize=ReleaseFast -- 20000 0 9000 0.003 1 base 30 /tmp/cand.bin 1 par
```

`eval_games = 0` は「学習だけ」で評価表を出さない。`train_games = 0` は重みを読むだけで学習しない。
学習は **1 本ずつ**（並列は実装・レビューだけ）。

## 現時点の最良候補（#207 まで）

`dist` 20,000 ゲーム（教師 458,752 サンプル、every 4 / div 3）が **+1,184 ± 9,370** で最良。
これは「同梱 net を壊さない」上限で、採用線 +15,000 には届かない（全ラウンドは下の L4 の表）。
`ft2` は廃止し、同じことをするならラベルを 1 回だけ固定する `tch` → `dist` を使う。
次に効くのはゲーム数ではなく教師ラベル自体の改善（`docs/nn-backlog.md` の #184 / #185）。

## スループットの before / after（L3）

同じ機械（4 物理コア、ReleaseFast）で測った値。avg_score と 8192 到達はどの行も変えていない。

| 何を | before | after | 根拠 |
|---|---|---|---|
| `zig build nn` の前段 | 実行ステップが install に依存し、game-2048 / solve / ranking / ntuple-quant を毎回 ReleaseFast で作り直す（~25 分） | 自分の exe だけ（再ビルド込み ~150 s） | `build.zig`（PR #214） |
| `base` 20,000 ゲーム | 112 s（178 games/s、1 コア） | 32 s（625 games/s、`par` + sqrt 則） | #208 / #203 |
| 評価表 1 本 | 15 行（3-ply 行が支配的）で 20 分超 | `rows 0,2` で ~70 s | #203 |
| `dump` 65,536 局面 | 37 分（`dumpp2` 200 局） | 同じ（探索コストが本体） | #207 |
| `tch` 65,536 局面 | 100 s | 同じ（`tchadd` で追記すると 1 本 100 s のまま増やせる） | #207 |
| `pairw` 20 局 | 250〜280 s | 同じ | #210 |
| `dist` 20,000 ゲーム | 2,300 s（直列） | `par` で 90 s だが **30,000 点負ける**ので不採用 | #207 |

**速度だけを採らない**。`par` + `dist` は 25 倍速いが判定に通らず、`dist` は直列のまま回す。

## 候補ラウンドの記録（L4）

すべて A = 同梱 net、`pairw` 20 局 seed 9000。採否は `mean diff - 2*stderr > 0` かつ非ペア悪化なし。

| # | 候補 | 数値（± stderr） | 採否と理由 |
|---|---|---|---|
| 1 | `restart` 5,000 ゲーム × 3 seed（dump は `dumpp2`） | −16,594 ± 9,809 / −18,728 ± 6,655 / −8,596 ± 6,819 | 不採用。深い局面だけを踏むと前半の `V` を保つ信号が消える |
| 2 | `dist` 5,000（教師 65,536、every 8 / div 10） | −4,441 ± 8,391 | 不採用（互角）。50/50 の every 1 は −32,359 ± 6,553 で論外 |
| 3 | `dist` 20,000（教師 7 dump = 458,752、every 4 / div 3） | +1,184 ± 9,370 | 不採用。最良だが +15,000 に届かない（互角） |
| 4 | `par` + `dist` 20,000（batch_scale 1 / √8 / 8） | −31,273 / −34,859 / −30,117 | 不採用。25 倍速いが教師信号が薄まる |
| 5 | `ft2`（深い側の教師を `deep3` 固定ラベルに修正、lr 3e-5〜1e-6、deep_every 1〜8） | 発散（NaN、\|w\| 1e7 超） | 廃止。教師が学習中に動く限り `V` が自分のラベルを押し上げる |
| 6 | lr スケジュールで 200k〜400k ゲーム（固定 lr / 段階的半減 / 2 段 /4） | −11,260 ± 6,566 / −11,076 ± 7,912 / −12,723 ± 6,019 / −9,385 ± 6,870 | 不採用。200k 以降は平坦で同梱 net に届かない |
| 7 | `dist` 20k の配合振り（every 4/div 2、every 2/div 3、every 6/div 3） | −16,585 ± 11,599 / −20,110 ± 6,303 / −30,848 ± 5,586 | 不採用。互角は every 4 / div 3 の 1 点だけ |
| 8 | `dist` 20k、教師を `h4`（589,824、4-ply in endgame）で貼り直し | −15,588 ± 6,249 | 不採用。教師を深くしても伸びない（ラベル差 mean\|Δ\| 6.4） |
| 9 | `dist` 100k（every 4 / div 3、教師 458,752） | −4,696 ± 8,005 | 不採用。蒸留は互角で飽和する（1,788.7 s） |
| 10 | 128/128 の 300k ゲーム（スクラッチ、lr 0.003、par sqrt） | +8,624 ± 7,679（13 勝 7 敗）、非ペア 92,953 | 不採用。seed 2 で再試すると **−12,117 ± 5,848（5 勝 15 敗）**で再現せず、seed の引きだった |
| 11 | 容量の追加学習・蒸留での移植（128/128 → 64/32、`dson` 30k / `dist` 100k） | −68,521 ± 7,048 / −55,826 ± 6,287 | 不採用。広い net の `V` は自分の探索とセットでしか意味を持たず、生徒に移せない |

**同梱 net の差し替えは行っていない**（`ai/solver/nn_weights.bin` は未変更）。11 ラウンドすべて
負けか互角。幅を広げると seed 1 だけ正の差が出た（#220）が、seed 2 で符号が反転し、コストは 12 倍。
ゲーム数・lr スケジュール・蒸留の配合/教師/移植・容量のどれも +2σ に届かないので、次の一手は
報酬関数か局面表現の変更（`#184` / `#185` の前段にある「教師誤差の測定」）になる。

## 実測の目安（この Mac、ReleaseFast、4 物理コア）

| 何を | 時間 |
|---|---|
| `pairw` 20 局（`.deep`） | 250〜280 s |
| 非ペア `.deep` 20 局 1 本 | 35 s + `zig build` の再チェック約 150 s |
| `zig build test` 全ターゲット（Debug） | 約 50 分（nn のテストバイナリが支配的）、2 回目はキャッシュで 1 秒 |
| `base` 20k ゲーム（par sqrt） | 約 340 s |
| `dump` 1 本（65,536 深い局面, `dumpp2` 200 局） | 約 37 分 |
| `tch` 1 本（65,536 局面の deep3 ラベル） | 約 100 s |

`zig build nn` は **必ず `--` を付ける**。付けないと既定の 2000 ゲーム学習が走り、同梱 net を上書きする
（`git checkout -- ai/solver/nn_weights.bin` で戻す）。

## 再現性の確認（L1）

`scripts/eval-net.sh` を 2 つの net で通し、同じ数字が出ることを確認した（2026-11、ReleaseFast）。

| 渡した net | pairw mean diff | better/worse | 非ペア avg_score（9000 / 9100） | 8192 | 判定 |
|---|---|---|---|---|---|
| `/tmp/pdm_s1.bin`（`par` + `dist` 20k） | −31,273 ± 6,653 | 0/20 | 67,502 / 59,537 | 1/20 / 1/20 | REJECT |
| `/tmp/pd8.bin`（`par` + `dist` 20k、scale 8） | −30,117 ± 5,522 | 1/19 | 58,990 / 58,193 | 0/20 / 0/20 | REJECT |

同梱 net の行（79,657 / 91,348、4/20・5/20）も毎回同じ値で出る。1 本あたり約 6 分
（`pairw` 230 s + 非ペア 4 行 140 s）。
