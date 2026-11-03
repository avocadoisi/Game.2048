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

| 候補 | 作り方 | B − A（± stderr） | 判定 |
|---|---|---|---|
| 蒸留 20k（教師 458,752） | `dist`、every 4 / div 3 | +1,184 ± 9,370 | REJECT（互角） |
| 蒸留 5k | `dist`、every 8 / div 10 | −4,441 ± 8,391 | REJECT |
| `restart` 5k × 3 seed | `restart`、lr 3e-4 | −8,596 … −18,728 | REJECT |
| `ft2`（50% 深い局面） | 1 回だけラベルを貼る `tch` → `dist` に置き換え | 発散（NaN） | 廃止 |

`dist` が「同梱 net を壊さない」上限で、上の表のどれも +15,000 に届いていない。次に効くのは
教師ラベル自体の改善（`docs/nn-backlog.md` の #184 / #185）で、ゲーム数ではない。

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
