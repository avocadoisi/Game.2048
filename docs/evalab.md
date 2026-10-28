# evalab — 評価関数バリアントの比較

`ai/solver/evalab.zig` は同じ seed で複数の評価重みを対局させ、avg score / moves / best / 速度を並べる。
`Weights` の既定値は出荷時の評価と同じなので、`baseline` は従来の expectimax そのもの。

> 2026-10-27 以降、出荷時重みには `monotonic = 32` が入っている（[採用: monotonic](#採用-monotonic-117)）。
> それより前の節に出てくる `baseline` は `monotonic = 0` の意味なので、`monotonic=0` と読み替える。

```sh
zig build evalab -Doptimize=ReleaseFast -- "<specs>" <runs> <games_per_run> <depth> <seed>
# 例: zig build evalab -Doptimize=ReleaseFast -- "baseline;monotonic=24" 10 40 2 7000
```

spec は `;` 区切り、各 spec は `,` 区切りの `key=value`。
key: `layout` `anchor` `ordered` `empty` `order` `merge` `forced_down` `smoothness` `monotonic` `tile_sum`、
preset: `baseline` `ga`。

## 追加したノブ (#111)

| ノブ | 意味 | 既定 |
| --- | --- | --- |
| `layout` | 順序ペナルティの経路形状 (`snake` `column_snake` `spiral` `block2` `flex` `flex_reverse`) | `snake` |
| `anchor` | 経路の向き (0..7 = 90°CW 回転 ×4 × 列ミラー)。0 が出荷時 | `0` |
| `ordered` | 経路の先頭 N マスだけ順序を課し、残りは自由マスにする | `16` |
| `smoothness` | 隣接タイルの \|log2 差\| 和のペナルティ | `0` |
| `monotonic` | 各行・各列の単調性違反量のペナルティ | `0` |
| `tile_sum` | 盤面の log2 合計ボーナス | `0` |

すべて 0 (または既定値) のとき出荷時の評価と完全に一致する（テストで固定）。
`ordered=16` は全マスに順序を課す＝従来どおり。

## 実験 2026-10-27 (depth 2, ReleaseFast)

共通: 各バリアント `runs` × `games_per_run` 局、全バリアントが同じ seed を使う。

### 経路形状 (#112), 5×40, seed 4000

| layout | avg score | avg moves | best | games/s |
| --- | --- | --- | --- | --- |
| snake (baseline) | 24345.3 | 1311.1 | 4096 | 27.4 |
| column_snake | 24869.1 | 1320.5 | 4096 | 27.6 |
| spiral | 20591.1 | 1135.3 | 4096 | 31.0 |
| block2 | 13605.7 | 802.2 | 2048 | 46.4 |

行スネークと列スネークは同等（差は seed ノイズ）、spiral は −15%、block2 は −44%。
「角の 2x2 ブロックに大きく育てる」形は、この評価では育たない。

### アンカー (#113), 5×40, seed 4000

| anchor | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg score | 24345.3 | 24806.4 | 24715.3 | 24869.1 | 24946.4 | 24491.6 | 24806.0 | 24446.2 |

8 通りすべて 24345–24946 に収まり、差は ±1.2%。盤面評価は向きに依存しない（対称性どおり）。
`forced_down` は左上アンカー前提の項なので、回転させても利得は出ない。

### 順序を課すマス数 `ordered` と flex レイアウト (#112), 5×40, seed 4000

| spec | avg score | avg moves | best |
| --- | --- | --- | --- |
| snake, ordered=16 (baseline) | 24345.3 | 1311.1 | 4096 |
| snake, ordered=12 | 20059.2 | 1112.0 | 4096 |
| snake, ordered=10 | 9100.4 | 578.1 | 2048 |
| snake, ordered=8 | 11357.3 | 700.1 | 2048 |
| flex, ordered=7 | 4460.0 | 334.4 | 1024 |
| flex_reverse, ordered=7 | 9184.6 | 584.7 | 2048 |

`flex` は「7 マスだけ順序を決め、残り 9 マスは自由」という盤面（順序 1..7 = (1,3) (2,2) (2,3)
(2,0) (3,2) (3,0) (3,3)、他は自由）。

結果: 順序を課すマスを減らすほど単調に悪化する。この評価では「全 16 マスに順序を課す」ことが
主な駆動力で、自由マスを増やすと盤面が崩れて 1/5 以下のスコアになる。
flex の狙い（自由マスを残して身動きを良くする）は、depth 2 の探索では成立しない。

### smoothness / monotonic / tile_sum (#114)

seed 4000, 5×40:

| spec | avg score |
| --- | --- |
| baseline | 24345.3 |
| smoothness=1 | 25303.5 |
| smoothness=3 | 24700.8 |
| monotonic=1 | 24335.4 |
| monotonic=3 | 30097.1 |
| tile_sum=0.5 | 26077.7 |
| tile_sum=2 | 24049.8 |
| smoothness=2, monotonic=2 | 25881.4 |

monotonic が効く。重みを掃引（seed 5000 / 6000, 5×40）:

| monotonic | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 5000 | 25450 | 26993 | 29767 | 30484 | — | — | — | — | — | — | — | — |
| seed 6000 | — | — | — | 28740 | 31955 | 32110 | 32266 | 33402 | 34674 | 32375 | 29554 | 26665 |

seed 6000 では 24–32 に山がある。32 で 8192 に到達した局が出た（他は 4096）。

### 確認 (holdout, 10×40, seed 7000, depth 2)

| spec | avg score | avg moves | best |
| --- | --- | --- | --- |
| baseline | 24757.7 | 1322.3 | 4096 |
| monotonic=24 | 33896.0 | 1733.4 | 4096 |
| monotonic=32 | 34116.9 | 1749.1 | **8192** |
| monotonic=32, order=0 | 30870.2 | 1605.4 | 4096 |
| monotonic=24, tile_sum=0.5 | 33650.8 | 1727.5 | 4096 |

- monotonic を足すと平均スコア +37%（24758 → 34117）、平均手数 +32%、8192 到達が初めて出る。
- `order=0` にすると −9.5% なので、スネーク順ペナルティは monotonic と併用しても必要。
- tile_sum の上乗せは無い（差は誤差）。

### depth 3 での確認 (seed 7000, 2×8 局, 小標本)

| spec | avg score | avg moves |
| --- | --- | --- |
| baseline | 40692.3 | 1994.3 |
| monotonic=32 | 44134.8 | 2196.3 |

深く読むほど利得は縮む（+37% → +8.5%）。8 局ずつなので有意とは言えない。採用判断は #117。

## 再現手順の注意

- 速度列は比較用。並列ビルドなど他の負荷が乗ると下がる。
- `zig build test` の全体 ReleaseFast 実行は nn テストのコンパイルが 20 分以上かかる。
  単体テストは Debug（既定）か `-Dtest-filter=<名前の一部>` を使う。

## 既知の限界

- 評価は seed 4000–7000 の holdout で確認したが、depth 2 の expectimax のみ。depth 3 以上や
  nn ポリシーでの順位は未検証（depth 3 は 1 局あたりのコストが跳ね上がる）。
- avg score / moves / best だけを見ている。到達率の分布・死因（盤面が詰まる形）は未記録。

## 学習型評価の選好 (#115)

手作りの snake 評価と比べて、学習型の値関数（NN / n-tuple）が違う盤面形状を好むかを、
少数の固定プローブ盤面で測る。`ai/solver/probe.zig`:

```sh
zig build probe
# n-tuple のテーブルを差し替える場合
zig build probe -- ai/solver/ntuple_weights.bin
```

全プローブ盤面は同じタイル多重集合 `{2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4}`（残り 6 マスが空）で、
置き方（形）だけを変えている。タイルは経路に沿って降順に置く。

| board | 置き方 |
| --- | --- |
| `snake` | 左上から行スネーク（出荷時経路） |
| `column_snake` | 列スネーク（`snake` の anchor 3 と同じ経路） |
| `spiral` | 外回り時計回り → 内側 2x2 |
| `flex` | flex 経路（順位 1..7 のセル）に降順。2048 は (1,3) |
| `block2` | 2x2 ブロック単位。左上 2x2 に 2048–256 |
| `snake anchor 7` | snake 経路を 180° 回転＋列ミラー（右下アンカー） |
| `scrambled` | 大きいタイルを飛び飛びに散らした最悪形 |

評価器は 3 つ: `snake` = `solver.evaluateW`（出荷時重み）、`nn` = 埋め込み `nn_weights.bin`、
`ntuple` = `ntuple_weights.bin`（68 MB を実行時に読み込み）。値は評価値そのもの、`rank` は評価器内の順位（1 = 最良）。

```
board                     snake  rank             nn  rank         ntuple  rank
snake                   60.0000   1.5         0.9222     3         0.7336   2.5
column_snake          -371.7400     5        -3.3625     6         0.7336   2.5
spiral                  60.0000   1.5         1.5035     2         0.1955     5
flex                  -501.2620     7        -1.6711     5        -0.0048     7
block2                -155.8700     3        -4.3314     7         0.7336   2.5
snake anchor 7        -414.9140     6         9.0475     1         0.7336   2.5
scrambled             -328.5660     4        -1.4902     4         0.0445     6

spearman rank correlation (1 = same preference order, -1 = reversed, 0 = unrelated)
  snake  vs nn       0.126
  snake  vs ntuple   0.288
  nn     vs ntuple  -0.020
```

### 結論

- 評価器ペアの順位相関は snake↔nn 0.13、snake↔ntuple 0.29、nn↔ntuple −0.02 で、いずれも 1.0 から遠い。
  学習型評価は手作り snake と同じ形を好んでいない。ただし盤面 7 枚では Spearman の標準誤差が約 0.4 あり、
  個々の値は「無相関」と区別できない。言えるのは「一致していない」ところまで。
- `nn` は 180° 回転した snake 盤（9.05）と spiral（1.50）を左上 snake 盤（0.92）より高く評価し、
  flex・column_snake・block2 は負値にする。手作り snake が 3 位に置く block2 を nn は最下位にする。
  形の好みは明確に違う。
- `ntuple` は経路順に並んだ 4 盤（snake / column_snake / block2 / 回転 snake）が同値 0.7336 で並ぶ。
  `ntuple.indices` は各パターンで 8 対称変換の最小を取るため、評価は盤面の回転・鏡像で不変（テストで固定）。
  n-tuple は「角・アンカー」の好みを原理的に表現できない。値の順は spiral 0.196 > scrambled 0.044 > flex −0.005 で、
  flex 形状は最下位。
- 手作り snake との一致/不一致: 手作り snake は snake 経路に沿って降順の盤（snake / spiral、どちらも順序違反 0 で
  空きマス項だけの 60.0）を高く評価し、経路が回転・変形した盤（column_snake −372、anchor 7 −415、flex −501）を
  低く評価する。NN も flex・column_snake を負値にする点は同じだが、block2（snake は 3 位、nn は最下位）と
  回転 snake（snake は 6 位、nn は 1 位）で真逆になる。n-tuple は向きを区別しないので、向きを除けば
  snake に近い順（経路順 4 盤が上位、flex が最下位）を付ける。つまり「snake の経路形状が最良」は
  手作り評価に固有の主張で、学習型評価はそれぞれ別の形を好む。

### 既知の限界

- プローブ盤面は 7 枚、タイル多重集合は 1 種類（10 タイル）。順位相関の標本としては小さい。
- 比較しているのは value であり、対局結果ではない。depth 2 の expectimax や実プレイでの強さとの対応は未検証。
- NN は推論時に 8 対称を平均しないため、回転で値が跳ねる。上の nn 列の差は「学習した選好」と
  「推論時の非対称性」が混ざっている。
- n-tuple は min-over-8-symmetries のため形状の向きを測れない。向きの選好を見たい場合は別の評価器が要る。

## 採用: monotonic (#117)

`ai/solver/solver.zig` の出荷時重み `baseline` に `.monotonic = 32` を入れた（#114 の ablation で depth 2 +37%、#117 で depth 3 +42%）。
`monotonic = 0` が従来の挙動。

holdout (5×40, seed 7000, depth 2, ReleaseFast) を採用後に取り直した結果:

| variant | avg score | avg moves | best | games/s |
|---|---|---|---|---|
| `monotonic=0`（従来） | 25230.3 | 1343.5 | 4096 | 22.28 |
| `baseline`（採用後, monotonic=32） | 34052.5 | 1742.0 | 4096 | 18.25 |
| `monotonic=24` | 33341.2 | 1701.3 | 4096 | 17.87 |

- 従来比 **+35.0%**。#114 の測定（24757.7 → 34116.9）と一致する。
- コストは盤面走査 1 回分で **games/s が −18%**（22.3 → 18.3、depth 2）。
- 24 と 32 の差は小さい（+2.1%）が、depth 2/3 の両方で 32 が上なので 32 を採った。
- 戻すときは `baseline` の `.monotonic = 32` を消す（0 が既定）。

副作用として、`greedy` ポリシーの一手も変わる（評価関数を共有しているため）。`order_penalty` 系は据え置き。
