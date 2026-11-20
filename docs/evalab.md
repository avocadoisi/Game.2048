# evalab — 評価関数バリアントの比較

`ai/solver/evalab.zig` は同じ seed で複数の評価重みを対局させ、avg score / moves / best / 速度を並べる。
`Weights` の既定値は出荷時の評価と同じなので、`baseline` は従来の expectimax そのもの。
出荷時重みには `monotonic = 32` が入っている（[採用: monotonic](#採用-monotonic-117)）。それより前の測定の
`baseline` は `monotonic = 0` の意味なので読み替える。

```sh
zig build evalab -Doptimize=ReleaseFast -- "<specs>" <runs> <games_per_run> <depth> <seed>
# 例: zig build evalab -Doptimize=ReleaseFast -- "baseline;monotonic=24" 10 40 2 7000
```

spec は `;` 区切り、各 spec は `,` 区切りの `key=value`。key は `layout` `anchor` `ordered` `empty` `order`
`order_late` `merge` `forced_down` `smoothness` `monotonic` `tile_sum`、preset は `baseline` `ga`。

| ノブ | 意味 | 既定 |
| --- | --- | --- |
| `layout` | 順序ペナルティの経路形状（`snake` `column_snake` `spiral` `block2` `flex` `flex_reverse`） | `snake` |
| `anchor` | 経路の向き（0..7 = 90°CW 回転 ×4 × 列ミラー）。0 が出荷時 | `0` |
| `ordered` | 経路の先頭 N マスだけ順序を課し、残りは自由マスにする | `16` |
| `smoothness` | 隣接タイルの \|log2 差\| 和のペナルティ | `0` |
| `monotonic` | 各行・各列の単調性違反量のペナルティ | `32` |
| `tile_sum` | 盤面の log2 合計ボーナス | `0` |

すべて 0（または既定値）のとき出荷時の評価と一致する（テストで固定）。`ordered=16` は全マスに順序を課す＝従来どおり。

## 経路形状・向き・順序マス数（#112、#113）

depth 2・5 runs × 40 局・seed 4000。

| spec | avg score | avg moves | best | games/s |
| --- | ---: | ---: | ---: | ---: |
| `snake`（baseline） | 24,345.3 | 1,311.1 | 4096 | 27.4 |
| `column_snake` | 24,869.1 | 1,320.5 | 4096 | 27.6 |
| `spiral` | 20,591.1 | 1,135.3 | 4096 | 31.0 |
| `snake, ordered=12` | 20,059.2 | 1,112.0 | 4096 | — |
| `snake, ordered=10` | 9,100.4 | 578.1 | 2048 | — |
| `snake, ordered=8` | 11,357.3 | 700.1 | 2048 | — |
| `flex, ordered=7` | 4,460.0 | 334.4 | 1024 | — |
| `flex_reverse, ordered=7` | 9,184.6 | 584.7 | 2048 | — |

- 行スネークと列スネークは同等（差は seed ノイズ）、spiral は −15%、`block2` は −44%（13,605.7 / 802.2 / 2048）。
- 順序を課すマスを減らすほど単調に悪化し、自由マスを残すと 1/5 以下になる。flex の狙い（身動きを良くする）は
  depth 2 では成立しない。
- `anchor` 0..7 はすべて 24,345–24,946 に収まり差は ±1.2%。盤面評価は向きに依存しない（対称性どおり）ので、
  `forced_down` を回転させても利得は出ない。

## smoothness / monotonic / tile_sum（#114）

seed 4000・5×40 の一次判定: `smoothness=1` 25,303.5 / `=3` 24,700.8、`monotonic=1` 24,335.4 / `=3` 30,097.1、
`tile_sum=0.5` 26,077.7 / `=2` 24,049.8、`smoothness=2 + monotonic=2` 25,881.4（baseline 24,345.3）。
**monotonic だけが効く**。重みの掃引（seed 6000・5×40）:

| monotonic | 3 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avg score | 26,993 | 28,740 | 31,955 | 32,110 | 32,266 | 33,402 | 34,674 | 32,375 | 29,554 | 26,665 |

24–32 に山があり、32 では 8192 到達局が出た（他は 4096）。holdout（10×40、seed 7000、depth 2）:

| spec | avg score | avg moves | best |
| --- | ---: | ---: | ---: |
| baseline | 24,757.7 | 1,322.3 | 4096 |
| `monotonic=24` | 33,896.0 | 1,733.4 | 4096 |
| `monotonic=32` | 34,116.9 | 1,749.1 | **8192** |
| `monotonic=32, order=0` | 30,870.2 | 1,605.4 | 4096 |
| `monotonic=24, tile_sum=0.5` | 33,650.8 | 1,727.5 | 4096 |

+37%（平均手数 +32%）。`order=0` にすると −9.5% なのでスネーク順ペナルティは monotonic と併用しても要る。
tile_sum の上乗せは誤差。depth 3（seed 7000、2×8 局）では baseline 40,692.3 → `monotonic=32` 44,134.8 で
利得は +8.5% に縮むが、8 局ずつなので有意とは言わない。

## 採用: monotonic 32（#117）

`solver.zig` の出荷時重み `baseline` に `.monotonic = 32` を入れた。depth 2 で +37%（#114）、depth 3 で +42%

holdout（5×40、seed 7000、depth 2、採用後に取り直し）:

| variant | avg score | avg moves | best | games/s |
| --- | ---: | ---: | ---: | ---: |
| `monotonic=0`（従来） | 25,230.3 | 1,343.5 | 4096 | 22.28 |
| `baseline`（monotonic=32） | 34,052.5 | 1,742.0 | 4096 | 18.25 |
| `monotonic=24` | 33,341.2 | 1,701.3 | 4096 | 17.87 |

従来比 **+35.0%**、コストは盤面走査 1 回分で **games/s −18%**（depth 2）。24 と 32 の差は小さい（+2.1%）が
depth 2/3 の両方で 32 が上なので 32 を採った。戻すときは `baseline` の `.monotonic` を消す（0 が既定）。
評価関数を共有する `greedy` の一手も変わる。`order_penalty` 系は据え置き。

## 学習型評価の選好（#115）

手作りの snake 評価と、学習型の値関数（NN / n-tuple）が同じ盤面形状を好むかを固定プローブ盤面で測る
（`zig build probe`、差し替えは `zig build probe -- ai/solver/ntuple_weights.bin`）。全プローブ盤面は同じタイル
多重集合 `{2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4}`（残り 6 マスが空）で、置き方だけを変えている。

```
board                     snake  rank             nn  rank         ntuple  rank
snake                   60.0000   1.5         0.9222     3         0.7336   2.5
column_snake          -371.7400     5        -3.3625     6         0.7336   2.5
spiral                  60.0000   1.5         1.5035     2         0.1955     5
flex                  -501.2620     7        -1.6711     5        -0.0048     7
block2                -155.8700     3        -4.3314     7         0.7336   2.5
snake anchor 7        -414.9140     6         9.0475     1         0.7336   2.5
scrambled             -328.5660     4        -1.4902     4         0.0445     6

spearman rank correlation
  snake  vs nn       0.126
  snake  vs ntuple   0.288
  nn     vs ntuple  -0.020
```

- 順位相関は 0.13 / 0.29 / −0.02 でいずれも 1.0 から遠い。7 枚では Spearman の標準誤差が約 0.4 あるので
  言えるのは「一致していない」ところまで。
- `nn` は 180° 回転した snake 盤（9.05）と spiral（1.50）を左上 snake 盤（0.92）より高く評価し、flex・
  column_snake・block2 を負値にする。手作り snake が 3 位に置く block2 を nn は最下位にする。
- `ntuple` は経路順に並んだ 4 盤が同値 0.7336 で並ぶ。`ntuple.indices` が各パターンで 8 対称変換の最小を取る
  ため評価は回転・鏡像で不変（テストで固定）で、角・アンカーの好みは原理的に表現できない。
- つまり「snake の経路形状が最良」は手作り評価に固有の主張で、学習型評価はそれぞれ別の形を好む。

## CMA-ES で重みを最適化（#105）

`ai/solver/cmaes.zig` が汎用の簡易 CMA-ES、`ai/solver/tune.zig` が `Weights` への適応。

```sh
zig build tune -Doptimize=ReleaseFast -- --gens 20 --pop 6 --games 15 --depth 2
# best weights: order=60.515,order_late=17.706,merge=20.342,monotonic=45.042
```

CSA（累積ステップサイズ適応）+ 共分散の rank-one 更新のみ（rank-mu なし）、共分散は Cholesky 分解して
サンプリングする（固有分解なし）。探索範囲外のサンプルはクリップして評価し、seed 固定で決定的（テストで固定）。
探索次元は `order_penalty` `order_penalty_late` `merge` `monotonic` の 4 本だけで、他は出荷時重みで固定する
（`--dims` で部分集合を選べる。選ばない次元は探索範囲を `[baseline, baseline]` に潰す）。範囲は
`order_penalty` / `order_penalty_late` 0..120、`merge` 0..40、`monotonic` 0..96。目的関数は `bench.runW` の
avg score で、1 評価 = 15 局（seed 8000..8014、depth 2）、初期点は出荷時重み・初期 sigma 8、時間予算 600 秒
（この設定の実測 100 秒 = 120 評価 1,800 局、8 コア）。

| 集合 | variant | avg score | avg moves | best | games/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 訓練（seed 8000..8014、15 局） | `baseline` | 25,550.7 | 1,373.3 | 2048 | 22.6 |
| 訓練 | CMA-ES の最良 | 44,696.3 | 2,203.9 | 4096 | 17.7 |
| holdout（seed 9000、5×40） | `baseline` | 32,380.8 | 1,670.5 | 8192 | 18.6 |
| holdout | CMA-ES の最良 | 33,656.1 | 1,730.0 | 4096 | 15.1 |

holdout は `zig build evalab -Doptimize=ReleaseFast -- "baseline;order=60.515,order_late=17.706,merge=20.342,monotonic=45.042" 5 40 2 9000`
で再現できる（`order_late` キーはこのために追加した）。

- 訓練 +74.9%、holdout **+3.9%**。holdout 200 局でも同じ `baseline` は seed 7000 で 34,052.5 と 5% 動くので、
  +3.9% は seed 間ばらつきと同じ桁で改善とは言えない。GA（24,046）と比べても同条件 `baseline` 32,381 を
  僅かに上回るだけ。**出荷時重みは差し替えていない**。
- 訓練と holdout の差（+75% vs +4%）は過学習。1 評価 15 局の平均は標準誤差が十数%あり、CMA-ES は
  「その 15 局でたまたま良かった重み」を選ぶ。動いた向きは手動スイープと整合する（`monotonic` 32 → 45.0、
  `merge` 10 → 20.3 に上げ、`order_penalty_late` 25 → 17.7 に下げる）。
- 出力 spec は 3 桁丸めで、`monotonic` を 45.042 → 45.043 に変えるだけで 44,696 → 47,603 と動く（ゲームが
  混沌的で重みの微小変化が手の選択を反転させる）。訓練の数字は 3 桁 spec を測り直した値で、探索内部の
  最良値 46,728.5 とは一致しない。
- 次に潰すのは目的関数の分散（100 局 or 複数 seed 平均）、訓練/holdout の seed を増やす、次元を増やす
  （rank-mu 更新と固有分解が要る）、depth 3 での最適性。

## 反復深化 + 確率枝刈り（#104）

`Policy.expectimax_id`。chance node で経路確率が `prob_cutoff = 1e-4` 未満の枝を落として残りの確率和で
再正規化し（値のスケールを保つ）、`depth <= 1` の葉に近い層では切らない。`bestMoveID(state, budget_ms)` は
depth 2 から 1 ずつ深くし、予算内に**完了した**最後の深さの最善手を返す（`id_max_depth = 8` で頭打ち）。
`depth` 引数はこのポリシーでは時間予算（ms）として渡す。枝刈りで同じ盤面・深さの値が変わるので TT の種別を
`max_pruned` / `chance_pruned` に分け、`.max` / `.chance` と混ぜない。

5 runs × 10 局、seed 7000、ReleaseFast:

| policy | 予算 | avg score | avg moves | best | games/s | 到達深さ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `expectimax_tt`（depth 3） | — | 51,814.2 | 2,488.3 | 8192 | 1.19 | 3 |
| `expectimax_id` | 1 ms/move | 69,382.6 | 3,189.3 | 8192 | 0.17 | 8（上限） |
| `expectimax_id` | 2 ms/move | 76,811.5 | 3,492.6 | 8192 | 0.07 | 8（上限） |

+34%（1 ms）/ +48%（2 ms）で到達深さは上限 8 に張り付く。代償は時間で、1 ms 予算でも 5.8 s/game（depth 3 は
0.84 s/game）。予算は深さの切れ目でしか見ないので 1 手あたり最大 1 回ぶん超過する。予算が実時間なので同じ
seed でもマシン負荷で手が変わり（1 ms の 2 回の測定で 71,008.2 と 69,382.6）、固定 seed の再現性を保つため
`ranking` の既定ポリシーには入れていない。6 s/game に収めるなら 1 ms が上限で、速いマシンほど強くなる。

## 3-ply + 確率枝刈り（#108）

`nn.Mode.deep3` は `deep` の 1 手先（自手→出現→自手→出現→自手→出現→NN 葉）。素の 3-ply は 2-ply の約 25 倍
かかるので、`Search.cutoff`（経路確率 `cprob * spawn.p / 空きマス` が `deep3_cutoff = 1e-2` 未満の spawn を
落として残りの確率和で再正規化。`solver.zig` の `chanceValueTT` の `cut = true` と同じ式）を足した。空きマス
11 以上では第 1 層の 4 出現とその下の第 2 層が落ち、葉の直前の層は全幅のまま。全枝が切れたノードは
afterstate の葉値で代用する。`cutoff = 0` が枝刈り無効で、`.deep`（= `nn2`）はこの経路を通らない（golden の
checksum は不変）。`solver.Policy.nn3` として `zig build solve` / `/api/solve?algo=nn3` / wasm から選べるが、
1 局 2.5 分なので `ranking` の既定リストには入れていない。

5 runs × 2 局 = 10 局、seed 7000、ReleaseFast:

| policy | avg score | avg moves | best | games/s | ms/move |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nn2`（2-ply） | 70,256.0 | 3,198.3 | 4096 | 0.67 | 1.9 |
| `nn3`（3-ply + 枝刈り） | 72,973.6 | 3,298.8 | 4096 | 0.02 | 46.3 |

+3.9%（手数 +3.1%、best tile は同じ）。10 局ではばらつきが数 %あるので有意ではない（seed 7000 の 1 局目だけ
なら nn2 60,480 → nn3 75,092）。代償は約 25 倍/手（30 倍/局）で、1e-2 の枝刈りを入れても 1 手 46 ms と WASM の
目安 20 ms を超える。時間予算に入れるなら反復深化 + ノード上限が要る（#104 と同じ形）。1e-2 は「落とす確率和
< 約 10%」を見込んだ粗い値で、第 1 層の 4 出現を落とすため 4 の位置を変える手を過小評価しうる。

## 既知の限界

- 評価は seed 4000–7000 の holdout、depth 2（一部 depth 3）の expectimax のみ。nn ポリシーでの順位は未検証。
- avg score / moves / best だけを見ており、到達率の分布と死因（盤面が詰まる形）は未記録。
- 速度列は同じ実行・同じマシンの相対値で、並列ビルドなど他の負荷が乗ると下がる。
- `zig build test` の ReleaseFast 実行は nn テストのコンパイルに 20 分以上かかる。単体テストは Debug（既定）か
  `-Dtest-filter=<名前の一部>` を使う。
- プローブ（#115）は盤面 7 枚・タイル多重集合 1 種類で、比較するのは value であって対局結果ではない。NN は
  推論時に 8 対称を平均しないため回転で値が跳ね、上の nn 列は「学習した選好」と「推論時の非対称性」が混ざる。
