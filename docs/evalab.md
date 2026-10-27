# evalab — 評価関数バリアントの比較

`ai/solver/evalab.zig` は同じ seed で複数の評価重みを対局させ、avg score / moves / best / 速度を並べる。
`Weights` の既定値は出荷時の評価と同じなので、`baseline` は従来の expectimax そのもの。

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
