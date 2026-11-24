# Solver history: what worked

記録は変種ごとに「何を試したか・決めた数字・判定」だけを残す。深さ 2 のスコアは
`zig build solve` の `runs=10 games/run=20 depth=2 seed=1` か、明記した評価コマンドの値。

## Evaluation: the timeline

1. **Corner bonus (expectimax)**: empty ×10 + corner bonus. depth 1 3,673; depth 2 7,606 (seed 1) / 7,866 (seed 2);
   depth 3 11,876 (20 games, 23 s).
2. **Row order + forced-down penalty**: order 1..16 row by row, largest tile bottom-right, −100 when only down is legal.
   depth 2 18,596 / 19,812.
3. **Largest tile at top-left**: 19,198 / 19,352, depth 1 3,343. The earlier 5,421 / 5,106 came from a bug
   (the first non-empty cell was compared against 0); fixed by starting at the first non-empty cell (`?f64`).
4. **Order penalty ×30** (empty 10, mean of seeds 1+2): ×3 10.5k, ×10 19.3k, ×30 **22.5k adopted**, ×50 20.3k, ×100 21.1k.
   Empty 5 → 20.9k, empty 30 → 20.0k, so 10 was kept.
5. **Snake order (adopted)**: row 0 L→R, row 1 R→L, …; weights empty 10 / order 30. depth 2 26,716 / 24,935.
6. **Merge bonus ×10 (adopted)**: `+10 × (adjacent equal non-empty pairs)`. Holdout 200 games: seed 100 26,232 vs 24,627,
   seed 300 25,078 vs 24,140. ×5 (25,428 / 23,983) inconsistent, ×15 24,753, ×30 18,954. Smoothness ×1/×3 25,015 / 24,866 (noise).
   The bonus counts pairs that have not merged yet, so a move that only forms a pair scores; the unit test
   `bestMove takes a merge over a move that does nothing` was rebuilt on a full board where only left/right are legal.

Rejected at this stage: **gap penalty** ×20 (empty cells between two tiles on the snake path) with **down cost** ×50 → 20,199;
down cost alone ×50 → 24,226, both under the snake baseline 25,827. The forced-down penalty already covers the extreme case;
the gap weight was never re-tuned (a smaller one is untested).

## Benchmark (`zig build solve`, runs=10, games/run=20, depth=2, seed=1, ReleaseFast)

| Variant | avg score | avg moves | best tile | games/s | elapsed |
|---|---|---|---|---|---|
| Row order, order ×30 | 20,744 | 1,129 | 4096 | 6.72 | 29.8 s |
| **Snake, order ×30 (adopted)** | **25,827** | **1,366** | **4096** | **5.04-5.14** | **38.9-39.7 s** |
| Snake + down cost 50 | 24,226 | 1,311 | 4096 | 5.12 | 39.1 s |
| Snake + gap 20 + down cost 50 | 20,199 | 1,119 | 4096 | 6.42 | 31.2 s |
| Snake + merge ×10 (holdout seed 100) | 26,232 | 1,398 | 4096 | 2.86 | — |

Weights were tuned on seeds 1-2 (100 games each) and this benchmark reuses seed 1, so the numbers are optimistic;
depth 1 / 3 were not measured for the snake order. The evaluation is hand-set in `ai/solver/solver.zig`; the n-tuple table replaced it.

## Policy ranking (`zig build ranking`)
Served at `/api/ranking` from `src/web/ranking.json`:
`zig build ranking -Doptimize=ReleaseFast -- 5 20 1000 > src/web/ranking.json` (seed 1000 is not used for tuning).
Result: expectimax (depth 2) 25,267 > greedy 3,424 > random 1,062.

## Parallel candidates (ranking seed 1000, 5 runs × 20 games)
- `expectimax_tt` (depth 3 + transposition table): 41,216, about 4× faster than depth 3 (3.4 vs 15.2 s/game).
- `mcts` (200 rollouts/move): 29,722 at 2.0 s/game.
- `expectimax` depth 2: 25,267.
- `beam` (width 32, 3 plies, spawn approximated as 2): 12,696 at 0.07 s/game.
- greedy 3,424; random 1,062.
Speed rule: no policy slower than depth 3 (6.8-15 s/game depending on load).

## Deep learning: small MLP value net (`zig build nn`)

- Model: `ai/solver/nn.zig`, one hidden layer MLP over afterstates, TD(0) in pure Zig (no dependencies).
- Training: 20,000 self-play games, lr 0.003. Weights in `ai/solver/nn_weights.bin` (tracked, embedded by `solver.zig`;
  16-byte header + little-endian f64, see `nn.encode` / `nn.parse`).
- Eval seeds 2000..2099 (100 games, `zig build nn -- 20000 100 2000 0.003`):

| Policy | avg score | avg moves | best tile | time (100 games) |
|---|---|---|---|---|
| NN 1-ply | 21,227 | 1,129 | 4096 | 0.6 s |
| NN + 1 spawn lookahead | 38,734 | 1,919 | 4096 | 36.2 s |
| expectimax depth 2 | 25,004 | 1,328 | 4096 | 19.7 s |

- lr sweep, 20 games on seed 1000 (noisy): 0.0003 → 5,571 / 14,493; 0.001 → 11,452 / 20,286; 0.003 → 18,234 / 34,645.
- One training run, one seed range; the lookahead is not a like-for-like depth comparison and costs about 2× the baseline.

## GA over evaluation weights (`zig build ga-rl`)

- Evolves the 4 snake weights (empty, order_penalty, merge, forced_down) with tournament selection, uniform crossover,
  clamped log-normal mutation: `16 15 8 2 1 500 100` (population 16, 15 generations, 8 games/fitness, depth 2, seed 1, holdout 500, 100 games).
- Holdout 500..599: baseline 24,067 → evolved 25,095 (+4%). Evolved weights: empty 7.028, order 43.174, merge 0, forced_down 0.
- merge and forced_down collapsing to 0 with one run and one holdout range means the +4% may be noise.

## Policies wired into the solver (`ga`, `nn`)

- `ga`: `solver.ga_weights` (3 decimals from `zig build ga-rl`), depth 2. `nn`: embedded `ai/solver/nn_weights.bin` (148 KB), chance-mode net.
- Selectable in `/api/solve?algo=ga|nn` and the UI dropdown; ranking.json regenerated with `zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`.
- Ranking seed 1000: expectimax_tt 41,216 / nn 36,885 / mcts 29,722 / expectimax 25,267 / ga 24,046 / beam 12,696 / greedy 3,424 / random 1,062.
  The GA weights lose to plain expectimax here; the +4% held only on seeds 500-599.

## NN: longer training (200,000 games, lr 0.003 decaying to 10%)

- `zig build nn -- 200000 100 3000 0.003`: 3,316 s, same 64/32 MLP, 148 KB, checkpoint every 5,000 games, linear decay added.
- Eval seeds 3000..3099: NN 1-ply 35,687 (4096); **NN + 1 spawn lookahead 66,600 (8192, 73.8 s)**; expectimax depth 2 24,839 (37.9 s). Previous: 38,734.
- Ranking seed 1000 (5×20): nn 64,113 (was 36,885), above expectimax_tt 41,216, at about 2× expectimax depth 2 per game (0.7 s/game), still faster than depth 3.
- Not measured: wider net 128/64 (slower to train), TD(λ), n-tuple tables.

## Speed work (no change in play)

Every row gives identical moves for the same seed (`zig build solve -- 2 20 2 1`: 22135.1, 1219.0).

| Change | games/s (depth 2, 40 games) |
|---|---|
| Baseline | 5.4 |
| `evaluate`: `@log2(value)` → `@ctz(value)` (tiles are powers of 2) | 7.3 |
| `core.mergeLine`: 65536-entry row table, general path for non-powers-of-2 or ≥ 65536 | 8.3 |

Dropped: a precomputed slide index table (`slide_lines`) in `core.slide` (about 27 ns/slide either way).
The row table costs about 512 KB in the binary and 15-20 s of ReleaseFast compile time. The remaining cost is
`evaluateState` (up to 4 slides per call); a bitboard core would be the next step but changes `Board` everywhere.

## Tried and not merged: packed nibble search (branch `exp/packed-search`)

Rows packed as 4 nibbles (log2 per tile) make a slide one table lookup per row: about 3 ns vs about 28 ns for `GameState.slide`.
`core.Packed` with expectimax on packed boards is bit-identical (`zig build solve -- 2 20 2 1` → 22135.1) but runs 5.5-5.9 games/s,
the same as the merged version: the profile moved to `evaluateStateP` and `Packed.slide`. A pack/unpack wrapper around
`GameState.slide` was slower (about 40 ns). A real speedup needs `GameState` stored packed, which touches about 78 `.board`
uses and needs a separate path for tiles ≥ 65536. Not done.

## Training focus: near-loss states (proposal)

`nn.trainGame` runs TD(0) over every move of every game with the same weight, so boards close to game over — the ones that
decide the score — are rare examples. Candidates, cheapest first:
1. **Diagnose first**: log per-move `|target − V(s)|` by empty count (0..3, 4..7, 8+) and by distance to game over (last 10, 11..30, earlier).
2. **Backward sweep per game**: keep the (afterstate, reward) pairs and train last → first so the terminal value propagates in one pass.
3. **Replay the last K moves** of each lost game (K ≈ 30) in a second pass with a smaller lr.
4. **Prioritized replay** by `|δ|^α` (α ≈ 0.5) with a uniform share (e.g. 50%) so the distribution does not drift.
5. **Restarts from hard boards** with few empty cells (risky: they may be unreachable under the policy, so keep the share small).

Safety rules for all of them: focusing on losing states is off-policy, so keep the uniform per-move update; compare at equal
**moves processed**, not equal games; evaluate with `zig build nn -- 20000 100 2000 0.003`, then a fresh range (3000..3099);
record every run here. All three cheap candidates were built and measured (#32, #33, #34) — conclusion below.

## Diagnostic run (#32)

`zig build nn -Doptimize=ReleaseFast -- 2000 1 2000 0.003 diag` (2,000 games, seed 1; weights restored afterwards).

| empty | moves | steps | mean \|err\| | mean drop | share err |
|---|---|---|---|---|---|
| 0..3 | last 10 | 17979 | 0.2966 | -0.2193 | 7.9% |
| 0..3 | 11..30 | 20110 | 0.1582 | -0.0045 | 4.7% |
| 0..3 | earlier | 62048 | 0.1669 | 0.0224 | 15.3% |
| 4..7 | last 10 | 2021 | 0.1632 | -0.1198 | 0.5% |
| 4..7 | 11..30 | 19620 | 0.1165 | -0.0388 | 3.4% |
| 4..7 | earlier | 355867 | 0.1002 | -0.0061 | 52.6% |
| 8+ | earlier | 150594 | 0.0702 | -0.0136 | 15.6% |

The last 10 moves carry 3-4× the per-step error of the early game (0.2966 vs 0.0702-0.1002) and the value drops about −0.22 there,
but they are only about 8% of total error; 53% sits in the 4..7 empty, earlier bucket. So near-loss states are badly fit per step
but too few to drive the fit: mix them in instead of replacing the uniform update. One run, one seed, 2,000 games.

## Eval variance (#37)

`zig build nn -- 0 20 1000 0.003 5` (eval only, committed weights, seeds 1000..1099 as 5 disjoint ranges of 20 games;
mean ± stderr over the 5 range means, stderr = sd/√5).

| policy | avg score | avg moves | best tile |
|---|---|---|---|
| nn 1-ply | 18851 ± 503 | 1028.1 | 2048 |
| nn + 1 spawn lookahead | 38572 ± 1608 | 1917.1 | 4096 |
| expectimax depth 2 | 24776 ± 1267 | 1319.8 | 4096 |

nn 1-ply vs expectimax: +5,925 with stderr ≈ 1,363 (4.3σ, p < 0.01); lookahead vs expectimax: +13,796, stderr ≈ 2,070 (6.7σ).
The independent-range stderr is conservative because policies share seeds. 100 games per policy total is still thin:
a single 20-game range gives no stderr at all, so policy comparisons need `seed_ranges` ≥ 5. Wall time: lookahead 394 s, 1-ply 5 s per 100 games.

## Fair comparison (#38)

Same command as #37; score/s = total score / CPU seconds for 100 games.

| policy | avg score | avg moves | elapsed | score/s |
|---|---|---|---|---|
| nn 1-ply | 18851 ± 503 | 1028 | 0.6 s | 2,957,818 |
| nn + spawn lookahead | 38572 ± 1608 | 1917 | 42.2 s | 91,306 |
| expectimax depth 2 | 24776 ± 1267 | 1320 | 18.2 s | 136,202 |

The lookahead wins per game (+13.8k) but loses about 33% per CPU second (91k vs 136k), so the gain does not hold at equal time.

## Parallel eval (#40)

`ai/solver/parallel.zig` runs eval games on `std.Thread.spawn` threads, `min(cpu count, 8)`; each game keeps its own PRNG seeded
by its seed, so results do not depend on thread count. `ranking` parallelizes per run (one PRNG per run, games in order inside a run).
Machine: 8 logical CPUs.

| policy | serial | parallel | speedup |
|---|---|---|---|
| nn 1-ply | 0.6 s | 0.1 s | 6× (too short to measure) |
| nn + spawn lookahead | 44.2 s | 9.9 s | 4.5× |
| expectimax depth 2 | 13.1 s | 2.7 s | 4.9× |
| sum | 57.9 s | 12.7 s | 4.6× |

Whole command 58.1 s → 29.4 s (includes build); `zig build ranking -- 5 20 1000` 7m00s → 2m17s. Results are unchanged:
18851.1 ± 503.4 / 1028.1 / 2048, 38572.0 ± 1607.6 / 1917.1 / 4096, 24775.9 ± 1266.8 / 1319.8 / 4096, and the ranking JSON is
byte-identical (`cmp`). Tests: `parallel.map` vs a serial loop, nn games vs a serial loop over 8 seeds, `bench.run` vs a serial loop.
Caveat: `score/s` in `zig build nn` is now per **wall** second, not per CPU second.

## Near-loss focus: conclusion (#28)

Closed after #32 (diagnostic), #33 (replay / backward sweep, PR #48), #34 (priority replay, PR #50).

- At equal training budget (20,000 games, training seed 1; eval on 2000 and fresh 3000..6000, 20 games each):
  base **18,677 ± 717**, replay K=30 **14,973 ± 838**, priority (m=32, α 0.5, 50% uniform) **19,488 ± 1,490**, backward **2,116 ± 80**.
- Nothing beats the uniform per-move TD(0) update. Priority is closest (+811 against a combined stderr ≈ 1,650, 0.5σ) at equal
  moves (14.02M vs 13.97M, +0.3%); replay is 3.4σ worse; the backward sweep collapses. The 8% error share explains it:
  oversampling near-loss states takes fit away from the mid-game states holding half the error.
- Candidate 5 (restarts from hard boards) was never implemented: same 8% argument plus the risk of training on states the policy never reaches.
- Kept: the `diag` flag in `nn_main.zig`; replay/backward/priority stay off by default with the default path pinned by the
  recorded-checksum test ("default trainGame weights match the recorded checksum (5 games, seeds 11/12)").
- Not done: replay at equal moves (it reached 92% of base's moves), K/m/α sweeps, a second training seed.
## Training parallelization (#35)

`trainGame` in parallel batches (opt-in `par`): `N = min(cpu count, 8)` games per batch, game `i` seeded `train_seed + i`,
each training its own copy of the batch-start net; the next net is `net_start + (1/N) * Σ_i (net_after_i - net_start)`
summed in game index order, so weights do not depend on thread count. Serial path unchanged; `diag` merges per-game tables.
(#99 extends batching to the stateless variants — see `par for stateless variants` at the end.)

- `zig build nn -Doptimize=ReleaseFast -- 2000 1 2000 0.003` vs the same plus `1 par`, 8 CPUs, batches of 8, 250 batches:
  9.1 s → 1.6 s (5.7×). The parallel run applies the batch-start lr to the whole batch, so the schedule differs slightly.
- Reproducibility: the test `parallel training batch matches a serial run of the same game seeds` compares a 4-game batched run
  byte-for-byte with a serial loop over seeds 77..80; two parallel CLI runs write byte-identical files.
- The batch average is not "apply 8 games one after another"; it is the aggregation chosen for determinism, and `par` is opt-in,
  so every recorded result above keeps the serial behaviour.

## Replay and backward sweep (#33)

`ai/solver/nn.zig` (`trainGameOpts`, `Options`), default off, default path pinned by the recorded-checksum test.
- `backward`: no per-move update while playing; afterwards every stored afterstate is trained last → first, target `reward + V(next)` recomputed.
- `replay`: normal per-move updates, then the last K = 30 afterstates are re-trained once, oldest first, at 0.5 × lr with fresh targets.
- CLI: `zig build nn -- <train> <eval> <seed> <lr> <-|diag> <base|backward|replay> <K> <weights_path>`; variant runs write to a temp path.

20,000 games, training seed 1; eval on ranges 2000, 3000, 4000, 5000, 6000 (20 games each; 1,000-apart ranges run individually, stderr by hand):

| variant | train moves | nn 1-ply | nn + 1 spawn |
|---|---|---|---|
| base | 13,969,906 | 18,677 ± 717 | 39,715 ± 1,103 |
| backward | 6,272,449 | 2,116 ± 80 | 2,116 ± 80 (no lookahead gain) |
| replay K=30 | 12,811,345 | 14,973 ± 838 | 33,241 ± 740 |

Per-range (1-ply): base 19,110 / 16,668 / 19,592 / 17,423 / 20,589; backward 2,288 / 2,101 / 1,821 / 2,196 / 2,176;
replay 16,664 / 13,497 / 12,655 / 16,871 / 15,177. expectimax depth 2 on seeds 2000..2019: 21,840 in every run.

Neither candidate beats base. Replay is about −20% (3,704 against combined stderr 1,103, 3.4σ); backward collapses to ~2,100
(the policy dies after ~200 moves). Un-tuned: only lr 0.003, only K = 30, no backward lr sweep; replay reached 92% of base's moves,
so the comparison is at equal games, not equal moves.

## Priority replay (#34)

`Options.priority` (CLI variant `priority`), default off. Per-move TD(0) updates unchanged; every update is also stored in a
1024-entry ring buffer (afterstate, reward, next afterstate, terminal flag, |TD error|) that persists across games. After each
game, `m = 32` extra updates draw an entry uniform with p = 0.5, else with p ∝ `(|δ| + 1e-3)^0.5`, target recomputed, the drawn
entry's |δ| refreshed. Tests: sampling share against the analytic mixture (tolerance 0.01), priority pass changes the weights.

Same protocol as #33:

| variant | train moves | nn 1-ply |
|---|---|---|
| base | 13,969,906 | 18,677 ± 717 |
| priority (m=32, α=0.5, 50% uniform) | 14,016,958 | 19,488 ± 1,490 |

Per-range: base as above; priority 18,278 / 14,513 / 23,450 / 20,639 / 20,559. +811 against a combined stderr ≈ 1,650 (0.5σ),
with a much wider spread (13.7k range between ranges). +0.3% moves, so game play barely changed. One training seed, m and α unswept.

## Priority replay across training seeds (#56)

The training seed became CLI argument 9 (after `weights_path`, default 1), so #34's +811 could be re-run per seed.

`zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - <variant> 30 /tmp/nnseed/<variant>_s<s>.bin <s>` for seeds 1..5,
each net evaluated on ranges 2000/3000/4000/5000/6000 (20 games each), nn 1-ply. Variants stay on the serial path (`par` trains only plain per-move).

| training seed | base | priority | delta |
|---|---|---|---|
| 1 | 19,611 | 17,413 | -2,197 |
| 2 | 18,011 | 17,107 | -905 |
| 3 | 17,870 | 22,388 | +4,518 |
| 4 | 20,431 | 19,528 | -903 |
| 5 | 17,846 | 19,957 | +2,111 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **19,279 ± 959** | **+525 ± 1,224** (t = 0.4) |

Priority wins 2 of 5 seeds, so the earlier +811 was a single-seed artefact. Training-seed spread (base 17.8k-20.4k, priority 17.1k-22.4k)
is about 1.5× the between-variant difference, which is why one seed could not decide. Base seed 1 reproduces #34 (19,611 vs 18,677).
Cost: 10 runs × 20,000 games, 300-410 s each. `par` reaches 2,000 games in 1.5 s against 8.6 s serial but cannot run variant paths;
it now accepts `weights_path`, so a parallel run need not overwrite the committed `ai/solver/nn_weights.bin`.

## Mid-game focus (#59)

#32 localised 52.6% of the TD error in the "4..7 empty cells, earlier move" bucket against 7.9% in the last 10 moves, so the extra
post-game updates were drawn from that bucket instead of the near-loss one.

`Options.midgame` (CLI variant `midgame`): per-move TD(0) untouched, same 1024-entry ring buffer, `m = 32` extra updates drawn
**uniformly from entries with 4..7 empty cells** (whole buffer while empty), target recomputed. Default off; default path pinned by
the checksum test (`md5 0a8e6bf01e256adf2fc481317c5072c9` at 2000 games before and after).

Protocol as #56 (5 training seeds, ranges 2000..6000 × 20 games, nn 1-ply; base re-evaluated with the same binary):

| training seed | base | midgame | delta |
|---|---|---|---|
| 1 | 19,611 | 13,311 | -6,300 |
| 2 | 18,011 | 21,087 | +3,076 |
| 3 | 17,870 | 18,056 | +186 |
| 4 | 20,431 | 15,700 | -4,731 |
| 5 | 17,846 | 17,488 | -358 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **17,128 ± 1,290** | **-1,625 ± 1,710** (t = -1.0) |

No gain and a much wider spread (13.3k-21.1k); two seeds lose 6.3k and 4.7k, the same failure as `replay`: replacing the uniform
sample with a narrow bucket takes fit from the rest of the distribution. So #32 says where the error is, not that concentrating
updates there helps — `replay`, `priority` and `midgame` all point the same way. Untried, in order of expected value: mid-game bucket
mixed with a uniform share, more training games at the same protocol, an lr sweep. Cost: 5 runs, 145-358 s each.

## Training games: 20,000 vs 40,000 (#61)

Protocol as #56, no code change, only `train_games` 20,000 → 40,000 (linear lr decay is a function of the fraction trained, so the
schedule also stretches):

| training seed | 20k | 40k | delta |
|---|---|---|---|
| 1 | 19,611 | 30,770 | +11,160 |
| 2 | 18,011 | 30,283 | +12,272 |
| 3 | 17,870 | 33,257 | +15,387 |
| 4 | 20,431 | 33,314 | +12,883 |
| 5 | 17,846 | 25,906 | +8,060 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **30,706 ± 1,352** | **+11,952 ± 1,195** (t = 10.0) |

+12,000 for doubling the games, far above seed noise, and the curve has not flattened at 40,000. The committed weights were not
replaced: they are 200,000 games and score 40,282 on these same ranges (38,641 / 34,894 / 34,196 / 50,210 / 43,467). The point:
the #33/#34/#56/#59 protocol uses 20,000-game nets, so all those variant comparisons sit far down the curve. Cost: 5 × 40,000 games, 514-559 s each.

## Mid-game mixed replay (#63)

#59 lost the uniform share; #34 (50% uniform + 50% |TD error| weighted) came closest to base — so the two were combined.

`Options.midmix` (CLI variant `midmix`): per-move TD(0) untouched; after each game `m = 32` extra updates, each either uniform over the
whole buffer (p = 0.5) or `(|δ| + 1e-3)^0.5` weighted **among 4..7-empty entries**, target recomputed. Default off, default path
byte-identical (`md5 0a8e6bf01e256adf2fc481317c5072c9` at 2000 games).

Protocol as #56/#59:

| training seed | base | midmix | delta |
|---|---|---|---|
| 1 | 19,611 | 20,775 | +1,164 |
| 2 | 18,011 | 21,868 | +3,857 |
| 3 | 17,870 | 21,637 | +3,767 |
| 4 | 20,431 | 14,644 | -5,787 |
| 5 | 17,846 | 17,883 | +37 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **19,362 ± 1,377** | **+608 ± 1,762** (t = 0.3) |

4 of 5 seeds positive and the best mid-game variant so far, but the spread (14.6k-21.9k) is three times the effect. It is however the
first variant not worse on average, which fits the #32/#34 story: keep the uniform share and mix the bucket in rather than replacing it.
Stays opt-in, committed weights unchanged. Cost: 5 runs × 20,000 games, 194-353 s.

## Model update: fine-tuning the committed weights (#66)

`train_games > 0` always started from random init, so the only way forward was 200,000 games from scratch (#62). CLI variant literal
`resume` (positional 6) loads `weights_path` and trains `train_games` more on the plain per-move path:
`zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.001 - resume 30 /tmp/x.bin 1`. Default path unchanged (checksum test).

In-protocol ranges 2000..6000 vs held-out 7000..10000 (never used to choose a candidate), nn 1-ply, 20 games per range:

| candidate (from committed) | in-protocol | held-out | vs committed (held-out) |
|---|---|---|---|
| committed (200k, seed 1) | 40,282 | 37,233 | — |
| +20k, lr 1e-3, seed 1 | 42,856 | 43,224 | +5,990 |
| +20k, lr 3e-4, seed 1 | 42,544 | 42,577 | +5,344 |
| +20k lr 1e-3 then +20k lr 3e-4, seed 1 | 43,187 | 45,342 | +8,109 |
| +20k lr 1e-3 then +50k lr 3e-4, seed 1 | 45,118 | 42,089 | +4,856 |
| +50k, lr 1e-3, seed 1 | 37,894 | 40,611 | +3,378 |
| +20k, lr 1e-3, training seed 2 | 33,847 | 38,663 | +1,430 |
| +20k, lr 3e-4, training seed 2 | 41,157 | 37,576 | +343 |

Chosen: **+20k lr 1e-3 then +20k lr 3e-4** (40,000 extra games, 1,028 s + 628 s). It wins in-protocol and held-out, and "high lr
first, low lr to settle" beats the same budget at one lr. Fresh ranges **11000..15000 at 40 games each**: committed 38,402 → 42,609
(+4,207, +11%), ahead on 4 of 5 and tied on the fifth. Canonical bench: nn 64,112.9 → **65,409.9** (+1,297, best tile 4096 → 8192),
all other policies unchanged. New weights `md5 e501bae176f5cf2ff11a25c6f99e75a0` (old `6c9a164b5e05519ca2298c26e30431c4`).
Fine-tuning cannot pass what the 200k run already reached, so the gain is bounded; the longer schedules were only tried at training seed 1.

## Fine-tune extra rounds: overfit past the #66 peak (#97)

Whether more `resume` rounds on the current committed weights keep paying. Protocol as #66 (training seed 1, in-protocol 2000..6000,
held-out 7000..10000, 20 games per range):

| candidate (from committed) | in-protocol | held-out | vs committed (held-out) |
|---|---|---|---|
| committed (200k + 40k) | 43,187 | 45,342 | — |
| +20k, lr 3e-4 (a2) | 43,083 | 41,936 | -3,406 |
| +20k lr 1e-3 then +20k lr 3e-4 (a1r2) | 42,974 | 41,290 | -4,052 |

Held-out is the decider (#66's rule): both are flat in-protocol (~200 against per-range noise of several thousand) and lose 3.4k-4.1k
on ranges that never select a candidate. The recipe that won +8,109 once loses -4,052 the second time. Not adopted, weights unchanged;
more of the same is spent — further gains need new signal (search-based self-play, or the batched-update line in #98/#99).
Cost: 3 × 20,000 games (460 s / 319 s + 345 s / 460 s) plus 3 nets × 9 ranges × 20 eval games.

## 200,000 games from scratch across seeds (#62)

| training seed | in-protocol (2000..6000) | vs committed (40,282) |
|---|---|---|
| 1 | 40,282 | 0 (byte-identical, `md5 6c9a164b...`) |
| 2 | 38,116 | -2,166 |

Seed 1 reproducing the committed file byte-for-byte confirms the committed weights are exactly `200000 games, lr 0.003, seed 1` and
that the pipeline is deterministic. Seed 2 lands below it: retraining from scratch does not improve the model, and the win came from
fine-tuning (#66) at 40,000 games (about 1/5 of the cost). Cost: 2 × 200,000 games, 4,118-4,254 s each.

## Symmetry augmentation + TD(lambda): diverges (#72)

Two ways to get more out of each self-play game: all 8 dihedral orientations per position, and TD(λ) traces backwards along the game.

Opt-in `nn.Options`: `symmetry` applies each per-move TD(0) update to all 8 dihedral transforms with the same target (no RNG);
`lambda` switches to TD(λ) with accumulating traces (`trace = λ * trace + grad V`, `net -= lr * (out - target) * trace`, λ = 0 = TD(0)).
CLI variants `sym` and `symlam` (symmetry + λ = 0.7). `base` and earlier variants stay byte-identical; checksum test untouched.
Correctness: one `sym` step equals eight explicit `forward(transformBoard(k, board))` + `train` steps (1e-15), λ = 0 matches TD(0),
the 8 transforms are distinct and closed under composition, and `forward` on a transformed board equals `forward` with transformed weights.

20,000 games per seed, ranges 2000..6000 × 20 games, nn 1-ply:

| variant | lr | per-seed mean | mean ± stderr | weights |
|---|---|---|---|---|
| base (reference, #56) | 0.003 | — | 18,754 ± 534 | fine |
| `sym` | 0.003 | 2,116 / 3,472 / 2,116 / 2,772 / 3,035 | 2,702 ± 264 | 2 of 5 nets NaN/Inf |
| `sym` | 0.000375 (lr/8) | 4,274 / 6,828 / 2,762 | 4,621 ± 1,187 | finite, collapsed (V ≈ -51 everywhere) |
| `symlam` | 0.000375 | 2,116 / 4,958 / 2,116 | 3,064 ± 947 | 2 of 3 nets all NaN |

Eight updates per position is not eight times the data: it is eight correlated steps against a target from the pre-step net, which
TD(0) with function approximation cannot absorb. At the normal lr the weights blow up to NaN within 20,000 games; at lr/8 they
collapse to a near-constant V (dead ReLUs); λ = 0.7 multiplies the same over-update. Dividing the lr keeps the total step size but
not the dynamics (the per-game lr decay schedule is compressed 8×). Not adopted; both stay off by default. A version that could work
would average the 8 gradients from one snapshot into a single step (true batch augmentation) and clip the gradient norm — a larger
change, not a knob. Cost: 2 settings × 5 seeds and 1 × 3 seeds, ~150-800 s per run.

## Batched symmetry update: symbatch (#98)

Follow-up of #72: reuse the 8 orientations without eight correlated steps.

Opt-in `nn.Options.sym_batch` (CLI variant `symbatch`): for one afterstate and target `t` it runs `forward` + `gradOf` on the 8
transformed boards **while `net` is untouched**, averages them elementwise (`mean_k (V_k - t) * grad V_k`), clips the averaged
gradient's global L2 norm to `nn.sym_batch_clip = 1.0`, and applies one step `net -= lr * clip(mean_k ...)`. Deterministic.
`base` and earlier variants stay byte-identical; checksum tests untouched.

Why 1.0: averaged-gradient norms over one self-play game are 0.04..0.26 on a fresh net but 4..300 late in training on the committed
net (TD errors grow with the value scale, ~36..40 there). A unit clip is a no-op early (where lr is largest) and binds late, capping
one update's displacement at `lr = 0.003` against a committed mean |w| of 0.44 — #72's blow-up path (weights moving by their own
magnitude per move, 8× per position) is cut off instead of pre-empted by a smaller lr.
Correctness: one `symbatch` step equals the averaged gradient applied once (1e-15), differs from eight sequential `sym` steps, and at
t = 100 the applied norm is exactly `lr * sym_batch_clip` while the unclipped average is >10× the clip (the clip branch is exercised).

20,000 games, training seed 1, ranges 2000..6000 × 20 games, nn 1-ply:

| variant | lr | mean over ranges | per-range | train moves (games) |
|---|---|---|---|---|
| base (#56) | 0.003 | 18,754 ± 534 (5 seeds) | — | — |
| `sym` (#72) | 0.003 | 2,702 ± 264 | — | — |
| `sym` (#72) | 0.000375 | 4,621 ± 1,187 | — | — |
| `symbatch` | 0.003 | 1,314 | 1,292 / 1,337 / 1,404 / 1,309 / 1,227 | 2,568,519 (~128/game) |
| `symbatch` | 0.000375 | 794 | 732 / 883 / 766 / 788 / 803 | 1,858,935 (~93/game) |

No NaN (unlike `sym`), but it collapses: 90-130-move games dying at tile 128-256, below even the collapsed `sym` rows. Likely the
clip: norms reach 4-300 late, so a 1.0 cap throttles updates to 1/4-1/300 of natural size exactly when V would mature, and lr/8
shrinks the early updates too — hence the lr/8 row is worse. #72's failure mode is gone, but TD(0) with a bootstrapped target gains
nothing from the extra transforms and the throttled step cannot recover. Not adopted; `sym_batch = false` stays default. If retried:
a clip sweep (10, 100, none) and/or a target averaged over the same 8 transforms so the regression is self-consistent.
Cost: 2 × 20,000 games (435 s + 335 s) plus 5 ranges × 20 eval games per net.

## n-tuple network runner (#70)

`ai/solver/ntuple.zig` holds flat f32 pattern tables (`[16^k]f32`) indexed by the same `levelOf` encoding as the MLP;
`ai/solver/ntuple_main.zig` trains them by TD(0) on afterstates and evaluates them.

- Patterns, one table each, shared across the 8 dihedral transforms: row 0, row 1, 2x2 top-left, 2x2 middle column, main diagonal
  (4-tuples), 2x3 top-left (6-tuple). The index is the minimum over the 8 transforms, so rotated/mirrored boards hit the same weights.
- Index cost: a compile-time `cell -> (table, shift)` slot table, so one 16-cell scan yields all 48 transform accumulators.
- Update: `table[p] += lr * (r + V(next) - V(after)) / n_tuples`, reward = score delta / 1000.
- File: 8-byte magic + version + endianness + pattern count + per-table lengths + f32 LE payload.
- CLI: `zig build ntuple -- <train_games> <eval_games> <eval_seed> <lr> <base|chance> <table_path> [train_seed]`; `train_games = 0`
  loads the table. Eval prints nn-shaped lines labelled `ntuple`. `chance` (spawn-lookahead self-play) is much slower than `base`.
- Not wired into `solver.Policy` / `ranking` / `server` (adoption was a follow-up PR).

Smoke (`zig build ntuple -- 200 20 2000 0.01 base /tmp/t.bin 1`): 200 games, 37,341 moves, 0.2 s. Eval on seeds 2000..2019:
1-ply 2,439, +1 spawn 8,710, expectimax depth 2 21,840. At 2,000 games (10 eval games) 3,118 / 10,213, so the tables do learn.
Table file: 68,419,628 bytes raw, 106,259 bytes with `gzip -9` after 200 games (182,822 after 2,000).

### Scaling the n-tuple tables (#70)

Serial, initial lr 0.01 (linear decay to 10%), training seed 1, 68.4 MB raw (5 four-tuples + one 2x3 six-tuple):

| train games | moves | time | 1-ply | + 1 spawn |
|---|---|---|---|---|
| 200 | 37,341 | 0.2 s | 2,439 | 8,710 |
| 20,000 | 6,218,061 | 3.7 s | 4,426 | 11,385 |
| 200,000 | 86,039,320 | 51.0 s | 7,783 | 15,100 |
| 1,000,000 | 492,306,926 | 288.0 s | 8,832 | 17,093 |

Still climbing at 1M games (~60 games/s) but far below the MLP (`nn2` 72,205). The classic n-tuple result needs tens of millions of
games and a much lower per-sample lr; at this scale the MLP is better per unit time. The six-tuple dominates the raw size while the
gzip payload stays small early. Not adopted as a policy; follow-ups in #73.

## Deeper search: nn2 (2-ply) (#69)

`nn` is 1-ply + one spawn lookahead. Every extra ply costs 4 moves × ~6 empty cells × 2 tile values, but the afterstate value is
cheap enough for one more level, so the question was how much score it buys.

`nn.Mode.deep`: my move → exact spawn over all empty cells × {2, 4} (p = 0.1 for 4) → my move → spawn → leaf = afterstate value
+ score_delta / reward_scale. Deterministic (no RNG), bounded value cache, exposed as `solver.Policy.nn2` (ranking depth 2) and
`/api/solve?algo=nn2`. `nn` unchanged and still default; checksum test untouched.

Committed weights, ranges 2000..6000 × 20 games, ReleaseFast:

| range | nn | nn2 | ms/move nn | ms/move nn2 |
|---|---|---|---|---|
| 2000 | 64,052 | 72,220 | 0.10 | 0.97 |
| 3000 | 64,231 | 73,277 | 0.10 | 1.07 |
| 4000 | 62,665 | 72,320 | 0.10 | 0.81 |
| 5000 | 63,276 | 73,410 | 0.04 | 0.47 |
| 6000 | 67,640 | 69,797 | 0.05 | 0.48 |
| **mean** | **64,373** | **72,205** | **0.08** | **0.76** |

+7,832 (+12.2%) for ~10× the time per move, still under 1 ms/move and ~1,200 leaves/move, so it is affordable in the browser
(expectimax depth 2 needs 0.04-0.08 ms/move for ~22-25k). Adopted as an extra policy: `nn` stays the default for speed, `nn2` is the
quality option. Canonical bench after it landed: nn2 **71,611.4** > nn 65,409.9 > expectimax_tt 41,216.4 > mcts 29,722.0 >
expectimax 25,266.8 > ga 24,046.1 > beam 12,695.9 > greedy 3,423.8 > random 1,061.5; `src/web/ranking.json` regenerated.

## Row-LUT board representation: no gain (#71)

Board packed into one `u64` (4 bits per cell, the `nn.zig` encoding) with 65,536-entry row tables for left/right and a transpose
for up/down — four lookups per slide instead of a per-row scan. `ai/solver/fastboard.zig` (`fromBoard`/`toBoard`,
`slidePacked(board, dir) -> {board, score, changed}`, `emptyMask`, `setCell`), wired into `solver.zig`'s expectimax
(`maxValuePacked`/`chanceValuePacked`, old path kept as the test oracle).
Correctness: 1,000 pseudo-random boards × 4 directions reproduce `core.GameState.slide` exactly (board, score delta, `changed`),
round-trip `toBoard(fromBoard(b)) == b`, and searched values/best moves match. The scores below are byte-identical, so the LUT is right.

ReleaseFast, seed 2000:

| mode | moves/s (branch) | moves/s (main) | avg score | avg moves |
|---|---|---|---|---|
| expectimax depth 2, 20 games | 10,845 | 12,707 | 26,153.6 | 1,373.8 |
| expectimax depth 3, 10 games | 280 | 301 | 42,718.8 | 2,084.0 |

15% slower at depth 2 and 7% slower at depth 3 with the same scores: a slide is 4 lookups plus a transpose for half the directions
while `core.slide` is already a tight 4-row scan, and every leaf pays a `toBoard` because the evaluation still takes a `core.Board`.
Move generation was never the bottleneck — chance-node recursion (6-8 spawns per node) and evaluation dominate. Not adopted, code
dropped: it would only pay off with a packed-native evaluation and packed spawn, i.e. rewriting the search around the representation.

## n-tuple: deep search, light tables, resume (#73)

#70 left the runner at 1-ply / 1-spawn and 68 MiB raw dominated by one 6-tuple.

- `Mode.deep` — my move → exact spawn → my move → spawn → leaf = afterstate value + score_delta / 1000 (same leaf rule as `nn2`).
  Deterministic; the eval line reports `leaves`, `leaves/move`, `ms/move`.
- `Set.light` — only the five 4-tuples, 327,680 entries, **1.3 MiB raw** instead of 68 MiB.
- `resume` / `lightresume` — load the table and train more games; the load validates length, so a full-set file cannot be loaded into
  a light table (`error.TableSizeMismatch`).

20 games per policy on eval seed 2000, ReleaseFast:

| training | 1-ply | + 1 spawn | depth 2 | ms/move (depth 2) |
|---|---|---|---|---|
| light, 2,000 games | 2,444 | 7,502 | 15,915 | 0.39 |
| light, 500,000 games | 7,134 | 17,987 | **25,760** | 0.24 |
| expectimax depth 2 (no table) | — | — | 21,840 | 0.02 |

Self-play training is cheap (500k games in 124 s, 202M moves) but a 1-ply table stays weak (7,134). Almost all strength comes from
depth: the same table jumps to 17,987 with one spawn and 25,760 at depth 2 (0.24 ms/move, ~4.6k leaves/move), beating expectimax
depth 2. Against the MLP on the same protocol (seed 2000, 20 games): `nn` 64,052, `nn2` 72,220, depth-2 n-tuple 25,760 — about 2.5×
below `nn`, so the table is not a replacement for the learned value, only an experiment/runner (and a possible cheap browser-side
fallback if 1.3 MiB matters more than strength). Adopted as the best non-MLP option, so `light` + `deep` is the default sweep config.
Not wired into `algo` (ranking regeneration and `Policy.nntuple` were follow-ups). Cost: 124 s training + 6.5 s per 20 depth-2 games.

## Static WASM build (#68)

`zig build wasm -Doptimize=ReleaseFast` produces a static site in `zig-out/web/` (`index.html` + `game2048.wasm`) that plays and
solves entirely in the browser — no backend, so it can be hosted on GitHub Pages.
`src/wasm.zig` exposes a C ABI (`init(seed_lo, seed_hi)`, `move(dir)`, `solve(algo, depth)`, `board_ptr()`, `score()`,
`best_tile()`, `algo_count()`, `algo_name(i)`) over the same `core`/`solver`; the nn weights are `@embedFile`-ed, so all nine
policies work offline. `src/web/index.static.html` is a separate static UI (it never talks to `/api/*`). `scripts/wasm-smoke.mjs`
(node, no dependencies) builds a game, applies moves and calls every policy, exiting non-zero on a bad direction or unchanged board.

Two browser fixes: `nn.ValueCache.get` indexed with `key & (capacity - 1)` where `usize` is 32-bit on wasm32 (slot made explicitly
`usize`); and the 8,192-entry value cache (128 KiB) lived on `deepMove`'s stack and overflowed the wasm stack
(`RuntimeError: memory access out of bounds` on the first `nn2` call) — the capacity is now comptime-selected, 1,024 on wasm and
8,192 elsewhere, so native `nn2` is unchanged.
Measured: module 745,167 bytes; smoke test passes for all nine policies and ten applied moves.

## Truncated-rollout MCTS with the NN leaf: nnmcts (#82)

`mcts` rolls out whole random games and scores them by raw game score, so it only knows what a random player accumulates. `nnmcts`
keeps the same root UCT but stops each rollout after 8 random plies and scores the truncated board with the trained MLP:
reward = `score gained / 1000 + net value of the final board` (1000 = the `reward_scale` the net was trained on), and a dead board
adds 0 future value.

`nnmctsMove` builds the four root arms from slide legality and per simulation runs `selectArm` (UCT rule extracted verbatim from
`mctsMove`) plus one `nnmctsRollout` reward. The rollout takes uniformly random legal slides and spawns after each ply except the
last, so the leaf is a post-slide, pre-spawn afterstate — the shape the net was trained on. The first version evaluated the
post-spawn board and scored 8,745 (200 sims × 8 plies) against 17,023 for the same budget once the leaf matched: the leaf's board
shape matters more than the rollout length.
Wiring: `Policy.nnmcts` after `nn2`, so `algo=nnmcts`, `algo_count`/`algo_name`, the static UI and the smoke test pick it up;
`src/web/ranking.json` deliberately not regenerated, so the UI still lists the nine adopted policies.

ReleaseFast, committed weights, eval seed 1000, 10 games per row, serial:

| sims × plies | avg score | ms/move |
|---|---|---|
| 200 × 2 | 17,085 | 0.27 |
| 200 × 8 | 17,023 | 0.91 |
| 200 × 16 | 13,717 | 1.19 |
| 500 × 16 | 20,054 | 2.82 |
| 1,000 × 2 | 17,115 | 1.65 |
| 1,000 × 4 | 23,646 | 3.16 |
| 1,000 × 8 (shipped) | 20,137 | 5.41 |
| 1,000 × 16 | 23,113 | 4.12 |
| 2,000 × 8 | 27,346 | 9.61 |
| 5,000 × 2 | 20,026 | 2.91 |

References on the same seed: `mcts` 29,722 (committed ranking, 5 × 20 games); `nn` 64,601 at 0.34 ms/move and its 1-ply greedy
variant 38,616 at 0.008 ms/move (10 games).

Truncation plus a net leaf does not beat plain `mcts`: 2,000 sims reach 27,346, i.e. `mcts`'s 29,722 at ~7× the per-move cost, and the
shipped 1,000 × 8 point lands at 20,137 for 5.4 ms/move (~6.9 s/game against `mcts`'s ~1.3 ms/move). Random play drives the board off
the training distribution, so after 8 random plies the leaf value is noisy and the arm means collapse toward each other; the same leaf
used the way it was trained (argmax over 1-ply afterstates plus one spawn sample) is worth 64,601 at 0.34 ms/move — 3× the score for
1/16 of the time per move. More sims or deeper rollouts produced no consistent improvement even at the best point.
Not adopted for `ranking.json` or as a default; kept as an experimental policy to document the limit of pairing random rollouts with
this value net. The canonical 5 × 20 `zig build ranking` run did not finish within the working timeout, so the decision rests on the
10-game table above (a few thousand points of noise per row). `mcts`/`nn`/`nn2` behaviour unchanged: the UCT and best-arm rules were
extracted verbatim with the same RNG call order. Wasm: three separate 148 KB per-move net copies (one per NN arm) overflowed the wasm
stack (`solve` failed even for `algo=0`, since the copies live in `choose`'s frame); the three arms now share one copy, and the module
grew 745,167 → 753,911 bytes (286,245 gzipped). Smoke test passes for all ten policies and ten applied moves.
## NN-value beam with sampled spawns (#83)

`beam` uses the hand-written snake/merge evaluation and assumes a 2 on every empty cell; `nn`/`nn2` walk only one or two own moves.
`nnbeam` puts the learned value under a small beam: fixed width 8, horizon 3 own moves, 2 sampled spawns between plies.

`nnBeamSearch(net, state, rng, horizon)` keeps two `[8]NnBeamNode` arrays (`state`, `first_move`, `reward`, `value`) with
`insertBeam`-style insertion; rewards are `score_delta / nn.reward_scale`, leaves are `nn.forward` on the afterstate (same rule as `nn`).
Spawns are sampled only *between* plies (uniform empty cell, 2 with p = 0.9 else 4); the last ply inserts the afterstate without a
spawn, matching `nn2`, which ends on `chance(..., 0)` — a post-spawn leaf would ask the net about a board it never saw. A full board
keeps the afterstate as leaf; no legal root move returns `null`. `first_move` is copied from the ply-0 direction (a test asserts
`canSlide`), randomness comes only from the caller's `std.Random` (a test replays 12 random boards twice), `depth` is ignored, and
`Policy.nnbeam` is registered so it reaches the server, the wasm `algo_count`/`algo_name` and the static UI. Wasm: routing `nnbeam`
through its own `nn.parse` arm pushed `choose`'s frame over the wasm stack (even `random` failed, since a frame is allocated on entry);
the three NN policies now share one `nn.parse` per call.

ReleaseFast, one run per policy, same seed and order (`zig build ranking -Doptimize=ReleaseFast -- 1 20 2000`, 3m47s):

| seed 2000 / 20 games | avg score | avg moves | ms/move |
|---|---|---|---|
| nn2 | 72,960.0 | 3,318.7 | 1.947 |
| nn | 63,486.0 | 2,911.5 | 0.200 |
| expectimax_tt | 46,744.8 | 2,267.9 | 0.695 |
| mcts | 29,310.6 | 1,523.3 | 1.001 |
| expectimax | 26,153.6 | 1,373.8 | 0.081 |
| **nnbeam** | **25,480.2** | **1,327.9** | **0.124** |
| ga | 22,189.8 | 1,205.3 | 0.078 |
| beam | 12,584.6 | 760.1 | 0.067 |

`nnbeam` is exactly 2.03× `beam` (25,480 vs 12,585) at 0.124 ms/move, so the sampled beam plus a learned leaf does lift the shallow
baseline, but it stays just below `expectimax` depth 2 and far below `nn`: three plies of width 8 with 2 spawns evaluate ~64 leaves
per move, each reached through one sampled spawn per gap, so the search commits to one spawn outcome instead of averaging it —
the structural reason a sampled beam trails 1-ply-plus-expectation. Kept as an experimental policy, not a default. Correction: the
first version of this section reported seed-1000 10-game numbers (23,875 / 28,493 / 16,513) from an implementation whose last ply
inserted a post-spawn child; those were dropped, and the test `nnbeam evaluates the afterstate before spawn at the final ply` pins the
rule at horizon 1. `src/web/ranking.json` not regenerated (the canonical 5 × 20 run costs ~5 min).

## n-step TD: worse than TD(0) at n = 4 (#84)

TD(0) trains each afterstate from the next afterstate's value, one move old. A finite n-step return
(`reward_t + … + reward_{t+n-1} + V(s_{t+n})`) propagates a terminal reward back n moves per update and stays online, unlike the
failed backward sweep (#33).

Opt-in `nn.Options.nstep` (0 = off, 1 = TD(0), n > 1 = n-step): `NStepWindow` holds the pending steps; each new step's `V` closes
the target of the step leaving the window (`sum of its n rewards + V`), and `nextFinish` at game over flushes the last
min(n, game length) steps with rewards alone (`V(terminal) = 0`). The trajectory matches TD(0) exactly (`n = 1` is byte-identical,
tested); other options are ignored; the weight format is unchanged. CLI variant `nstep` (n = 4).

20,000 games, training seed 1, ranges 2000..6000 × 20 games (same seeds for both nets), ReleaseFast:

| variant | lr | train moves | train time | 1-ply | + 1 spawn | 2-ply |
|---|---|---|---|---|---|---|
| base TD(0) | 0.003 | 13,346,693 | 145.0 s | 19,175 ± 895 | 30,918 ± 816 | 43,048 ± 1,549 |
| `nstep` (n = 4) | 0.003 | 10,150,588 | 100.6 s | 13,055 ± 641 | 23,755 ± 296 | 29,886 ± 679 |
| `nstep` (n = 4) | 0.00075 (lr/4) | 8,335,318 | 89.1 s | 9,019 ± 231 | 15,253 ± 415 | 20,351 ± 584 |

(expectimax depth 2 on the same seeds: 25,004 ± 1,108.)

At the normal lr n = 4 loses 6,120 on 1-ply (−32%) and 13,162 on 2-ply (−31%), far outside stderr, and lr/4 is worse again
(−53% / −42%) — so the n-step target is the problem, not the step size. The same game budget yields 24% fewer training moves: the
policy plays shorter games, so n-step training compounds its own regression (the faster wall time is that, not a gain). A state's
update lands n−1 moves after it was visited, against a net that has moved on, so the bootstrapped part is stale and the target scale
is n rewards — the same failure as the backward sweep (#33) and TD(λ) (#72): value-side tricks that lag or multiply the online update.
Not adopted; `nstep = 0` stays default. Un-tested: n = 2/3, other lr, a frozen-snapshot batch average of n-step targets with a
normalised target scale. One seed, one budget. Cost: 3 × 20,000 games (145 / 101 / 89 s) plus 4 policies × 5 ranges × 20 eval games.

## NN weights parsed once per thread, not once per move (#87)

`choose` re-parsed the embedded ~148 KB weight block into a `Net` on every move of every NN policy. Now a `threadlocal` cache parses
once per thread and each move passes a `*const Net`, so parallel ranking / HTTP / WASM keep one copy per thread with no shared
mutable state. The `parse_count` seam in `nn.zig` pins it (a fixed-seed nn/nn2 game parses exactly once); golden tests pin the moves.
Correctness: seed 1000 / 5 games identical before and after (nn2 71,880.0 / 3,267.2, nn 69,640.8 / 3,151.2, nnbeam 21,096.8 / 1,143.8,
nnmcts 12,864.8 / 763.4); `zig build test` and the wasm smoke (11 policies, 10 moves) pass on the rebased branch.

`latency_ms_per_move` from the #89 instrument:

| policy | before (b389ba5) | after | delta |
|---|---|---|---|
| nn2 | 1.8864 | 1.8658 | -1.1% |
| nn | 0.1995 | 0.1955 | -2.0% |
| nnbeam | 0.1233 | 0.1197 | -2.9% |
| nnmcts | 2.5574 | 2.5930 | +1.4% (noise) |

Seed 2000 / 10 games reproduces the direction (nn 0.1961 → 0.1947, nnbeam 0.1210 → 0.1190); same-code repeats span nn
0.1902-0.1947 and nnbeam 0.1175-0.1190, so every before run sits above every after run for those two, while nn2's -1.1% is inside the
band. The removed work is one 148 KB copy per move — the visible 1-3% for the fast policies; nn2's search dominates and hides it.
Adopted: byte-identical behaviour, and the enabling step for reusing a parsed net inside a search. `src/web/ranking.json` untouched.

## Search workspaces reused per thread and per game (#88)

`nn2`'s `ValueCache` (built inside every `Search`, i.e. every move) and `expectimax_tt`'s 8192-entry TT (allocated and cleared per
`bestMoveTT` call) are now reused workspaces: `nn.sharedValueCache()` and `ttWorkspace()` hand out one threadlocal workspace each
(lazy-cleared on first use), `nn.bestMoveWithCache` lets a caller own the cache, `nn.playWithStats` keeps one cache per game, and
`nnmcts` reuses `sharedValueCache()` across rollouts and moves. Entries validate the full board plus depth and kind (`ValueCache` keys
on the 64-bit board hash and drops on mismatch), so a warm workspace changes speed only.
Correctness: seed 1000 / 5 games identical before and after (nn2 71,880.0 / 3,267.2, expectimax_tt 51,376.0 / 2,426.2, nnmcts
12,864.8 / 763.4). Four tests cover fresh-vs-warm identity over 30 moves (nn) and 2×40 moves across two games (TT, also asserting one
shared workspace), plus slot-collision behaviour (a colliding entry is dropped, never returned). Wasm smoke passes (755,150 bytes,
11 policies).

| policy | before (3 runs, 8986c30) | after (2 runs, 3ac831d) | delta |
|---|---|---|---|
| nn2 | 1.8399-1.8668 | 1.7395-1.7427 | -5.5% |
| expectimax_tt | 0.6965-0.7058 | 0.6613-0.6620 | -5.0% |
| nnmcts | 2.5701-2.5920 | 2.5348-2.5362 | -1.4% |

Bands do not overlap for any of the three. Temporary counters (seed 1000, 200 moves): `nn2` with the shared cache hits 88.2%
(2,418,915 / 2,743,720) and evaluates 1,624.0 leaves/move vs 1,745.6 fresh (−7.0%, 324,805 vs 349,113 leaves); `expectimax_tt` probes
1,261,389 (6,306.9/move) vs 1,313,235 fresh (−3.9% expansions), hit rate 55.7% vs 54.5%. Memory moved from per-call stack to TLS:
TT 8192 × 32 B = 256 KiB/thread, `ValueCache` 2^13 × 16 B = 128 KiB/thread (16 KiB on wasm); the 128 KiB `@splat` reset and the
8,192-entry `valid = false` loop now run once per thread instead of once per move. nnmcts shares only the cache and gains least;
expectimax_tt gains from both the removed clear and cross-move hits. Adopted.

## Order penalty follows the game phase, looser after 4096 (#92)

#92 proposed tightening the snake-order penalty late, so the endgame builds the merge chain that reaches 8192. What shipped is the
mechanism, and the measurement inverted the hypothesis: `Weights.order_penalty_late` is used by `evaluateW` once the largest tile
reaches 4096 (level ≥ 12), defaulting to `order_penalty`, so pre-#92 behaviour is unchanged unless a weight sets it; `ga_weights`
keeps one gene for both phases.

expectimax_tt depth 3, 50 games per cell, seeds 1000-1049 (avg score / ≥4096 / ≥8192 / dead <2048):

| config (mid / late) | avg score | >=4096 | >=8192 | dead <2048 |
|---|---|---|---|---|
| A 30 / 30 (old baseline) | 37,985.2 | 9 | 1 | 10 |
| B 43.174 / 43.174 | 39,981.5 | 14 | 0 | 8 |
| 30 / 60 (tighten late) | 35,069.0 | 9 | 0 | 10 |
| 43.174 / 80 (tighten late) | 39,221.7 | 14 | 0 | 8 |
| 43.174 / 60 | 40,437.0 | 14 | 1 | 8 |
| **43.174 / 25 (loosen late, adopted)** | **42,230.7** | 14 | 2 | 8 |
| 43.174 / 10 | 39,996.6 | 14 | 0 | 8 |
| 43.174 / 0 | 37,956.9 | 16 | 0 | 8 |

Tightening lost to the constant in every variant; loosening won, best at 25. Seeds 2000-2049 confirm: 43.174/25 → 38,383.7 (≥4096 16)
vs A 38,025.2 (≥4096 13). Across both sets 43.174/25 averages 40,307 vs A 38,005 (+6.1%) with 30 vs 22 runs reaching 4096; 50-game
cells reproduce exactly. The 5-game `ranking` snapshot on seed 1000 moved the other way (expectimax 41,107 → 19,964, beam
14,369 → 12,329) but that is small-sample noise: at 50 games expectimax depth 2 is 24,297.7 (C5) vs 24,403.5 (A) on 1000-1049 and
25,385.7 vs 24,220.1 on 2000-2049, beam 12,346.7 vs 11,373.8, greedy 3,195.5 vs 3,109.8, mcts 30,709.7 both (evaluation-independent).
No weight-driven policy regresses at 50 games.

Adopted as `baseline = .{ .order_penalty = 43.174, .order_penalty_late = 25 }`. Reading: the mid-game wants the snake order enforced
(43.174 > 30, as the old GA weights found), but once the 4096 anchor is secured the penalty should relax — holding the chain rigidly
costs the merges that grow 4096 into 8192. The 8192 tail (2 runs vs 1) is suggestive, not conclusive at 100 games. Limits: one machine,
100 games per config; the 5-game ranking snapshots below are too small to compare weight sets; GA evolution is unchanged (one gene);
no sweep between 25 and 43.174.

## par for stateless variants (#99)

`trainBatch` hard-coded `trainGame`, so only the plain per-move variant could batch. It now delegates to `trainBatchOpts(.., opts)`,
which runs each game with `trainGameOpts(.., opts)`: same `net_start + (1/N) * Σ_i (net_after_i - net_start)` aggregation over
variant-trained per-game nets. `trainBatch` stays as the default-options wrapper, so the plain path and its checksum are unchanged.

- CLI: arg 6 `par` keeps its meaning; the new positional arg 10 `par` batches the variant named in arg 6:
  `zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - sym 30 /tmp/nnb/sym.bin 1 par`. Arg 10 was an unused trailing slot
  (any 10th positional forced the usage error), so every existing command line keeps its meaning. The log prints `variant=sym+par`.
- Works: `base`, `backward`, `replay`, `sym`, `symlam`, `nstep`, `resume` (the batch starts from the net loaded out of `weights_path`).
- Not supported: `priority`, `midgame`, `midmix` — their 1024-entry ring buffer is cross-game state shared by pointer and the batch mean
  would change their semantics, so the CLI prints usage and refuses; `trainBatchOpts` asserts the buffer is null.
- Reproducibility: `parallel training batch with variant opts matches a serial run of the same game seeds` runs 4-game batches with
  `symmetry` and with `replay_k = 4` on threads and compares byte-for-byte with the serial reconstruction over seeds 77..80; two CLI runs
  each of `sym` + `par` and `replay` + `par` wrote identical files. `zig build test` passes.
- Speedup (8 CPUs, batches of 8): `sym` 2,000 games 16.5 s → 2.7 s (**6.1×**); the #66 `resume` fine-tune +20,000 games lr 3e-4
  460 s → 90.9 s (**5.1×**), so #35's 5.7× carries over to the heavier variants.
- Does batching change the outcome? `resume` +20,000 lr 3e-4, training seed 1, the #97 `a2` recipe as `resume` + `par`: in-protocol
  (2000..6000) 42,736 vs serial 43,083, held-out (7000..10000) 43,388 vs 41,936 — within eval noise on both sets, so batching trades
  nothing measurable for ~5×. (Both stay below the committed 45,342 held-out, which is #97's overfit finding, not a batching effect.)

## Death reward learns moves-to-death, but the fragility penalty does not pay (#101)

Two pieces shipped. `nn` gains an opt-in `death` variant: per-move reward `-1/reward_scale` instead of the score delta, so
V ≈ −(moves to game over)/1000. Training uses the backward full-return sweep, not one-step TD — with −1 per move the TD target is
identical on every board and the net collapsed to a constant (observed, then fixed; the naive scale also blew up to NaN, hence the
`/reward_scale` unit). `evaluateW` gains `Weights.fragility` + `Weights.death_net`, adding `fragility * (0 - V)`; both default inert,
so every existing policy is bit-identical unless a caller opts in.

Probe (temporary harnesses, removed): 1k and 5k death games, then V vs true moves-left on expectimax_tt depth 3 games
(seeds 3000+). In-distribution (the death policy's own games) it works: pearson(V, ml) = −0.447 (1k) → −0.606 (5k). On real games it
does not: 0.023 (1k) / −0.001 (5k) overall, and inside every empties bucket |r| ≤ 0.12 with mixed signs. The ml ≤ 100 tail keeps the
signal: n = 2000, r = −0.230 overall, empties 1-6 → −0.137 / −0.030 / −0.216 / −0.056 / −0.197 / −0.496, terciles monotone. So V ranks
near-death boards and saturates OOD — exactly the training horizon.

A/B of the penalty (50 games × 2 seed sets, expectimax depth 2, death net trained 5k games):

| fragility | avg (1000-1049) | avg (2000-2049) | dead <2048 (both sets) |
|---|---|---|---|
| 0 (off) | 24,297.7 | 25,385.7 | 52/100 |
| 300 | 19,420.6 | 24,644.3 | 58/100 |
| 1000 | 26,354.1 | 25,614.8 | 50/100 |
| 3000 | 26,345.5 | 24,204.2 | 49/100 |

The λ = 1000 bump did not replicate: a paired run (same seeds, off vs on) over 200 games on fresh sets 1000/2000/3000/4000 gives
+2056.4 / +229.1 / −1306.3 / −1490.8, total −127.9 ± 1477.3, and deaths move the wrong way (93 → 104 of 200). Noise with a hint of harm.

Not adopted: `fragility` stays an opt-in field (tested, default 0), nothing sets it, committed weights unchanged; the death variant
works as advertised within its horizon and stays as `nn … death`. The tail correlation (r ≈ −0.23..−0.50 at ml ≤ 100) is real but too
thin to rank mid-game moves, and OOD saturation makes the term nearly constant across candidate boards. A usable fragility signal
needs training on the real policy's game lengths, not on self-play deaths. Limits: one machine; 4 λ × 100 games plus a 200-game paired
run; the ml ≤ 100 cut is post-hoc; death nets trained only 1k/5k games; the A/B plays expectimax depth 2, not nn/nn2.

## 楽観的初期化 (OI) の検証 (#107, 2026-10-28)

Guei et al. 2021 の楽観的初期化を `nn` の TD(0) afterstate 学習に移植し、`base` と同条件で比べた。**この MLP 構成では不採用**。

`nn.scaleInit(net, k)` で初期重み (w1/w2/w3) を k 倍してから学習する。variant `oi` (k=4) と `oi16` (k=16)。既定経路と committed
weights は不変。出力バイアス `b3` を持ち上げる方式は採らなかった: 定数を足しても全 afterstate が同じだけずれるので argmax が
変わらず、学習者に見えるデータも変化しない（no-op）。

`zig build nn -Doptimize=ReleaseFast -- <games> 10 2000 0.003 2 <variant> 30 <tmp weights> <train_seed> par`。
訓練 4k / 20k / 100k / 400k、train_seed 1 と 2、lr 0.003（既存の線形減衰）。評価は seed 2000.. の 20 局 × 2 レンジ、
`nn 2-ply (deep)`。

| 訓練ゲーム数 | base seed1 | base seed2 | oi (×4) seed1 | oi (×4) seed2 | oi16 (×16) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 12,151 | 11,719 | 6,841 | 7,750 | 10,333 |
| 20,000 | 21,279 | 15,212 | 8,813 | 13,069 | 10,333 |
| 100,000 | 30,145 | 25,669 | 14,113 | 11,069 | 10,333 |
| 400,000 | 76,977 | — | 26,952 | — | 10,333 |

expectimax depth 2 は同じ評価 seed で 36,410（base は 100k で追い抜く）。

- **×16 は発散する。** 保存重みの最大絶対値 1e+291、NaN が 4,374/18,561。NaN で比較が常に偽になり手の選択が固定されるため、
  訓練ゲーム数も train_seed も変えて同じスコア (10,333) になる。1-ply と 1-spawn lookahead も同値で、価値関数が定数に潰れた。
- **×4 は発散しないが学習が大幅に遅い。** 100k で base の 1/2、400k でも 27.0k 対 77.0k。曲線は上昇中なので最終性能が
  並ぶ可能性は残るが、「少ないゲームで同じスコア」という OI の主張は逆方向に出た。
- OI の効果は「初期値の分散を上げて自己対戦を促す」ことに尽きるが、この MLP は既定のランダム初期化ですでに分散を持ち、
  スケールを上げると TD(0) の更新が不安定になるだけだった。

残作業: **MS-TD（段階別評価）は未着手**（空きマス数などの段階ごとにパラメータ集合を持ち、重みファイル形式と `Cache`/`train` の
変更を伴う。「学習が遅くなる」報告 (Matsuzaki 2017) があるため別 issue で設計から）。→ 下の節で実装・測定した。

## n-tuple の systematic パターン集合 (#106, 2026-10-29)

6-tuple を 6 本（`{0,1,2,3,4,5}`, `{0,1,2,4,5,6}`, `{0,1,2,5,6,7}`, `{0,1,4,5,6,9}`, `{0,1,5,6,9,10}`, `{0,4,5,9,10,14}`）使っていたが、
Szubert & Jaśkowski の systematic な集合ではなく、文献の 1M ゲーム結果 (1-ply 100,178) には遠く及ばない。足りない要因が
パターン集合か学習率かを切り分けるため、まず学習率だけを掃引した（既存 6 パターン、1M ゲーム、seed 1、20 局 × seed 2000）:

| lr | 1-ply | +1 spawn | depth 2 |
| ---: | ---: | ---: | ---: |
| 0.01 | 8,831.8 | 17,093.4 | 30,750.4 |
| 0.0025 | 6,953.4 | 16,435.0 | 25,763.4 |

低い学習率の方が悪い。文献の α=0.0025 は 2 段階 (afterstate) TD とセットの値で、そのまま持ってきても効かない。**パターン集合が
本質**と判断して次を実装した。

2x3 矩形の対称軌道を列挙すると 12 位置は 2 軌道に分かれる: 8 位置の軌道（既存 `{0,1,2,4,5,6}` が代表）と 4 位置の軌道
（中央寄り）→ `{4,5,6,8,9,10}`。`indices` が 8 対称変換の最小を取る実装なので同一軌道の別位置は同じ表の重複になり、
**追加は 1 本だけ**で 7 本（`.wide`）が実質的な上限。`Set = .light | .full | .wide` を追加し、既存の `ntuple_weights.bin`
（6 本）は `.full` としてそのままロードできる。

| set | パターン数 | entries | raw |
| --- | ---: | ---: | ---: |
| `.light` | 5 | 14,253,888 | 54.4 MiB |
| `.full` | 6 | 17,104,896 | 65.3 MiB |
| `.wide` | 7 | 33,882,112 | 129.3 MiB |

7 本目で entries が 2 倍になるのは中央の 6-tuple の組み合わせ数が多いため。`gzip -9` で 2,963,242 B (2.9 MB) まで縮むが、
static 配信で 1 MB 未満に収めるには量子化が要る。

| 学習 | games | lr | train seed | 1-ply | +1 spawn | depth 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full (6 本) | 1M | 0.01 | 1 | 8,831.8 | 17,093.4 | 30,750.4 |
| wide (7 本) | 1M | 0.01 | 1 | 13,795.0 | 22,773.8 | **39,344.0** |
| wide | 5M | 0.01 | 1 | 15,777.6 | 27,567.4 | 50,103.2 |
| wide | 5M | 0.01 | 2 | 18,788.8 | — | 45,586.4 |
| wide | 5M | 0.02 | 1 | 17,547.4 | — | **50,399.6** |
| wide | 5M | 0.0025 | 1 | 14,576.6 | — | 34,692.8 |
| wide | 5M | 0.01 | 3 | 17,347.0 | — | 46,434.4 |
| wide | 10M | 0.01 | 1 | 20,013.8 | — | **51,950.6** |

（expectimax depth 2 は同じ評価 seed で 36,410。）

- **パターン 1 本の追加で 1M ゲームの depth 2 が 30,750 → 39,344 (+28%)**、同じ計算量で expectimax depth 2 を超える。
  学習率の掃引より効果が大きい。5M で 50,103 (+27%) とまだ頭打ちが見えない。
- 学習率は 0.01〜0.02 が良く 0.0025 は明確に悪い (34,693)。文献の α はパターン集合とセットで意味を持つ値。
- seed 差は 5M で 50,103 対 45,586 (±5%) あり、この規模の比較は複数 seed が要る。depth 2 の評価では 1 ゲーム 2,400 手超。
  8192 到達はまだ出ていない。
- 学習時間は 1M で 554 s、5M で 4,197 s（4 並列時の実測、単独なら ~2,000 s）。5M→10M では depth 2 が 47,374（3 seed 平均）→
  51,951 で +10%、1-ply は 17,304 → 20,014 で +16% と鈍化し、depth 2 は 50k 前後で頭打ちが見え始めた（9,300 s で 8192 到達も無し）。

残作業: `algo=ntuple` の API / WASM 接続はしていない（wide は 129 MiB で、同梱には量子化と 1-ply 運用が前提。`nn2` の 72,205 に
まだ届かないため、接続は頭打ちになってから判断する）。次に効かせるならパターン追加（`.wide` が 7 本で上限）ではなく、
量子化して 2-ply を安く回すか、afterstate の 2 段階 TD に切り替えるか。

## expectimax_id の統計 (#104, 2026-10-30)

`docs/evalab.md` の反復深化の節は 50 局（5 × 10）だったので標本を増やした（seed 9000）。

| policy | 予算 | 局数 | avg score | avg moves | best | 到達深さ | games/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expectimax（重み付き, depth 3） | — | 200 | 52,735.7 | 2,528.6 | 8192 | 3 | 0.18 |
| expectimax_id | 2 ms | 100 | **75,202.3** | 3,429.3 | 8192 | 8（上限） | 0.07 |

depth 3 は `zig build evalab -- "baseline" 4 50 3 9000`、id は `zig build solve -Doptimize=ReleaseFast -- 10 10 3 9000 expectimax_id`
（予算は `solver.zig` の `id_budget_ms`）。**+43%** で、50 局のときの +34%（1 ms）と整合する。
注意: `baseline` は `#117` で `.monotonic = 32` を採用済み（`ai/solver/solver.zig:52`）なので、`monotonic=32` を明示しても
`baseline` と同じ結果になる（no-op）。
## MS-TD（段階別評価）の検証 (#107, 2026-10-30)

段 = afterstate の空きマス数で 7 分割（`0..2`, `3..4`, `5..6`, `7..8`, `9..10`, `11..12`, `13..16`）。段ごとに独立した MLP を持ち、
TD(0) の更新は遷移先 afterstate の段のネットだけに当てる。探索は `nn.Value = union(enum) { single: *const Net, staged: *const Staged }`
を通して透過に使えるので、1-ply / 2-ply の探索コードは共有のまま staged 価値でも動く。重みファイルは専用 magic を持ち、
既存の単一ネット形式はそのままロードできる。

学習 `zig build nn -Doptimize=ReleaseFast -- <games> 10 2000 0.003 2 {base,mstd} 30 /tmp/w.bin 1`、評価は 20 局
（10 局 × 2 レンジ、seed 2000 から）で全 variant が同一 seed。`par` は使わない。

| 訓練 | 1-ply | +1 spawn | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: | ---: |
| base 100k | 44,901 | 69,546 | 87,824 | 1,064 s |
| mstd 100k | 47,427 | 73,045 | 89,079 | 1,414 s |
| mstd 400k | **2,288** | 2,288 | **10,333** | 1,447 s |

（expectimax depth 2 は同じ評価 seed で 36,410。）

- **100k では base と同等**（89.1k 対 87.8k、stderr ±4.9k / ±1.1k）。段を分けた容量増はこの規模では誤差に埋もれる。
- **400k では崩壊する。** 1-ply 2,288、2-ply 10,333。学習ログの総手数は 147.6M = 369 手/ゲーム（100k では 143.1M = 1,431 手/ゲーム）
  なので、100k 以降に方策が短手数に落ち、以降の自己対戦が学習信号を失っている。段の境界をまたぐブートストラップ（γ=1、報酬は
  終端のみ）は段ごとに独立した価値スケールを持つため、段間の整合が崩れると一気に発散しやすい。`lr * (1 - 0.9 * frac)` の減衰でも
  抑えられなかった。
- 対策案（未検証）: 段の境界をまたぐとき目標を切り離す（段内ブートストラップに限定）、段ごとに学習率・γ を分ける。ただし 100k で
  利得が見えていないので、この MLP では追う価値が薄い。

**結論: この MLP + TD(0) の構成では MS-TD は採用しない**（100k で同等、400k で不安定化）。実装は variant `mstd` として残す。

## 副作用として判明: `par` バッチ学習は 3 倍遅い (2026-10-30)

MS-TD の比較用に `par` 無しの base を回したところ、同じ 100k ゲーム・lr 0.003・seed 1・同一評価 seed で大きな差が出た。

| variant | 1-ply | +1 spawn | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: | ---: |
| base + `par` | 8,832 | 17,093 | 30,145 | 187 s |
| base（par なし） | 44,901 | 69,546 | 87,824 | 1,064 s |

`par` は 8 ゲームぶんの勾配を平均し norm clip して 1 ステップにするので、1 ゲームあたりの実効学習率がおよそ 1/8 になる。
**nn の学習は既定で `par` を使わない（または lr をバッチ数倍にする）**のが正しく、これまでの nn 系の計測（OI の A/B など）は
par 前提だったため絶対値が控えめに出ている（同一条件どうしの相対比較はそのまま有効）。速度は 187 s 対 1,064 s（5.7 倍）なので、
速度が要る場面では lr を上げた par を使う選択肢が残る（未検証）。

## n-tuple 表の量子化と `algo=ntuple` の接続 (#129, 2026-10-30)

`.wide` は 33,882,112 entries × f32 = 129.3 MiB でそのままでは同梱できないので、int16 に量子化して 64.6 MiB にし、
`solver.Policy.ntuple` として API から使えるようにした。

- `ai/solver/ntuple.zig` の `Quantized`: int16 本体 + パターンごとのスケール（f64 × 7）。`q = round(w / scale)`、
  `scale = max|w| / 32767`（表ごと）、復元 `w ≈ q * scale`。実際のステップはパターンあたり 1.7e-4（値域 ±6 程度なので丸め誤差は無視できる）。
- ファイルは magic `2048-NQ\0` + version + endian + パターン数 + 表長 + スケール + i16 本体（f32 形式 `2048-NT\0` も読める）。
- 変換は `zig build ntuple-quant -- <in.bin> <out.q16.bin>`。`zig build ntuple` も出力パスが `.q16.bin` で終われば量子化して保存する。

精度（10M ゲーム wide 表、20 局 seed 2000、量子化表を f32 に戻して同じ評価コードで測定）:

| 表 | 1-ply | +1 spawn | depth 2 |
| --- | ---: | ---: | ---: |
| f32（129.3 MiB） | 20,013.8 | 32,243.4 | 51,950.6 |
| int16（64.6 MiB） | 20,060.8 | 28,912.0 | 53,592.2（8192 到達） |

差は 20 局のばらつき（±10%）に埋もれる。**量子化による劣化は観測できない。**

接続: `Policy.ntuple` を enum の末尾に追加し、`solver.choose` は `ntuple.bestMoveQuantized`（1-ply）を呼ぶ。表は**遅延ロード**
（初めて選ばれたとき、スレッドごとに 1 回だけ `ai/solver/ntuple_wide_q16.bin` を `std.heap.page_allocator` に読む）。
`/api/solve?algo=ntuple` は表が無ければ 503 と理由を返す（`solver.ntupleAvailable()`）。WASM はファイルを読めないので
`src/wasm.zig` の `algo_count()` が `-1` して一覧に出さない（enum の末尾に置いたのはこのため）。`ranking.zig` の既定リストには
入れていない（64.6 MiB を CI/計測で読ませない）。

限界: 1-ply のみ（`chance` / `deep` は solver 側から未接続）。1-ply で約 20,000 なので `nn2`（72,205）には遠く、`ranking.json` にも
入れていない。64.6 MiB を git 管理している（既存 `ntuple_weights.bin` 68.4 MiB と合わせて約 133 MiB）。

### 副産物: WASM ビルドが main で壊れていた

`#104` の `clockNow()` が `std.Io.Threaded.global_single_threaded.io()` を呼ぶため、wasm32-freestanding では `std.Io.Threaded` の
`getrandom` / `IOV_MAX` 解決に失敗して `zig build wasm` が通らなくなっていた（#104 の PR 以降、誰も wasm をビルドしていなかった）。
`clockNow` を freestanding 用のスタブ（固定タイムスタンプを返す）に分岐させて修復した。当初ここには「その場合 `expectimax_id` は
深さ 2 で止まる」と書いていたが**間違いだった**: スタブは 0 を返すので `elapsedMs()` が常に 0 になり、
`while (... elapsedMs(start) < budget_ms)` が常に真 → `id_max_depth = 8` まで深化して 1 手 80 s 以上かかっていた（#145）。
いまは `shouldDeepen()` が「時計が進むターゲット」を条件に入れているので、wasm では `id_min_depth = 2` に固定される。

## n-tuple の TD 目標: afterstate + max backup (#131, 2026-10-31)

現行の目標は「実際に選んだ手の次の afterstate の価値」＝on-policy なサンプル目標。文献（Szubert ら）の **max backup**
（出現後の盤面で `r + max_a V(afterstate(s', a))`）に変えて、同じ総ゲーム数で比較した。

`ntuple.Target = enum { on_policy, max_backup }`、`trainGameWith(net, rng, lr, mode, target)`（`trainGame` は
`trainGameWith(..., .on_policy)` のままで挙動不変。同じ seed で同じ表になることをテストで固定）。`.max_backup` の目標は
`p.reward + bestAfterValue(net, state)`（`state` は出現後の盤面）。variant 名 `maxb` / `maxbchance` / `maxbdeep`（`.wide`）。

20 局、評価 seed 2000、lr 0.01、`.wide`:

| 学習 | 1-ply | depth 2 | 学習時間 |
| --- | ---: | ---: | ---: |
| on-policy 1M, seed 1 | 13,795.0 | 39,344.0 | 554 s |
| on-policy 1M, seed 2 | 14,097.8 | 41,389.2 | 613 s |
| maxb 1M, seed 1 | **19,347.4** | 43,102.0 | 720 s |
| maxb 1M, seed 2 | 13,607.4 | 37,849.2 | 758 s |
| on-policy 5M, seed 1 | 15,777.6 | 50,103.2 | 4,197 s |
| **maxb 5M, seed 1** | **17,567.6** | **52,953.4** | 6,436 s |

1M の平均は 1-ply 13,946 → 16,477（+18%）、depth 2 40,367 → 40,476（±0%）。5M の同一 seed 比較は 1-ply +11%、depth 2 +5.7%
（seed 間のばらつき ±5% と同程度）。学習は 1.2〜1.5 倍遅い（`bestAfterValue` が 1 手あたり 4 回の表引きを足し、ゲームも長くなる）。

- **1-ply の価値は明確に良くなる**: サンプルされた手ではなく最善手の価値で更新するので、表が「今の方策」より一歩先を評価する。
- **2-ply 探索の利得は縮む**（maxb 5M は 52,953 / 17,568 = 3.0 倍、on-policy は 50,103 / 15,778 = 3.2 倍）。価値自体が先読みを
  含むぶん、追加の探索で得られるものが減る。
- 合計では depth 2 で +5.7% だが seed のばらつきに埋もれる。**既定の学習目標は変えない**（`maxb` 系は variant として残す）。
  学習時間あたりでは on-policy が有利。

`zig build ntuple -Doptimize=ReleaseFast -- 5000000 20 2000 0.01 maxb /tmp/table.bin 1`

## 既定 nn 重みの学習レシピ修正 (#133, 2026-10-31)

`#107` で `par` が実効学習率を約 1/8 に薄めると分かった（同一 100k・lr 0.003 で `par` あり 30,145 / なし 87,824）ので、
**`par` なしで 100k 学習した `nn`** を `expectimax_id` と同じプロトコル（100 局 = 10 ゲーム × 10 seed レンジ、seed 9000–9009、2-ply）
で測り、既定重みと比べた。

| 重み | 1-ply | 2-ply | avg moves | 出典 |
| --- | ---: | ---: | ---: | --- |
| 既定 `nn_weights.bin`（200k, 旧レシピ） | 41,611 ± 2,201 | **72,397 ± 832** | 3,282.6 | 再測 |
| 100k / par なし / train seed 1 | 36,562 ± 2,198 | **86,864 ± 2,503** | 3,865.8 | 本節 |
| 100k / par なし / train seed 2 | 39,711 ± 1,895 | **93,852 ± 3,173** | 4,119.0 | 本節 |
| `expectimax_id` 2 ms（#104） | — | 75,202 | 3,429.3 | #104 |
| `expectimax` depth 2（手作り評価、同条件） | — | 34,000 ± 1,616 | 1,745.1 | 再測 |

別の評価 seed（9100–9109）での追試: train seed 2 = 89,622 ± 1,674、train seed 1 = 81,540 ± 2,449。seed 2 が両方の評価で上なので採用。
2 seed 平均 **90,358**。

- **既定の `nn` / `nn2` は 2-ply で 90k 前後**になり `expectimax_id`（75,202）を上回り、しかも 0.57 games/s で 8 倍速い。
- 既定重みが悪いのは `par` ではなく**学習ゲーム数**。100k がこの recipe のピークで、400k まで回すと 73,007 に劣化する
  （自己対戦が短くなり学習信号が減る）。さらに延ばすなら lr 減衰か早期終了が要る。
- 評価は 2-ply（`nn.deepMove`）で leaves/move ≈ 1,040、0.41 ms/手。1-ply は 36k〜42k で伸びない。
- golden テストは 1 局だけなので `nn`（1-ply）の golden は 70,364 → 32,380 と下がったが、母集団の 1-ply は 41,611 → 39,711 でほぼ同等。
  **1 局の golden で優劣を判断しない**（`nn2` の golden は 73,964 → 77,164 に上がっている）。

再現: `zig build nn -Doptimize=ReleaseFast -- 100000 10 9000 0.003 10 base 30 ai/solver/nn_weights.bin 2`

## nn の学習を 100k より延ばせるか (#135, 2026-10-31)

`#133` で 100k / `par` なしが既定になったが、`ai/solver/nn_main.zig:220` を見ると**`par` なしの経路には既に線形減衰
（`lr * (1 - 0.9 * frac)`、100% → 10%）が入っている**。したがって「減衰なしで延ばすと劣化する」ではなく「減衰させても延ばすと
劣化する」が正しく、残るノブは減衰の形と fine tuning だった。train seed 3 で統一、評価は 100 ゲーム（10 × 10 レンジ、seed 9000–9009、2-ply）。

| 学習 | 1-ply | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: |
| 100k / lr 0.003（減衰 0.003 → 0.0003、対照） | 36,743 ± 1,489 | **81,296 ± 1,986** | 1,229 s |
| 200k / lr 0.003（減衰を 200k に引き伸ばす） | 39,633 ± 1,998 | **70,210 ± 1,213** | 2,677 s |
| 100k から +100k / lr 0.001（0.001 → 0.0001 で fine tuning） | 45,662 ± 2,285 | **74,841 ± 476** | 1,800 s |

- **100k がこの recipe の最適点**。減衰を引き伸ばして 200k 回すと −14%、低 lr で追化学習すると −8%。
- fine tuning は **1-ply を大きく上げる**（36,743 → 45,662, +24%）が **2-ply は下げる**。低 lr で追い込むと価値が「今の方策の
  先読み」に寄り、探索の上乗せが効かなくなる。1-ply だけを見て採用判断すると罠になる。
- 100k の train seed 間のばらつきは大きい（seed 1 = 86,864 / seed 2 = 93,852 / seed 3 = 81,296、平均 87,337 ± 6,300）。
  #133 で出荷した seed 2 の 93,852 は**上振れの側**なので、「既定 nn は 90k」は 1 seed の値であって期待値ではない。
  母集団としては 85k 前後と見るのが妥当。
- seed を選んで出荷している以上、この種の数値には選択バイアスが乗る。新しい候補を比べる時は同一 train seed で並べること。

## ranking 表の再生成と速度列（#136）

`src/web/ranking.json` は 2026-10-26 の生成物で、`latency_ms_per_move` / `games_per_sec` を出す前の形式だった。プレビューの表に
速度が無く、スコアも #133 出荷前（nn2 が 72,397）だった。`zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`（5 run × 20 ゲーム、
seed 1000）で作り直し、UI の表に games/s と ms/move の列を足した。生成物を `src/web/ranking.json` に置いたら**ビルドし直す**必要が
ある（`src/server.zig` が `@embedFile` している）。

| policy | depth | avg score | best tile | ms/move | games/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| nn2 | 2 | 82,772 | 8192 | 3.15 | 0.42 |
| nn | 0 | 69,717 | 8192 | 0.31 | 4.8 |
| expectimax_tt | 0 (内部 depth 3) | 55,636 | 8192 | 1.57 | 1.1 |
| expectimax | 2 | 33,218 | 4096 | 0.19 | 15 |
| mcts | 0 | 29,722 | 4096 | 1.67 | 1.8 |
| ga | 2 | 24,046 | 4096 | 0.23 | 16 |
| nnbeam | 0 | 21,048 | 4096 | 0.18 | 25 |
| beam | 0 | 20,440 | 2048 | 0.15 | 28 |
| nnmcts | 0 | 17,037 | 2048 | 4.72 | 1.0 |
| greedy | 0 | 7,789 | 1024 | 0.0007 | 9,891 |
| random | 0 | 1,062 | 512 | 0.0005 | 61,275 |

計測中は nn3 のベンチマークが 2 コアを占有していたので絶対速度は低め。表の中では同条件なので順序
（nn2 ≪ nn ≪ expectimax_tt ≪ expectimax）は信用できる。nn2 は 1 手 3.2 ms（expectimax の 17 倍遅い）でスコアは 2.4 倍。
速度なら nn、強さなら nn2。

## WASM 版はモデル選択をやめて nn2 固定（#143）

（方針は #152 で `nn3` に、速度は「常に最大」から select に変えた。以下の実測値はその時点のもの。）静的版は 13 方針と速度の
ドロップダウンを出していたが、ブラウザで選ばせる意味が薄いので外した。実測（同マシン、V8 上の wasm、1 手あたり、終盤盤面、
depth 2、単一スレッド）:

| policy | ms/move | native avg score |
| --- | ---: | ---: |
| `random` | 0.001 | 1,062 |
| `greedy` | 0.003 | 7,789 |
| `nn` | 0.13 | 69,717 |
| `nnbeam` | 0.12 | 21,048 |
| `ga` | 0.18 | 24,046 |
| `expectimax` | 0.37 | 33,218 |
| `beam` | 0.49 | 20,440 |
| `nn2` | 1.28 | 82,772 |
| `mcts` | 2.63 | 29,722 |
| `nnmcts` | 9.92 | 17,037 |
| `nn3` | 22.9 | 116,205 |
| `expectimax_id` | 深化が止まらず実質無限（下記） | 75,202 |
| `ntuple` | 同梱不可（表 64.6 MiB） | 20,014 |

- **`nn2` 固定**（1 手 1.3 ms で、自動再生でも 1 フレーム 8 ms に収まる）。速度優先なら `nn`、強さ優先なら `nn3`（22.9 ms /
  116,205）だが `nn3` はメインスレッドを掴むので既定にしない（#152 で既定に戻した。22.9 ms は終盤だけの値で序盤は 0.3〜0.4 s）。
- 方針は **index ではなく名前で解決**する（`algo_name` を走査して `"nn2"` を探し、無ければ `expectimax`）。`Policy` の並びが
  変わったときに黙って別の方針になるのを防ぐ。
- 自動再生は `requestAnimationFrame` ごとに 8 ms ぶん解いて返す。以前の "max (no delay)" は `while` で 1 ゲームを同期実行して
  いたのでタブが固まった。
- **`expectimax_id` はブラウザで壊れている**: freestanding の `clockNow` が常に 0 なので時間予算が効かず、`id_max_depth = 8` まで
  深化する。以前の記述（「深さ 2 で止まる」）は実装と食い違っていた。選択肢から外したので実害は無いが、WASM で出すなら時計か
  ノード上限が要る。
- WASM の制約（実測）: モジュール 763,545 B、初期メモリ 2.44 MiB（`memory.grow` 無し）、スタック 1 MiB、TT 640 KiB + `cached_net`
  148 KB が bss。スレッドは使わない。

## 静的版を nn3 に戻し、再生速度を選べるようにした（#152）

（#154 で方針 select が戻り、同梱は `nn2` と `nn3` の 2 つになった。以下の「方針は固定」「smoke test は `SOLVER_NAME` を見る」は
#152 時点の記録。）最大速度では 1 秒に何十手も進んで目で追えないので、既定を `nn3`（native 116,205、`nn2` は 82,772）にし、
自動再生の速度を select で選べるようにした（当初 `max` / 250 ms / 500 ms / 1 s / 2 s、既定 `max`。1 秒以上は使わないので #154 で
1 s / 2 s を外した）。

wasm（ReleaseFast、V8）での 1 手あたり。`nn3` のコストは盤面に強く依存し、#143 の 22.9 ms は終盤だけの値:

| 局面 | 手数 | `nn2` | `nn3` |
| --- | ---: | ---: | ---: |
| 開始直後 | 0 | 14 ms | 270 ms |
| 256 タイル | 250 | 9 ms | 390 ms |
| 2048 タイル | 1,000 | 9 ms | 400 ms |
| 4096 タイル | 2,500 | — | 4 ms |
| 4096 タイル（約 3,500 手） | 3,500 | — | 80 ms |

序盤は空きマスが多く分岐が最大なので 0.3〜0.4 s、詰まると 10 ms 前後。1 ゲーム 4,000 手前後なので `nn3` の自動再生は数分〜十数分。

- 速度は「1 手あたりの待ち時間」。`max` は 1 フレーム 8 ms の予算で回すが、`nn3` は 1 手で予算を超えるので実質 1 フレーム 1 手。
- 待ち時間は *solve が返ってから* 測り直す（solve 自体が 0.3 s の手に 250 ms を上乗せして 2 倍遅くしないため）。
- `nn3` の深さは `nn.bestMove(net, state, .deep3)` で固定なので、`solve` の depth はフォールバック（`expectimax`）用。
- 自動再生のループは 1 本だけ。`stopAuto` が保留中の `requestAnimationFrame` を `cancelAnimationFrame` するので、停止→開始が
  1 タスク内で起きても 2 本目が生まれない（nn3 の solve が 0.3 s 固まる間に押されると起きていた）。
- 速度の `<select>` にフォーカスがあるときは矢印キーを盤面に使わない（以前は select の上下キーが `preventDefault` されて手になった。
  サーバー版の `src/web/index.html` も同じ）。
- `scripts/wasm-smoke.mjs` は「静的版の方針が wasm に載っている」ことを検査する。当時はページ側の定数 `SOLVER_NAME` を見ていたが、
  #154 でページは方針名をモジュールから取るようになり、検査は同梱一覧と既定（`DEFAULT_POLICY`）の突き合わせに置き換わった。

## 同梱する方針を設定ファイルで決め、nn2 / nn3 の 2 択にした（#154）

#143 で 13 方針のドロップダウンを外したが、#152 で `nn3` を既定にすると「速すぎて何が起きているか分からない」局面が出たので
方針 select を戻した。13 方針を全部載せる理由は無いので、**同梱する方針はコードではなく設定ファイル** `wasm.policies` に持たせた
（1 行 1 名、`#` 以降はコメント）。既定は `nn2` と `nn3`。

- `build.zig` が `@embedFile("wasm.policies")` でそのまま `options` モジュールに渡し、`src/wasm.zig` が comptime で解析する。
  未知の名前・重複・n-tuple 表が要る `ntuple` はビルドエラー。ファイルを編集すると wasm が再ビルドされる。
- 同梱した方針だけをリンクするため、`solver.choose` を `chooseOne(comptime policy, ...)` に分け、`solve` は
  `inline 0...policies.len - 1` でインデックスごとに定数方針を渡す。実行時の `switch (policy)` は全アームをリンクしてしまう。
- ABI の `algo` は「モジュールに入れた順」の index になった。`Policy` の並びは ABI ではなくなり、`wasm_policy_count`
  （"表が要る方針は最後" を強制する仕掛け）は消した。
- ページの方針 select は `algo_count()` / `algo_name()` から組み立て、選んだ方針の**測った平均スコアと 1 手の時間**を
  `algo_score()` / `algo_speed()`（`policy_info` の `score` と `speed`）で select の下に出す。実装の説明（`description`）は
  ページには長すぎるので出さない（#155 の指摘）。`speed` はブラウザ実測（V8 on wasm）で native の ms/move とは別物なので、
  測っていない方針は空にしページはスコアだけを出す。ページが自分で持つ方針名は既定を指す `DEFAULT_POLICY = "nn3"` の 1 つだけで、
  同梱していない方針は選べず、UI とモジュールの一覧はずれない（既定が同梱に無いときだけ最後のエントリに落ちる）。
- `scripts/wasm-smoke.mjs` は `wasm.policies` を読み、モジュールの一覧が名前も順序も一致すること、各エントリに説明があること、
  ビルド済み HTML が `algo_count()` で一覧を作っていること、その `DEFAULT_POLICY` が同梱一覧に含まれることを検査する
  （最後は、既定を `wasm.policies` から外したときにページが黙って別の方針になるのを防ぐ）。`scripts/deploy-pages.sh` は
  デプロイログに同梱方針を出す（インラインコメントは落とす。`--skip-smoke` のときは「モジュールと設定ファイルの突き合わせを
  していない」と警告する）。

サイズの実測（ReleaseFast、`zig build wasm`、`wc -c`）:

| 同梱した方針 | バイト |
| --- | ---: |
| 13 方針すべて（#152 まで） | 764,897 |
| `nn2` + `nn3`（既定） | 701,728（説明文をページに出さなくして -298 B、`speed` で +142 B） |
| `nn3` のみ | 701,167 |
| `nn2` のみ | 701,107 |
| NN 以外の 8 方針 | 591,768 |
| `random` のみ（床） | 529,463 |

13 → 2 で 63 KB（8%）しか減らない。`nn2` と `nn3` は同じ net と探索コードを共有していて、2 つ目は 700 B しか増えないため。
方針を削る効果はここで頭打ちになる。

### レビューで見つかって直したこと（#154）

- `solve(algo, depth)` の `depth` を捨てていた。同梱が `nn2`/`nn3` の間は害が無い（両方とも深さ固定）が、`wasm.policies` に
  `expectimax` か `ga` を足すと `depth = 0` が `chanceValueW` で `0 - 1` の u8 ラップで 255 手になり、タブも smoke テストも
  戻ってこない。`depth` を通し、`0` は `policy_info` の既定深さ（`expectimax`/`ga` は 2）を使う、という `expectimax_id` と同じ
  約束にした。`expectimax` を足したビルドで `solve(0, 0)` が 1 手で返ることを確認済み。
- パーサが `nn2#コメント` の名前を捨てていた（`#` の分岐が区切りの処理より先にあった）。区切りとコメントの前で必ず名前を
  確定させ、先頭 BOM も読み飛ばし、`found[32]` の上限も `@compileError` にした。
- smoke テストの 2 s 上限は呼び出しが返ってからしか見ないのでハングを検出できなかった（#143/#145 の保護は実は「depth 引数」）。
  方針ごとに `--solve-one` の子プロセスで回し、5 s で `spawnSync` が kill するようにした。
- `src/wasm.zig` の `depth` ABI 説明と `expectimax_id` の説明（「既定の40ms」→ 実際は `id_budget_ms`）が実装とずれていた。
- 内訳（既定ビルド）: code 11,787 B / data 689,801 B。data の大半は `src/core.zig` の `row_table`（`[65536]RowEntry` = 512 KiB）。
  次にサイズを削るならここ（`RowEntry` は 8 B。score を半分にして u32 に詰めれば 256 KiB 減る）で、方針を削っても効かない。
## 学習の両側を 2-ply に揃える（#158）

評価を 1-ply で学習し探索で 2-ply を使うと、学習目標（サンプルされた手の価値）と方策（最善手）がずれる。
`Options.teacher`（`pickTeacher`）と `Options.target`（`tdBootstrap` の分岐）で両側を独立に切り替えられるようにした。

| セル | 目標 | 手を選ぶ基準 |
| --- | --- | --- |
| `base`（対照） | `reward + V(next afterstate)` | 1-ply argmax |
| `t` | `search2`（2-ply 探索値） | 1-ply |
| `p` | TD | `.deep` |
| `tp` | `search2` | `.deep` |

どのセルも spawn を列挙するので乱数の消費は `base` と同じ（ゲームの spawn 列が揃う）。

**目標の二重計上（レビューで修正）**: 最初の `search2` は `Search.value(spawned, 2, 1)` で、`value` の葉が
「最後の報酬を足した後の afterstate」を評価するため `V` に次の報酬が二重に入っていた（決定的な鎖で
固定点が真値の約 1.5 倍。`test "search2 target does not double count the last reward (#158)"` が差を 8/reward_scale
として固定）。`Search.bootValue(state, rewards, cprob)`（同 `bootChance`）を足して葉を 1 手先へずらし、
`search2Value` は `r_t + r_{t+1} + V(A_{t+2})` を返す。`Search.ply` を `search2_ply + 1` にして置換表のキーも
葉の深さに合わせた（値自体は深さ非依存）。

## 深さを空き数で切り替えてもコストは下がらない（#157）

`deep`（2-ply）/ `deep3`（3-ply, cutoff 1e-2）/ `phase_h1..h3`（空き数で深さを混ぜる）を、同梱 net
（md5 `f584d99667f499184d7da2d5d0e9e0a1`）で 3 局（seed 9000-9002）測った。

| 行 | avg score | ms/move | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: |
| 1-ply | 60,980 | 0.00 | 2/3 | 0/3 |
| +1 spawn lookahead | 64,479 | 0.08 | 2/3 | 0/3 |
| 2-ply (deep) | 69,272 | 0.72 | 3/3 | 0/3 |
| expectimax depth 2 | 23,335 | 0.05 | 0/3 | 0/3 |
| 3-ply (deep3) | 100,728 | 16.58 | 2/3 | 2/3 |
| phase h1 / h2 / h3 | 93,912 / 110,843 / 108,901 | 21.4 / 26.4 / 29.6 | 3/3 | 1/3 / 1/3 / 2/3 |

**空き数で深さを切り替えてもコストは下がらない**（h1-h3 は deep3 より高い）。1 局の手数の大半が空き 4-7 で
消費され、そこを 3-ply にすると一律 3-ply と同じだけ読むため。序盤だけ 2-ply にした分は安いが、全体は深い
局面の 3-ply が支配する。到達率は deep3 と h3 が 8192 に 2/3、deep は 0/3。n=3 なので avg score の差は
判断材料にしない。

**計測事故（同じ轍を踏まないため）**: 最初の計測は作業ツリーの `ai/solver/nn_weights.bin` が #133 以前の
200k+40k net（md5 `0a8e6bf01e256adf2fc481317c5072c9`）に入れ替わっていて全行 4-14 倍低く出た。コードは壊れて
おらず golden（`ai/solver/solver.zig:1865-1879`、`nn` seed 20240601 = 32,380 / `nn2` = 77,164）は同梱 net で緑。
**評価の前に md5 を控え、`/tmp` のコピーを渡す。**

## 終盤だけ 4-ply（#171）

`phase_schedules` に `phase_h4` を足した。中盤は `deep3` のままで、空きマスが 2 以下になってから 4-ply に
する。終盤は分岐が潰れて 1 ply 増やすコストが小さく（終盤 1 手 4 ms 前後、遅くても ~80 ms）、4096 → 8192 の
判断がその数手で決まるため。cutoff は `deep3` と同じ 1e-2（空き 2 では spawn の分岐が数個しかなく木が小さい）。

3 局（seed 9000-9002）: `deep3` 100,728 / 21.4 ms/move / 4096 2/3 → `phase_h4` **131,201** / 31.7 ms/move / 4096 3/3。
**上乗せは約 +48%**、avg score +30%、手数 +28%、8192 は 2/3 のまま。n=3 なので avg score は参考値。

10 局（seed 9000-9009、同梱 net）:

| 方針 | avg score | moves | ms/move | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-ply | 50,496 | 2,384 | 0.00 | 6/10 | 0/10 |
| 2-ply (deep) | 86,565 | 3,828 | 0.58 | **10/10** | 3/10 |
| 3-ply (deep3) | 106,936 | 4,567 | 12.04 | 8/10 | **6/10** |
| phase h4 (#171) | **117,566** | 5,016 | 22.88 | **10/10** | **6/10** |

- **4096 は 2-ply で 10/10**（1 手 0.58 ms）。3-ply の 8/10 は 2 局が 4096 直後で死んだノイズ。
- **8192 は 2-ply 3/10、3-ply 6/10、phase_h4 も 6/10 で上がらない**。一方 avg score は +10%、手数 +10%、
  コスト 1.9 倍。「終盤を深く読むと、8192 に届くかは変わらないが、届いたときの伸びが良くなる」。
- **8192 到達率を動かすには探索の深さではなく評価関数そのもの**（3 段の差はすべて V の質）を良くする必要が
  ある。#167 / #175 の蒸留が本命。

## 深い探索を教師にする蒸留（#167）

3-ply で自己対戦し、目標をその探索値にして、遊ぶときは 2-ply の評価 1 枚で済ませる。

- **目標**: `tdBootstrap` が `teacherValue(vfn, spawned, mode)` = `Search.value(spawned, ply, 1)` を返す
  （`spawned` は指した手の後の spawn 済み盤面、`ply` はモードの位相行の深さ）。`.deep3` は `ply = 3` /
  `cutoff = 1e-2`、`.deep` は #158 の `search2` と同じ木の `value(S, 2)`。
- **方策**: `Options.teacher` を立てると自己対戦も同じモードの探索で指す（`pickTeacherValue`:
  `reward + Search.chance(after, ply, 1)`）。方策と目標が別の木を見ると #135 の再現になるので両側を同じ木から取る。
- セル: `d2`（教師 `.deep`、目標 `value(spawned, 2)`、自己対戦 `.deep` の手）/ `d3`（`.deep3`、`value(spawned, 3)`、
  `.deep3` の手）。既定は不変（`teacher = null`）なので `base` / `t2` / `p2` / `tp2` の checksum はそのまま。
- **コスト**: `d3` の 1 ゲームは `tp2` の 15 倍前後（3-ply は 15-23 ms/手、2-ply は 1.2 ms/手）。`par` は `d2` / `d3`
  では弾く（教師探索が実行時間のほぼ全部）。比較は遊ぶ側を 2-ply で測る。**数値は未取得。**

## 追加学習では 8192 到達率が動かない（#159）

学習は 1 ゲームずつ順に回すので 1 コア。`par`（第 10 引数）は `min(cpu 数, 8)` ゲームを同時に進め、この機械
（8 論理 / 4 物理）で 2,000 ゲームが 31.8 s → 9.1 s（3.5 倍）。`par` は `mstd` / `dist` / `dson` / `dump` 系 /
#158 の 2-ply 系を弾く。`replay_k` は無視しない。

同梱 net から 5,000 ゲーム追加学習、評価 20 局（seed 9000-9019、2-ply）、基準 79,657 / 8192 4/20:
`par` lr 3e-4 86,064（seed 1）/ 77,760（seed 5）、lr 5e-4 93,126 / 81,661 / 91,066 / 88,062（seed 1-4）、
直列 + `replay_k=30` lr 3e-4 93,318 / 84,440 / 85,971、lr 5e-4 82,560。ゲーム数を 10k / 20k / 50k に増やすと
81.8k / 90.1k / 87.2k。蒸留（#175）は同予算で score を壊す（58.4k-63.6k）か浅い局面の価値を崩す（`dson` は
1-ply が 5k まで落ちる）。**自己対戦の追加学習では 8192 は動かない**（+8% は seed ばらつき ±5k と同程度）。

> 訂正（#208）: 上の `par` 行は `batch_scale = 1.0`（8 ゲーム平均 = 直列の 1/8 の更新回数）で回した「弱めの
> 学習」。同じ 20 局で測り直すと batch_scale 8 は直列より +17.3k ± 4.5k（18/20）。表中の score は当時の実測値
> なので残す。

## 行テーブルを 1 エントリ 3 バイトに詰める（#160）

wasm の data（689,801 B）のうち 512 KiB は `src/core.zig` の `row_table`（`[65536]RowEntry`、8 B）だった。
`[65536][3]u8` に詰める: bits 0..15 = マージ後の行（4 ニブル、0 は空）、bits 16..19 / 20..23 = 1 個目 / 2 個目の
合体タイルの指数。`score_delta` は 2 つの指数から `1 << exp` の和として厳密に戻る（4 セルは最大 2 回しか合体せず、
その 2 つは隣接同値ペアなので重複しない）。指数 16（32768 超）はニブルに入らないので `row_overflow`
（`[65536 / 8]u8` のビットセット）に記録し、`mergeLine` は従来の `ok = false` と同じく `mergeLineSlow` に落とす。

`row_table` 524,288 → 196,608 + `row_overflow` 8,192 = 204,800（data は -319,488 B、-31% の見込み）。
全 65,536 キーで `mergeLine` と `mergeLineSlow` の `line` / `changed` / `score_delta` が一致すること、ビットセットが
指数 16 のキーとちょうど一致することをテストで固定。`MergeResult` と `mergeLine` のシグネチャは変えていない。

## 共通乱数でのペア比較（#168）

同じ設定でも 20 局で ±5,000 ばらつくので、片方の出現列をテープに記録してもう片方に食わせる。

- `nn.Spawn`（cell 8 bit + 指数 8 bit）/ `nn.Tape`（`Recording = enum { record, replay }`）
- `playTapedValue` / `playTapedWithStatsValue`: record は平文と同じ 2 回の rng draw（セル → 値）を記録し、
  replay は記録を置くだけで rng を引かない。分岐して記録セルが埋まっていれば最初の空きセルに置き、テープが
  尽きたら通常抽選に戻る。record の結果は平文 `playValue` と盤面まで一致（テスト）。
- `nn.pairStats(diffs)` → `PairSummary { games, mean, stderr, better, worse }`。`pair` モードで `deep`(A) と
  `deep3`(B) を 1 局ずつ比較し、最後に平均差 ± stderr を出す。
- 分岐後の出現位置は記録と食い違うので、分岐前（序盤）の比較が最も鋭い。それでも独立 2 本よりずっと安定する。

## 並列学習の実効学習率（#137 `batch_scale`）

`par` はゲームごとの差分を平均するので更新回数が 1/8 になり実効学習率が薄まる（#133: 同一 100k・lr 0.003 で
`par` 30,145 / 直列 87,824）。arg 12 で倍率を渡せるようにした（`Options.batch_scale`、8 = linear 則 / 2.8284 = sqrt 則）。

同梱 net から 5,000 ゲーム、`par`、lr 3e-4、train seed 1、評価 20 局（2-ply）: 基準 79,657 → 1.0 **91,497**（6/20）/
2.8284 **97,214**（7/20）/ 8 **94,834**（6/20）。直列 + `replay_k=30` は 93,318（6/20）。sqrt 則が一番良い方向だが
seed ばらつき ±5,000 では 1 回で決められない → #168 で分散を落としてから確定。`Options.batch_scale` の既定は
1.0 のままで、#208 で CLI の既定だけバッチ数（linear 則）に変えた。

評価メモ: 評価行は絞り込めないので、行テーブルを 2 行（1-ply と 2-ply）に書き換えて回した。9 行構成では 20 局で 20 分以上。

## 蒸留の混合比と教師 lr（#206）

`#175` の `dist` は「浅い局面の TD」と「深い局面の教師回帰」を 1:1 で混ぜ、それが効かないことは実測済み
（5,000 ゲームで 58,361-63,633、基準 79,657）。原因は共有パラメータを深い局面の信号が引っ張ること。
ノブを 2 つ足した: `distill_every`（arg 12、1 = 毎手）と `distill_lr` の割り算（arg 13、0 = TD と同じ）。
`dson` にも同じ 2 つが効く。例: `... dist 30 /tmp/w.bin 1 - /tmp/deep2.dmp.tch 8 10`。**数値は未取得。**

## オフライン蒸留（#175 / `tch`）

`#167` は学習中の全手を 3-ply で値付けするので 30-46 ms/move、100k ゲームで 35-40 時間/seed。`#159` の dump
（最大タイル 4096 以上の局面を 65,536 個保存）を**一度だけ** 3-ply で値付けすれば 10-20 分で済み、その後の回帰は
通常速度（~0.5 ms/move）で回る。

    1. dump:  1000 ゲーム回して深い局面を貯める（1-ply 自己対戦 = dist の学習側と同じ分布）
    2. tch:   dump 1 局面につき spawn を 1 回引き、post-spawn 盤面を teacherValue(.deep3) で値付け
    3. dist:  自己対戦 TD と 1/2 ずつ混ぜて回帰

保存する目標は spawn 期待値の 1 サンプル推定（1 盤面 1 回の回帰が表現できるのもそこまでなので意図的）。`dist` は 1 手
ごとに確率 1/2 で教師ペアを一様抽選して同じ勾配経路で回帰する（深い局面にしか無いデータで序盤が壊れないように）。
**数値は未取得。**

## 重みの量子化（#170）

f32 形式（148,504 B）は学習・golden・web のすべてが前提にしているので触らず、**第 2 の形式**として int16 を足す。
既定の評価経路は f32 のまま。

| 位置 | 内容 |
| --- | --- |
| 0 | magic `2048-NQ1`（f32 形式の `2048-NN\0` とは別） |
| 8 | u32 version = 1 / u32 endian marker = `0x01020304` |
| 16 | f32 step x 6（ブロック `w1 b1 w2 b2 w3 b3` 順） |
| 40 | f32 step x 2（活性化 `a1 a2`） |
| 48 | i16 x 18,561（`Net` のフィールド順） |

`b3` だけ別ブロックなのは max|w| が 20.39 と `w3`（1.57）の 13 倍あり、同居させると `w3` の分解能が 13 倍悪く
なるため。量子化は `step = max|w| / 32767`、`q = round(w / step)`（32767 でクリップ）。`QPayload` はフィールドを
1 対 1 で写した `extern struct` なのでペイロードをオフセットごと 2 B 幅で読める。**148,504 → 37,170 B（25.0%）。**

`forwardQuantized` は `forward` と同じ形で、隠れ層の入力は i16 x i16 を i64 で合算する（`a2` の step が 3.2e-2 と
粗く、i32 では実際に溢れる）。層の境界で f32 に戻し、ReLU のあと次の層用に再量子化する。活性化の step は
「その層の最大値 / 32767」: 層 1 は `b1[i]` + 列 `i` の正の重み上位 16 個（1 枚が光らせる行は 16 個まで）で 38.78、
層 2 は 1037.4（`a1 = 1.184e-3` / `a2 = 3.166e-2`）。全正重みを足す素朴な上界だと誤差が 2.4 倍になる。

`zig build nn-quant -- ai/solver/nn_weights.bin /tmp/nn_weights_q.bin [games] [seed]` で 3 つ測る（`nn_weights.bin`
は読むだけで上書きしない、量子化ファイルはコミットしない）:

- **round trip**: 最悪 `1.026e-4` = step の 0.500 倍（`round` の理論値どおり）。
- **値の一致**（`games=60, seed=7` の 124,879 盤面）: 重みだけ量子化 mean 5.8e-4 / max 5.1e-3、活性化まで
  int16 で mean 1.0e-2 / max 6.9e-2（値域 −26〜+37 なので相対 3e-4 程度）。
- **対局**: 39,326（f32）/ 41,076（int16）、どちらも最大 4096。20 局だと 20% 開くが差は再現しない。
### 量子化形式の接続（#170 の続き）

`nn.Value` に `.quantized` を足した。`valueAt` が `forwardQuantized` に回すだけで、探索・再生のコードは `Value`
しか見ていないので `nn.playValue(.{ .quantized = &q }, rng, .greedy)` がそのまま動く。既定は `.{ .single = &net }`。
wasm の埋め込みを量子化形式に切り替える作業はしていない（web は `solver.zig` の `@embedFile("nn_weights.bin")`
経由で f32 のまま）。誤差の大半は `a2` の step なので、層 2 を int16 のままにするなら step を層ごとに詰めるか、
層 3 の入力を int8 にする余地がある。

## `par` の move 数と実効学習率を直す（#208）

`par` の要約行が `moves 0` を出していたのは集計漏れ（`use_par` の分岐が `total_moves` に足しておらず、`par` は
最後の 1 ゲームの score しか返していなかった）。`nn.Batch`（`{ score: u32, moves: u64 }`、`sumScores` / `sumMoves`）
を返すようにして直列と同じ moves を報告する。

残る差は #107 の実効学習率の希釈（8 ゲームの差分を平均して 1 ステップ = 更新回数 1/8）。#137 の `batch_scale` の
既定が 1.0 だったので linear 則（倍率 = バッチ数）を既定にしたが、#203 の 100,000 ゲームで崩れたので**今の既定は
sqrt 則（2.8284）**。linear 則は `8` を明示し、`-` で従来の 1.0 に戻せる。

`base` から 20,000 ゲーム、lr 0.003、train seed 1、`pairw` 20 局（A = 直列、B = `par`）:

| `batch_scale` | 学習時間 | moves | B − A（± stderr） | B が勝った局 |
| --- | ---: | ---: | --- | --- |
| 直列 | 112.4 s | 13,346,693 | — | — |
| 1（平均） | 16.7 s | 6,799,634 | −16,552 ± 4,595 | 2/20 |
| 2.8284（sqrt） | 23.8 s | 9,384,508 | −16,759 ± 4,191 | 4/20 |
| 8（linear） | 31.5 s | 11,784,409 | **+17,298 ± 4,474** | 18/20 |

linear 則なら `par` は直列同等以上（3.6 倍速い）。sqrt 則と 1.0 は同じだけ負ける（差は有意）。moves が直列より
少ないのは学習途中の net が弱くゲームが早く終わるため（linear 則で 88% まで戻る）。

同梱 net からの微調整（5,000 ゲーム、lr 3e-4、`replay_k=30`、train seed 1）では差が小さい: 直列 −2,920 ± 10,109
（8/20）、`par` linear −13,766 ± 6,987（6/20）、`par` sqrt −15,037 ± 7,352（7/20）。直列と `par` linear を直接比べると
−3,546 ± 7,677。**短い微調整では `par` の優位は出ない**（微調整は直列、20k ゲーム以上の長い学習は `par` でよい）。

`eval_games = 0`（第 2 引数）は「学習だけ」で 15 行の評価表を飛ばす（以前は usage エラーで、1 局 20 分かかる行も
必ず払っていた）。

## 同梱 net を `base` から焼き直す（#203）

`par`（`batch_size = 8`）、lr 0.003、`replay_k=30`、100,000 ゲームを 3 seed。**linear 則は 100k では崩れる**
（実効 lr 8 倍なので 20k までは直列に勝てたが、そこから overstep する）。A = 同梱 net、`pairw` 20 局（seed 9000、2-ply）:

| 学習 | seed | B − A（± stderr） | B が勝った局 |
| --- | ---: | --- | --- |
| `par` linear | 1 | −56,038 ± 6,322 | 0/20 |
| `par` linear | 2 | −26,379 ± 9,668 | 4/20 |
| `par` linear | 3 | −67,751 ± 6,353 | 0/20 |
| `par` sqrt | 1 | −1,440 ± 7,544 | 13/20 |
| `par` sqrt | 2 | −26,547 ± 5,983 | 2/20 |
| `par` sqrt | 3 | −25,328 ± 7,190 | 2/20 |

sqrt 則の 3 seed 平均は −17,772 ± 4,008。**同梱 net を上回る seed は無い**ので重みは差し替えない。linear 則の崩れは
lr でも戻らない（100k、seed 1: lr 0.003 → 33,878、0.001 → 50,956、0.000375 → 16,017。seed 2 の 0.003 は 73,944 なので
崩れ方に seed 依存が大きい）。

**ペア比較と非ペア表は食い違う。** 非ペア 20 局（`.deep`、seed 9000 / 9100）では sqrt 則 seed 1 が 82,600 / 81,813 で
同梱 net（79,657 / 91,348）を上回って見えるが、ペアでは −1,440 ± 7,544 で互角。**重みの判定はペア比較だけで行う。**
非ペア表: linear seed 1 33,878 / seed 2 73,944 / seed 3 22,499、sqrt seed 1 82,600 / seed 2 71,598 / seed 3 81,082
（8192 到達は 3 以下が多く 0/20）。

`rows` 引数で評価表の行を選べる（`0` = 1-ply、`2` = 2-ply。3-ply 行は 20 局で数十分かかるので測定では落とす）。

## 蒸留の学習率バグ、`ft2` と並列蒸留の否定（#207）

`dist` / `dson`（#175）が収穫を出さなかったのは**学習率の単位の取り違え**。`nn.Options.distill_lr` は絶対値なのに
`nn_main.zig` は `1 / div`（div 10 なら 0.1）を入れていた。TD が lr 0.003 の隣で教師ステップだけ lr 0.1、つまり
30〜300 倍のステップで回り、どの lr でも NaN に飛んでいた。修正は `distillRate(learning_rate, div) = learning_rate / div`。

修正後、同梱 net から 5,000 ゲーム（lr 3e-4、教師 65,536 サンプル）: every 8 / div 10（lr 3e-5）82,973 / 8192 5/20 /
−4,441 ± 8,391（9/20）、every 1 / div 10 61,589 / 0/20 / −32,359 ± 6,553（1/20）。50/50 は教師に引きずられて自己対戦の
半分が消える。20,000 ゲーム・教師 458,752 サンプル（7 seed の dump を `tchadd` で追記）: every 4 / div 3 が
**+1,184 ± 9,370**（9/20）、every 8 / div 10 −12,665 / −19,137、every 4 / div 10 −14,937。**採用バー +15,000（2σ）には
届かない** — 蒸留は同梱 net を「壊さずに保つ」ところまで。

**`ft2` は直せない。** 深い側の教師を自己参照（`pickMove` の値）から `deep3`（`tch` と同じ関数）に差し替えても発散する
（lr 3e-5〜1e-6、deep_every 1 でも 8 でも NaN 1,250〜2,647 個、max 1e89〜1e201）。原因は**教師が固定でないこと**:
`teacherValue` は葉を同じ net の `V` で打ち切るので深い局面の `V` が上がると教師も上がる（`FT2BAD out=1074.1
target=1198.7 wmax=20.82 tile=4096` のように 4096/8192 で `V` が 1,000 台まで育つ）。**`ft2` は非推奨**、同じことを
するならラベルを 1 回だけ固定する `tch` / `tchadd` → `dist` を使う（`dist` が安定なのは教師ファイルが学習中に変わら
ないからで、偶然ではなく設計の差）。

**`par` + `dist` も駄目**（20,000 ゲーム、every 4 / div 3、A = 同梱 net）: 直列 +1,184 ± 9,370（9/20、2,300 s）に対し
`par` mean −31,273 ± 6,653（0/20、90 s）/ √8 −34,859（0/20）/ 8 −30,117（1/20、53 s）。25 倍速いがどのスケールでも
30,000 点負けるので CLI でも拒否する。バッチは 8 ゲーム分の「スナップショットからの差分」を平均するのに教師サンプルは
ゲームごとに違うので、平均で教師信号が薄まるのだと思われる（未検証）。

`opts.batch_scale` が `par` の枝でしか設定されておらず、`sym` / `replay` を `par` で回すと常に 1.0 になっていたのも直した。

## lr スケジュールで 200k〜400k まで伸ばす（#204）

`#133` の「200k で −14%、400k で崩壊」は lr 固定の話で、`batch_scale` 導入後の学習器では測っていなかった。`base` から
seed 1、`replay_k=30`、`par`（sqrt 則）で 4 構成（`resume` を繋ぐだけでスケジュールになる）:

| 構成 | 段 | 学習時間 | moves |
| --- | --- | ---: | ---: |
| 固定 lr | 0.003 × 200k | 689 s | 263,788,497 |
| A 減衰引き伸ばし | 0.003 × 100k → 0.001 × 100k | 795 s | 282,690,191 |
| B 50k ごとに半減 | 0.003 × 100k → 0.0015 → 0.00075 → 0.000375 → 0.0001875 | 1,440 s | 490,473,958 |
| C 1/4 ずつ 2 段 | 0.003 × 100k → 0.00075 × 150k → 0.0001875 × 150k | 2,080 s | 699,065,146 |

ペア比較（20 局、A = 同梱 net）: 固定 200k −11,260 ± 6,566（7/20）、A −11,076 ± 7,912（3/20）、B 300k −12,723 ± 6,019
（4/20）、C 400k −9,385 ± 6,870（5/20）。非ペア（`.deep`、9000 / 9100）: 固定 200k 77,095 / 72,057、A 75,236 / 75,219、
B 83,504 / 80,619、C 76,246 / 73,671。

**崩壊は 400k まで再現しない。** 固定 lr 200k でも −11,260 で「−14%」ほど落ちず、200k / 300k / 400k のペア差は
−9.4k〜−12.7k で横ばい（100k から先はゲーム数を増やしても差が縮まないが、大きく開きもしない）。**スケジュールの優劣は
20 局では決まらない**（非ペアは B、ペアは C で順位が一致しない）。1 seed ずつなので選ぶなら 3 seed 必要。同梱 net を
上回る構成は 4 つとも無く、100k の net（#203、−1,440 ± 7,544）が最も近い。**同梱 net は 100k 規模で頭打ち**で、
次の一手はゲーム数ではなく教師（#207）。

## 蒸留の配合・教師の深さ・ゲーム数（#207 の続き）

すべて A = 同梱 net、`pairw` 20 局 seed 9000、lr 3e-4、学習 seed 1、教師は 7 dump 統合（458,752 サンプル、`deep3`）:

| 蒸留 | 教師 | B − A（± stderr） | better |
| --- | --- | --- | ---: |
| 20k、every 4 / div 3 | `deep3` 458,752 | **+1,184 ± 9,370** | 9/20 |
| 20k、every 4 / div 2 | `deep3` 458,752 | −16,585 ± 11,599 | 5/20 |
| 20k、every 2 / div 3 | `deep3` 458,752 | −20,110 ± 6,303 | 2/20 |
| 20k、every 6 / div 3 | `deep3` 458,752 | −30,848 ± 5,586 | 1/20 |
| 20k、every 4 / div 3 | `h4` 589,824 | −15,588 ± 6,249 | 6/20 |
| 100k、every 4 / div 3 | `deep3` 458,752 | −4,696 ± 8,005 | 5/20 |

配合が互角なのは 1 点だけで両側は −17k〜−31k（20 局の stderr 6〜12k ではほぼノイズ、+1,184 も再現値ではない）。
**教師を深くしても効かない**（`tch` に教師モード arg 12 を足し、`h4` で 9 dump を貼り直し。4.0 ms/sample 対 `deep3` の
1.5 ms/sample、ラベル差 mean|Δ| 6.4 / max 44.4。589,824 サンプルで −15,588 と 458,752 の `deep3` より悪い）。
**ゲーム数も効かない**（100,000 ゲーム 1,788.7 s / 191,593,848 moves で −4,696 ± 8,005、20k の +1,184 と同じ帯で飽和）。

**結論**: 教師ラベルを深くする・増やす・蒸留を長くする、のどれも同梱 net を超えない。蒸留は「壊さない」ための道具で、
伸ばす道具ではない。次に試す価値があるのは教師の中身そのもの（探索の目的関数、局面表現、ネット構造）で配合の調整ではない。

## ネットの幅を広げる（#220）

ゲーム数（#203 / #204）、lr スケジュール（#204）、蒸留（#207）が否定されたので容量を振った。`h1` / `h2` は #205 で
build option（`-Dh1=128 -Dh2=128`）。同梱 net は 64/32（148,504 B）。`base` からスクラッチ、lr 0.003、seed 1、`par`
sqrt 既定、評価は 2-ply 20 局（seed 9000）:

| 幅 (h1/h2) | ゲーム数 | 2-ply | 8192 | ms/move | 1 局のコスト |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/32（同梱） | — | 79,657.0 | 4/20 | 0.48 | 1.0x |
| 128/128 | 100,000 | 72,012.4 | 2/20 | 3.02 | 6.3x |
| 128/128 | 300,000 | **92,952.6** | **7/20** | 5.81 | 12x |
| 128/128 | 300,000 + lr 3e-4 で 300,000 | 84,118.2 | 2/20 | 5.18 | 11x |
| 128/32 | 300,000 | 78,499.8 | 3/20 | 1.23 | 2.6x |

**容量はゲーム数を食う**（128/128 でも 100k では 72,012 で負ける）。300k で初めて 92,953（+13,296）、8192 は 7/20。
同じ 300k でも 128/32 は 78,500 なので**効いているのは `h2` の広さ**で、`h1` を広げても意味がない。**継続学習は逆効果**
（300k の net から lr 3e-4 で 300k 追加すると 84,118 に落ちる）。

幅が違う net 同士は `pairw` が 1 プロセスで両方を回す都合で同じ `h1`/`h2` ビルドでないと比べられない。`scripts/widen-net.py`
で同梱 net をゼロ埋めすると**値がビット単位で同じ** 128/128 net が作れる（余分なユニットは `relu(0) = 0`）。ゼロ埋め net の
2-ply は 79,657.0 / 3,588.1 moves / 4-of-20 で同梱 net と完全一致した。ペア比較は **+8,623.8 ± 7,679.3**（13 勝 7 敗）で
採用バー（mean − 2·stderr > 0、≒ +15,000）に届かない。非ペアの +13,296 と同じ向きで効果自体は本物だが、12 倍のコストで
+8k は割に合わない。

| 続きの試行 | 結果 | 判定 |
| --- | --- | --- |
| 128/128 300k → lr 0.0015 で +120k（計 420k） | 2-ply 69,888.0、8192 0/20 | 追加学習は逆効果 |
| 64/128（`h2` だけ広い）195k | 70,222.8、1/20、2.70 ms/move | 300k には届かない見込みで打ち切り |
| 128/128 300k を教師に `dson` 30k（7.36M 回帰） | pairw **−68,521 ± 7,048**（0/20）、2-ply 23,080 | 広い net の値を 64/32 に移せない |
| 128/128 300k を教師に `dist` 100k（every 4 / div 3） | −55,826 ± 6,287（0/20）、39,612 | 同上 |

教師ラベルのスケールはほぼ同じ（128/128 net の `tch` は mean 16.48 / max 55.45、同梱 net は 15.79 / 50.54）なのに、
生徒（64/32）は真似すると崩壊する。**広い net の `V` はその net 自身の 2-ply 探索とセットで初めて意味を持つ**値で、
単体の回帰ターゲットには使えない（#185 の「残差だけ学習」も同じ理由で期待できない）。

**seed を変えると再現しない（重要）**: 128/128 を seed 2 で 200k〜280k まで伸ばすと非ペア 85,397.8 だが `pairw` は
**−12,116.8 ± 5,847.5（5 勝 15 敗）**。seed 1 の +8,624 と合わせて散らばりは ±20k 規模（64/32 の seed 分散と同じ）。
**128/128 の +8,624 は容量の効果ではなく seed の引き**で、採用候補にならない。

**結論**: `h2` 中心に 128/128 まで広げても seed で再現せず、コスト 12 倍、300k 超は lr を下げても劣化。伸びしろは 300k 付近で
止まっており、seed を増やしても平均は同梱 net と区別できない。

## 自己対戦の分布を広げる — 探索ノイズと開幕ランダム（#224）

`#169` / `#159` 系（`restart`、dump カリキュラム）は「深い局面から始める」で全滅した。同じ発想を自己対戦側から試した:
① `explore`（一定確率で合法手を一様ランダムに打つ）、② `open8` / `open16`（開幕 N 手だけランダム、その間は学習しない）。
`base` から、lr 0.003、train seed 1、`par` sqrt。非ペア 2-ply（seed 9000、20 局）、同梱 net は 79,657.0 / 4/20:

| 変種 | ゲーム数 | moves | 時間 | 2-ply | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `explore` 5% | 100k | 22,806,876 | 64.3 s | **21,224.4** | 0/20 |
| `explore` 5% | 400k | 114,043,957 | 345.7 s | 12,614.2 | 0/20 |
| `open8` | 100k | 64,155,562 | 176.1 s | 30,916.6 | 0/20 |

ランダム手を混ぜると 1 ゲームの moves が 1/4〜1/6 に縮む（同梱 net 級の自己対戦は 100k で 90〜130M moves）。
`explore` は moves を揃えても 12,614 で、**TD のターゲットにランダム手を含めた時点で値が壊れる**。`open8` はランダム手を
ターゲットから外したのに 30,917 まで落ちる。理由は分布のずれではなく**価値関数の外挿性の無さ**で、greedy の軌道から
外れた盤面では `V` の誤差が大きく、そこでの学習が
**結論**: この net は on-policy（greedy 自己対戦の軌道）でしか学習できない。`restart`（深い局面）、dump カリキュラム（序盤を飛ばす）、
`explore` / `open8`（ランダム手）という 4 系統の分布シフトがすべて −50k〜−70k で失敗しており、状態分布を広げる方向は打ち切り。

## 同梱 net は現行レシピの不動点（#225）

同梱 net から `resume` で自己対戦を続けると必ず数 % 下がり、lr とゲーム数を振っても同じ。**同梱 net はこのレシピ
（TD(0) + greedy 自己対戦 + sqrt 則）の不動点**とみてよい。

| 追学習 | moves | 非ペア 2-ply（9000） | `pairw` vs 同梱 net |
| --- | ---: | ---: | --- |
| 100k、lr 3e-4 | 214,702,634 | 84,839.4（8192 3/20） | −5,495.6 ± 6,887.4（8 勝 12 敗） |
| 300k、lr 1e-4 | 652,561,501 | 81,283.8（8192 2/20） | −5,341.4 ± 9,281.0（5 勝 15 敗） |

非ペア表では +5,182 / +2,628 と「改善」に見えるがペアでは負ける（seed 9000 の非ペア表は同梱 net 自身の 79,657 に対して ±5k
動く）。TD(0) で離れると裾（8192 到達）が先に落ち、平均は互角から少し下に張り付く。

**結論**: 学習ゲーム数・lr・スケジュール・蒸留（配合/教師/移植）・容量・探索ノイズ・開幕ランダム・対称性・backward・`mstd` の
すべてが同梱 net を超えない（13 ラウンド）。現行の目的関数と入力表現の範囲では同梱 net がほぼ最適点で、残る伸びしろは局面表現
（入力特徴）か報酬系の変更だけ。

## 入力に手作り特徴を足す（`-Dfeat`、#226）

one-hot 256 入力に大域特徴を足せるようにした（空きセル数 / 最大レベル / 単調性 / 滑らかさ / 最大タイルが角か / レベルの総和。
0..1 に正規化し `Net.wf[feature_count][h1]` の行として入力層に足す。既定 `-Dfeat=0` は現行の 256 入力とファイル形式のまま）。
`scripts/widen-net.py --feat 6` でゼロ埋めするとビット一致（79,657.0 / 4/20 を再現）。`base` から、lr 0.003、seed 1、64/32:

| 条件 | ゲーム数 | 学習 | 2-ply（9000） |
| --- | ---: | --- | ---: |
| 特徴なし（対照） | 20k | 直列 | 48,279.2 |
| `-Dfeat=1`（空きセル数だけ） | 20k | 直列 | **60,776.4**（14.1M moves） |
| 特徴なし（#203 sqrt s1） | 100k | `par` sqrt | 82,600 |
| `-Dfeat=1` | 100k | 直列 | 81,959.0（8192 2/20） |
| `-Dfeat=1` | 100k | `par` sqrt | 27,432.6（壊れる） |
| `-Dfeat=6` | 100k | `par` sqrt | 46,200.2（壊れる） |

**20k では特徴が効く**（+12.5k）が、100k では 82,600 → 81,959 で差が消える（one-hot の net が同じ情報を自力で覚える）。特徴は
「学習の初速」を上げるだけで収束先を変えない。`par` + 特徴は明確に壊れる（原因未特定。重みのスケールが小さく max|w| 1.76 対
同梱 20.39 で、バッチ平均が特徴行の方向を打ち消している可能性）。**バッチ経路は特徴ビルドを拒否する**ようにした
（`parSupported` が `nn.feature_count != 0` で false）。

## 2-ply に揃えた学習を測る — `t2` は発散し `p2` だけが残る（#228）

測る前に 2 つ修正: (1) `search2Value` と `pickTeacherValue` は重み更新をまたいで `sharedValueCache()` を読んでいた
（`ValueCache.clear` 自身のコメントが「重み更新の間の価格計算は空の表から始めろ」と言っているのに `teacherValue` だけが clear
していた）。両方の入口で clear し、`test "search2 target and deep2 pick price with the current weights, not the shared cache
(#228)"` で固定。(2) `parSupported` が `td1` + `greedy1` 以外を拒否していたので `search2` + `greedy1|deep2` を許可（価値表は
threadlocal で手ごとに clear するので衝突しない）。`use_resume` に `t2`/`p2`/`tp2` を足した。

同梱 net から、seed 9000、lr 3e-4（`par` は sqrt 則で 1e-3）:

| variant | ゲーム数 | moves/ゲーム | 結果 |
| --- | ---: | ---: | --- |
| `resume`（td1、対照） | 200 | 1,932 | 安定 |
| `t2`（`lr=0`） | 5 | 1,875 | 方策は `resume` と同一（9,376 moves 一致） |
| `t2` | 1 / 2 | 1,795 / 2,358 | まだ普通 |
| `t2` | 5 / 10 / 20 / 50 | 1,752 / 929 / 645 / 378 | 崩壊 |
| `t2` | 5,000 | 212 | 崩壊（最後の 1 局 1,864 点） |
| `tp2` | 50 | 692 | 崩壊 |
| `p2` | 50 | 4,180 | 安定（2-ply 評価は約 3,600 moves/ゲーム） |

lr を下げても崩壊は遅れるだけ（200 ゲームで lr 1e-4 は 357、1e-3 は 219 moves/ゲーム）。`lr=0` では `t2` と `resume` のゲームが
1 手も変わらないので目標の計算自体は正しい。`diag` では `lr=0` の `t2` の平均 |TD 誤差| が 0.17〜0.30 で `resume`（0.13〜0.23）と
ほぼ同じ、100 ゲーム後は全体 2.77 / 空き 0〜3 で 7.01、`V(s') - V(s)` の平均 −3.5（終盤で楽観側に発散）。

**読み**: `search2` の目標は `r_{t-1} + max_d [r_t + E_spawn max_d' V(A_{t+1})]`、つまり目標の中に max が入った 2 段の値反復
（γ=1、関数近似あり）。`td1` は方策評価で max が無いから同じ lr で安定する。max を内側に入れると V のノイズが楽観側に増幅され、
argmax がノイズで選ばれ、ゲームが数百手に縮み、崩れた盤面（最大タイル 256 以下が 74%）で学習する悪循環になる。`bootValue` /
`bootChance` の正規化（`mass` で割るので `p` に `/empties` が無くても比は正しい）は正しく、直すべきは演算子の側だった。

`p2`（自己対戦だけ `.deep`、目標は `td1`）は安定するので **#158 で残ったセルは `p2` だけ**。評価が訪れる深い盤面を自分で訪ねる
ので、`dumpp2` / `restart`（古い方策の深い盤面）と違って分布が今の方策に一致する。ところが測ると**負ける**。同梱 net から
1,000 ゲーム（lr 3e-4、直列、100 ゲームずつ 10 本）:

| 指標 | 同梱 | `p2` 1,000 |
| --- | --- | --- |
| `pairw` 20 局 seed 9000 | — | **−23,980.6 ± 7,205.4**（4 勝 16 敗） |
| 2-ply 20 局 seed 9000 | 79,657.0（4/20） | 71,903.0（2/20） |
| 2-ply 20 局 seed 9100 | 91,348.0（5/20） | 81,754.0（4/20） |

学習中のゲームは 3,251〜3,966 moves で崩れない（`t2` とは別物）が、評価は両方向に負ける。#135 の「2-ply 側に寄せると 1-ply は
+24%、2-ply は −8%」と同じ向きで、**探索の上乗せは V が 1-ply greedy の分布で鍛えられているから効く**という説明と一致する。

`p2` を `par` で回すと詰まる（8 ゲームのバッチが最も長い 1 局を待つので 4 時間で 2,700 ゲーム、CPU 時間 597 分・平均 2.5 コア、
終盤は 1 コア相当）。`t2`/`tp2` の崩壊 net が長いゲームを出すのも同じで、**ゲーム長の上限が無い**のが効いている。`p2` の実測は
直列 100 ゲーム刻み（1 本 9〜11 分）。**#158 の軸はここで閉じる**（`t2`/`tp2` は発散、`p2` は安定するが悪化）。

## 容量の切り分けをやり直す: 入力側を広げた 128/32 が +57,424 で採用（#233）

#220 は「128/128 は seed の引き、効いているのは `h2`」と結論していたが、128/32 の 300,000 ゲームを同条件（`base` スクラッチ、
lr 0.003、seed 1、`par`、`batch_scale` = sqrt(8) = 2.8284271247461903、評価 20 局 seed 9000）で回し直すと **113,682.2** が出て
記録の 78,500 と食い違った。今回の測定は 2 点で裏が取れている: 生成物が 296,472 B = 16 + 37,057 × 8 で 37,057 =
256×128 + 128 + 32×128 + 32 + 32 + 1（本当に 128/32）、同梱 64/32 を `scripts/widen-net.py --h1 128 --h2 32 --src-h1 64
--src-h2 32` でゼロ埋めすると **79,657.0 を小数まで完全再現**する。78,500 は 64/32 の 300k とほぼ一致するので、`-Dh1` が exe に
届いていなかったか `batch_scale` のスロットに `2` を渡していた（64/32 では 51,483 まで落ちる罠、#231）可能性が高い。
**以後、容量の測定は重みファイルのサイズとゼロ埋め対照で必ず裏を取る。**

| 評価 | 128/32 300k（採用） | 同梱 net（64/32） |
| --- | --- | --- |
| 非ペア 2-ply seed 9000 | **113,682.2**（8192 11/20、4,919 手、0.89 ms/move） | 79,657.0（4/20、0.87〜0.97） |
| 非ペア 2-ply seed 9100 | **129,456.2**（14/20、5,486 手、0.98） | 91,348（5/20） |
| 非ペア 1-ply seed 9000 / 9100 | 57,285.0 / 57,133.8 | 48,873.6 / — |
| `pairw` 20 局 seed 9000（A = ゼロ埋め同梱） | **+57,424.4 ± 7,567.9、19 勝 1 敗**（681〜682 s） | — |

`mean diff − 2*stderr = +42,288` で採用条件（≒ +15,000）を 3 倍近く上回る。**`h2` を広げる必要は無く、`h1` を 64 → 128 に
するだけで足りる**（128/128 の 300k は 92,953 なので `h2` はむしろ邪魔）。コストは学習 2.6 倍（100k で 5 分 → 300k で 61 分）、
推論 1.2 倍。

採用した変更:
- `ai/solver/nn_weights.bin` を 128/32 の 296,472 B に差し替え（旧 64/32 は `/tmp/ship64_32.bin` に退避）。
- `build.zig` の `-Dh1` / `-Dh2` の既定を 128 / 32 に変更（既定ビルドが同梱重みと一致する）。
- `nn.Staged` は 7 net（128/32 で約 2 MB）になったので `heapStaged` を足してヒープに置く（スタックだとテストランナーが
  セグメンテーション違反）。
- 固定値のテスト更新: golden play（`nn` 32,380 → 81,104 / `nn2` 77,164 → 132,164、moves・checksum も）、学習チェックサム
  （1621370333 → 3042928886、4199351724 → 3379822721）、量子化の重み数（18,561 → 37,057）とファイル長（37,170 → 74,162）、
  `mid_board` の greedy/chance（.left/.left → .up/.down）、death の学習ゲーム数（60 → 200、64 ゲームでは値が 0 付近で動かない）、
  量子化の丸め許容（step/2 + step×1e-6 → ×1e-3、f32 の丸めで比が 0.5003 になる）。
- `zig build test` は exit 0（約 50 分）。`zig build wasm -Doptimize=ReleaseFast` は 538,377 B で `node scripts/wasm-smoke.mjs`
  が OK（nn2/nn3 の 2 方策、10 手、最遅 solve 595 ms）。

seed 2 の再現は #234、300k より先は #204 / #220 で否定済み、`h1` 96/192/256 の追試は下記。

### 追試: h1=96/192/256 は 128/32 を超えない（#233 継続）

`h2=32`、300,000 ゲームをスクラッチから、lr 0.003、`par`、`batch_scale=2.8284271247461903`、train seed 1 と 2。重みのバイト数が
`16 + 8 × (256×h1 + h1 + 32×h1 + 2×32 + 1)` と一致することも確認（96/32 222,488 B、192/32 444,440 B、256/32 592,408 B）。
評価は同梱 128/32 を候補幅へゼロ拡張し同じ幅で回した（`pairw` 20 局 seed 9000 + 非ペア 2-ply）。

| h1 | seed | moves / train s | pairw B−A ± stderr（勝/敗） | 2-ply 9000（8192） | 2-ply 9100（8192） |
| ---: | ---: | ---: | --- | ---: | ---: |
| 96 | 1 | 518,762,880 / 2,192.3 | −42,002.4 ± 9,381.9（4/16） | 78,799.0（1/20） | 75,376.8（0/20） |
| 96 | 2 | 469,812,682 / 2,136.2 | −46,028.2 ± 10,577.7（3/17） | 77,286.4（0/20） | 75,743.0（0/20） |
| 192 | 1 | 570,706,944 / 5,218.7 | −94,978.0 ± 8,382.3（0/20） | 25,003.2（0/20） | 25,018.2（0/20） |
| 192 | 2 | 565,281,160 / 4,932.2 | −50,149.4 ± 9,327.7（2/18） | 75,324.8（0/20） | 75,277.4（0/20） |
| 256 | 1 | 562,207,909 / 6,878.0 | −64,878.2 ± 8,466.6（0/20） | 54,514.8（0/20） | 54,789.0（0/20） |
| 256 | 2 | 598,721,027 / 7,272.5 | −75,009.2 ± 11,080.3（2/18） | 66,123.8（0/20） | 66,254.0（0/20） |

全 6 run で負け、非ペアの平均点・8192 到達率も大きく下回る。192/256 は推論も遅い。**判定: 不採用**。300k・現行レシピでは
h1=128 が最良で、同条件の単純な h1 拡大は追加しない。

## 128/32 は seed で当たり外れが出る: 発散・lr・clamp（#234）

採用した 128/32 は seed 1 で 113,682 を出したが **seed 2 では発散した**: 300,000 ゲームが 96,836,342 手（**323 手/ゲーム**）で
終わり、`|w|` 1.685e86・**37,057 個のうち 6,334 個が NaN**。1-ply 2,091.6 / 2-ply 11,478.4（4096 到達 0/20、死 <2048 が 20/20）。
崩壊は 100,000 と 300,000 の間で起きる（seed 2 の 100,000 ゲームは 143,217,467 手 / 1,041.8 s で正常、2-ply **84,040.6**・8192 4/20）。
「学習の途中で値が暴走する」型で、急に別の解に落ちるわけではない。

| 条件（seed 2、128/32、300k） | 2-ply 9000 | 8192 | 2-ply 9100 | 手/ゲーム | max\|w\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr 0.003（発散） | 11,478.4 | 0/20 | — | 323 | 1.685e86 |
| lr 0.0015 | 76,366.6 | 0/20 | — | 1,818 | 21.96 |
| lr 0.003 + `clamp` 200 | 70,131.8 | 0/20 | 71,369.0 | 1,862 | 29.93 |
| （参考）seed 1、lr 0.003、採用 net | **113,682.2** | **11/20** | **129,456.2** | 1,844 | 27.06 |

**読み**: 採用 net の 8192 11/20 は、lr 0.003 が安定域のぎりぎり外側にある状態でしか出ていない。lr を半分に落としても、TD 目標を
200（スコア 200,000）で切っても、学習は「4096 で終わる」安定な解に落ち着き**裾が出ない**（`clamp` は 3-ply の代わりに V の上限を
作るので、終盤の「まだ伸びる」局面の価値を押し下げる）。1 本 61 分の当たり外れなので、**seed を 2〜3 回振って bar を超えたものを
採る**のが実務的な運用（#233 の +57,424 は 19 勝 1 敗なので採択 net 自体が引きである可能性は低い）。

**発散ガード（実装済み）**: 発散した実行は最後の良いチェックポイントを上書きしてしまう（5,000 ゲームごとに `nn.save`）ので、毎ゲーム
`nn.weightsFinite` を見て非有限か `|w| > 1e6` なら `diverged at N/M games (non-finite or |w| > 1e6), not saving <path>` を出して exit 1
する。lr 0.05 の人工発散では 1,976/20,000 ゲーム目で発火しファイルが作られないことを確認。`mstd` は 7 net すべてを見る
（`stagedFinite`）。

**`clamp` variant**: 13 番目の引数に上限を渡す（既定 200 = スコア 200,000、`par` と併用可）。上の表のとおり**不採用**。

## 反復 DAgger は「壊した分を戻す」だけ（#236）

`ft2` は #207 で発散すると分かっているので、**凍結ラベルの `tch` → `dist` を 1 周とみなし 2 周**回した（A = 同梱 128/32）。
dump は 1 周目 288 局 / 2 周目 144 局を lr 0 の `dumpp2` で集める（lr > 0 だとゲームが進むほど方策が壊れて 4096 に届かず dump が
0 件になる）。

| 周 | dump（局 / 記録 / 破棄 / 時間） | 教師 | `dist` 20k every4 div3 | 2-ply（9000 / 9100） | 8192 | `pairw` vs 同梱 |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | 288 / 65,536 / 918,953 / 96 分 | 65,536（214 s、3.3 ms/sample） | 1,048 s | 103,142.4 / 96,618.8 | 9/20, 7/20 | **−27,244.4 ± 12,873.1**（6 勝 14 敗） |
| 2 | 144 / 65,536 / 325,584 / 43 分 | 65,536（209 s） | 1,014 s | **114,409.8** / 115,239.8 | 13/20, 11/20 | **+2,366.8 ± 9,601.5**（10 勝 10 敗） |

1 周目で −27k まで落ち、2 周目は「壊した分」を戻して同梱と同点。seed 9000 の 8192 到達は 13/20 と多いがペアでは 10 勝 10 敗。
65,536 個の `deep3` ラベルへ 20,000 ゲームかけて引っ張るので方策が「深い局面で教師が指す手」に寄り、裾を作っていた自己対戦の
多様性が削がれる。**DAgger は安定化の道具**で、教師が同梱より強いときにしか上積みしない（今の `tch` の教師は #167 で終局前の
誤差 0.30 と測られ、同梱より強い保証がない）。コストは 1 周 2〜2.5 時間。

## `pairw` の共通 spawn を盤面ごとに一様に写像する（#237）

#235 は spawn の記録を「空きマス index」に直したが、A と B の空き数が違うと B 側で index を末尾へ clamp するため穴の分布が偏る
（4 穴の A から 2 穴の B へ渡す例では B の 2 番目の穴が 75%）。tape は盤面に依存しない一様な `u32 pick` を記録し、各盤面が
`floor(pick × empty_count / 2^32)` で自身の row-major 空きマスへ写像する（各 hole bucket は `floor(2^32 / n)` か `ceil(...)` なので
偏りは最大 1 pick / 2^32）。

`Spawn` は 1 → 8 bytes、`tape_capacity = 8192` は 64 KiB、`pairw` は 20,000 件を heap に確保するので 20 KB → 160 KB。記録側の
cell draw が bounded `usize` index から full-range `u32` に変わるので、同じ seed の `playValue` と byte-for-byte 同じゲームになる
保証はなくなった（tape の record / replay の再現性は維持）。テストは空き数 1..16 の全 bucket 境界、4-hole → 2-hole の偏り例、raw
pick の記録/replay、ゼロ容量と通常容量でゲーム・乱数状態が一致することを確認。**#237 より前の pairw 結果は index+clamp の旧テープ
による歴史値**で、今後候補を判定するときは現在の実装で A/B を取り直す。

## 学習 seed 3 は 60k で病的な長ゲームに落ちる（#241）

採用レシピ（128/32、300k、`base` スクラッチ、lr 0.003、`par`、sqrt 8）を train seed 3 で回したところ、
`train 55000/300000 last game score 12208`（開始 8 分）→ `train 60000/300000 last game score 51200`（開始 61 分）で、
55,000 → 60,000 の 1 バッチ（8 ゲーム）に 53 分かかった。CPU は動いており（`sample` で `nn.forward` / `nn.train`）、8 ゲームの
うち少なくとも 1 本が異常に長い。`nn.weightsFinite` は有限のままなので発散ガードは不発。seed 2 の「NaN」とは別の失敗形で、
**重みは壊れないがゲームだけが異常に長くなる**。開始 62 分でプロセスを止め、60k チェックポイント 296,472 B が残った。
**判定: 不採用（未完走）。** seed 3 の 300k から seed 1 の 113,682.2 を超える net は出ていない。

## 手数上限で病的な長ゲームを止め、seed 3 の 300k を判定する（#242）

- `ai/solver/nn.zig`: `default_move_cap = 100_000` と `Options.move_cap`（0 で無効、既定 100,000）。`trainGameOpts` の 3 経路
  （per-move / `nstep` / `backward`・`replay`・`death`）と `trainGameStaged` が上限で打ち切り `GameResult.capped = true` を返す。
  `par` は `BatchOutcome.capped_game` / `capped_moves` で「バッチの何番目が何手で上限か」を返す。
- CLI はどの経路でも `game 60001/300000 hit the 100000-move cap at 100000 moves, stopping the run and not saving <path>` を出して
  `error.MoveCap`（exit 1）。`diverged` と同じく最後の good checkpoint を書かない。
- 上限チェックは rng を引かないので、上限未満のゲームは従来と同じ乱数列・更新・`moves`。テスト
  `move cap leaves a normal game bit-identical and stops a long one (#242)` と
  `the parallel batch reports and aggregates the capped game (#242)` が固定。

**実測（seed 3、128/32、300k、lr 0.003、`par`、sqrt 8）**: `508,236,212 moves / 2,995.3 s`（wall 50.6 分）、上限メッセージなしで
**完走**。1 ゲーム 1,694 手。同コマンドを 60k で 2 回回すと checkpoint（296,472 B）とログがバイト一致で**再走は決定的**
（`trained 60000 games, 53164118 moves`、292.7 s と 308.1 s の差は wall だけ）。ただし #241 の 60k checkpoint とは**一致しない**
——あの run は環境差で別の軌道を走っており、「55k→60k に 53 分」は今日のこのマシンでは再現しない。上限は再発への保険。

| net | seed | avg_score | 8192 | ms/move |
| --- | ---: | ---: | ---: | ---: |
| 同梱 128/32 | 9000 / 9100 | 113,682.2 / 129,456.2 | 11/20 / 14/20 | 0.89 / 0.85 |
| seed 3 300k | 9000 / 9100 | 22,924.0 / 22,774.0 | 0/20 / 0/20 | 0.91 / 0.89 |

`pairw`（A = 同梱、B = seed 3、20 局、seed 9000）: **−97,245.0 ± 8,392.3**（0 勝 20 敗、468.5 s）。**判定: REJECT**（同梱 net の
約 1/5）。重みは有限のまま方策だけが壊れている。#241 の失敗形（長ゲームで止まる）とは別。lr 0.003 の 128/32 は **3 seed 中
1 seed だけが実用**（seed 1 = 113,682.2、seed 2 は発散、seed 3 は完走しても弱い）。seed を増やす方向は打ち切り、次は安定化。

### clamp の上限を緩めても seed 2 の裾は戻らない（#244）

`clamp` は TD 目標を ±ceiling で切る。200 は完走したが 8192 0/20 だったので 300 / 500 / 1000 / 2000 を seed 2 / 128/32 / 300k /
lr 0.003 / `par` / sqrt 8 で測った。**上限は on/off のスイッチ**として効く。

| 条件（すべて seed 2） | 完走 | 停止位置 | max\|w\| | moves | 手/ゲーム | 2-ply 9000 | 8192 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `clamp` 200（#234） | ○ | — | 29.93 | — | 1,862 | 70,131.8 | 0/20 |
| `clamp` 300 | ○ | — | 29.19 | 552,475,563 | 1,842 | 71,981.8 | 0/20 |
| `clamp` 500 / 1000 / 2000 / なし | × | 50,440 | 17.17（50k） | — | — | 69,586.6（50k） | 1/20（50k） |
| （参考）seed 1 採用 net | ○ | — | 27.06 | 553,202,634 | 1,844 | **113,682.2** | **11/20** |

500 以上は**一度も効かない**: 500 / 1000 / 2000 / なし の 4 本は停止位置（50,440）だけでなく 50k checkpoint の md5 まで一致
（`38818bb4c8b3cfa2fd2481ab3b946f25`、max\|w\| 17.17）。目標が 500 を超える前に `|w| > 1e6` へ達してガードが止めている。
300 は完走するが 71,981.8 / 0/20、best tile 4096、評価で 3,269 手/ゲーム（`clamp` は V の上限を作り、終盤の「まだ伸びる」局面を
押し下げる）。

**発散の原因は目標の大きさではない**: 目標 500 未満でも |w| は 50k の 17.17 から 440 ゲームで 1e6 を超える。次は**更新量**
（1 step の `|Δw|` か勾配ノルム）の制限。なお 50k 時点で既に 2-ply 3,221 手/ゲーム（8192 1/20）と長ゲーム側に寄っており、ガードが
止めたのは「重みが壊れた瞬間」で「方策が壊れ始めた瞬間」ではない。**判定: 不採用**（`pairw` も未実施）。

### 弱い clamp（300）は seed 3 でも効かず、net は clamp なしとバイト一致（#245）

seed 3 / 128/32 / 300k / lr 0.003 / `par` / sqrt 8 に `clamp` 300 を適用: 300,000 games / 508,236,212 moves / 3,053.8 s /
max\|w\| 28.80。できた net の md5 `78ed71734004c9a994abd7224193764b` は #243 の clamp なし seed 3 と一致し、サイズも同じ
296,472 B ——**`variant=clamp+par` と表示されても学習の軌道は `base+par` とバイト単位で同じ**。評価は 22,924.0 / 22,774.0
（0/20）、`pairw` −97,245.0 ± 8,392.3、`eval-net.sh` の VERDICT は REJECT（paired margin −114,029.6）。

**clamp の上限は seed ごとに効く/効かないが変わる on/off スイッチ**で、seed 3 の弱さは clamp では動かない（seed 3 は目標が 300 に
届く前に学習が終わる）。**判定: 不採用。**

### 更新量の上限 variant `clip` を足した（#246）

- `nn.Options.grad_clip`: `par` が 1 バッチで適用するマージ済み delta の L2 ノルム上限。超えたら delta 全体を同じ比で縮める
  （方向は保つ）。`sym_batch_clip` と同じ流儀。
- `nn.Options.step_clip`: 1 step の要素ごとの `|Δw|` 上限。直列の 1 手 1 更新と `par` の 1 バッチ 1 step の両方に効く
  （バッチ内の各ゲームは勾配を貯めるだけなので切らない）。
- どちらも既定 0 = 上限なし、0 では演算を足さないので既存 run はバイト一致。CLI は variant `clip`（arg 13 = 上限、既定 1 =
  `sym_batch_clip` と同スケール、arg 14 = `norm`|`step`、`norm` は `par` 必須）。ログは `variant=clip:norm+par` / `clip:step+par`。
- 200 ゲームのスモークで、上限 1e9 は base と md5 一致（`976fea58b0c4b1e28eca694e5b78c445`）、`step` 0.001 は
  `ddc4a1101382af9720be0dc6e4c1d0ff`、`norm` 1e-5 は `b1a79e29bbd5604b80457b62ada778e1` と重みが変わる。

### 更新量の上限は seed 2 を完走させるが、8192 の裾は戻らない（#247）

60k のスクリーニング（norm 1 / 3 / 5）は 3 本とも完走し、3 と 5 は上限なしと md5 一致（`bd70b6e0d4f7d17b893d6d759cc73cb8`、
59,774,960 moves）、1 だけ重みが変わった（`033f8fb2cd2716af43ef5d36353946ce`、max\|w\| 12.33 対 12.56）。ただし 60k は lr 減衰が
速く上限なしでも発散しないので**60k の完走では 300k の可否を予測できない**。実測した delta ノルムは序盤 2.3e-2 から 35k〜41k で
5e-1（要素最大 1e-2 → 1e-1）なので、300k には 3 と 10 を進めた。

| 条件（seed 2、300k） | 完走 | max\|w\| | moves | 手/ゲーム | 2-ply 9000 / 9100 | 8192 | ms/move |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `clip` norm 3 | ○ | 30.38 | 559,274,117 | 1,864 | 69,415.8 / 68,284.4 | 0/20 | 1.09 / 1.10 |
| `clip` norm 10 | ○ | 28.13 | 552,796,927 | 1,843 | 73,388.0 / 73,560.2 | 0/20 | 0.95 / 0.97 |
| `clamp` 300 | ○ | 29.19 | 552,475,563 | 1,842 | 71,981.8 / — | 0/20 | — |
| 上限なし（対照） | × | 17.17（50k） | — | — | 69,586.6（50k） | 1/20（50k） | — |
| 同梱 net | ○ | 27.06 | 553,202,634 | 1,844 | **113,682.2 / 129,456.2** | **11/20** | 0.91 / 0.88 |

**上限は「完走」だけを買う。** 有限・NaN なし・move cap 不発・手/ゲームも正常（1,864 / 1,843）なのに、評価は 4096 で終わる
3,000 手超の長ゲームに落ち、`clamp` 300 とほぼ同じ位置に着地する。発火は**1 回だけ**、50,4xx の巨大 step（`clipbind` の
`norm=4.960741e86 scale=2.016e-86`、位置は #244 の停止位置の直後）。3 と 10 の差は seed 内ばらつき程度で、**3〜10 のどこに
置いても結果は同じ**。**判定: 不採用**（`pairw` 未実施）。次に疑うのは上限の値ではなく発散前（40k〜50k）の軌道そのもの。

### seed 2 の 300k を 5k 境界で評価（#249）

`train_games=300000` 固定で 20k〜50k の checkpoint を採取（50,440 で発散する前に 50k まで書けた。50k の md5
`38818bb4c8b3cfa2fd2481ab3b946f25` が #244 と一致し同じ軌道を確認）。

| checkpoint | last game score | 2-ply 9000 | 手/ゲーム | 8192 | max\|w\| |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | 27,476 | 45,651.6 | 2,216.1 | 0/20 | 6.7368 |
| 25k | 15,360 | 55,266.8 | 2,620.0 | 0/20 | 9.6964 |
| 30k | 15,308 | 52,510.6 | 2,487.9 | 0/20 | 12.4800 |
| 35k | 15,348 | 59,061.2 | 2,784.3 | 0/20 | 15.1056 |
| 40k | 34,276 | 61,401.8 | 2,883.9 | 0/20 | 15.8696 |
| 45k | 51,392 | 63,885.6 | 2,982.0 | 0/20 | 16.1464 |
| 50k | 34,476 | 69,586.6 | 3,221.2 | 1/20 | 17.1687 |

50k の seed 9100 は 75,496.2 / 3,454.7 手 / 2/20。**結論: 8192 の裾が「落ちた」checkpoint は 20k〜50k に観測されない**（最初の
計測点 20k が既に 0/20 なので、消失は 20k 以前か、seed 9000 の 20 局では見えていない）。50k で 1/20 に戻ったので、#244 の
「50k で 1/20」は tail が突然落ちた位置ではなく、20k〜45k で見えず 50k で一度戻った結果。`last game score` は 15,308〜51,392 と
大きく揺れ、単独では長ゲーム化の開始点を指さない。評価の手/ゲームと max\|w\| は 2,216→3,221、6.74→17.17 と増える。

（注意: 各 N を `train_games` にして独立学習すると lr 減衰が `frac = g / train_games` に依存するので N ごとに別スケジュールになり、
300k レシピの途中 checkpoint とは別軌道になる。上の表は 300k horizon を固定して実際の 5k checkpoint を採った結果を優先。比較の
ために回した独立 N run は 20k〜50k のすべてで 8192 0/20 だったが、300k レシピの回答には混ぜない。）

### seed 2 の 300k を 5k〜15k で評価（#250）

同じ run の序盤 checkpoint（5k の md5 `9668ecdfc0de7e3b10ddecfa981d4a63`、10k `813c330ba2a4af1e614ad33aaf8bd274`、15k
`9d551ef0a83fe4e3af91e823377eef6e`、20k `2b541e6f1df0fb550906ed4bdf3ac9d6`）。

| checkpoint | last game score | 2-ply 9000 | 手/ゲーム | 8192 | max\|w\| |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 2,944 | 22,583.8 | 1,236.5 | 0/20 | 0.7573 |
| 10k | 8,736 | 28,080.0 | 1,474.9 | 0/20 | 2.0638 |
| 15k | 15,476 | **34,053.6** | **1,729.5** | 0/20 | 4.1976 |
| 20k | 27,476 | 45,651.6 | 2,216.1 | 0/20 | 6.7368 |

15k の seed 9100 は 33,017.6 / 1,680.0 手 / 0/20（同梱 net は 129,456.2 / 5,486 手 / 14/20）。**結論: 5k〜15k に有望な
checkpoint は無く、悪化開始点もこの窓では特定できない**（avg_score と手数は 5k から 20k まで順に伸びるが 8192 は全点 0/20）。
単一 seed の 20 局では裾の消失を 20k より前と断定できず、「20k 以前に落ちた」か「20 局では観測できなかった」かは未解決。

### `clip` norm 1 はさらに弱い（#251）

seed 2 / 128/32 / 300k / lr 0.003 / `par` / sqrt 8 / `clip` norm 1: 300k 完走、547,999,305 moves（1,826 手/ゲーム）、
max\|w\| 27.2242、NaN なし、296,472 B（md5 `cd786cd1b7f9a74bd9a96c17d2102d92`）、2,997.5 s（shell wall 99m22.7s）。上限なしが
50,440 で発散するので上限は少なくとも 1 回発火（回数は未計測）。

| 候補 | avg_score 9000 / 9100 | 手/ゲーム | 8192 | ms/move |
| --- | ---: | ---: | --- | --- |
| `clip` norm 1 | 46,417.4 / 46,907.4 | 2,069.6 / 2,072.6 | 0/20 | 1.08 / 1.03 |
| `clip` norm 3 | 69,415.8 / 68,284.4 | 3,134 / 3,076 | 0/20 | 1.09 / 1.10 |
| `clip` norm 10 | 73,388.0 / 73,560.2 | 3,327 / 3,335 | 0/20 | 0.95 / 0.97 |
| 同梱 net | **113,682.2 / 129,456.2** | — / 5,486 | **11/20 / 14/20** | 0.91 / 0.88 |

norm 1 は norm 3 より約 23k / 21k、norm 10 より約 27k / 27k 低い。**判定: 不採用**、clip 上限を下げる方向は打ち切り。

### 学習 seed 4 は同梱 net の 6 割（#252）

128/32 / 300k / `base` / lr 0.003 / `par` / sqrt 8 / train seed 4: 完走、512,708,842 moves、2,903.6 s、296,472 B
（md5 `ef6248aed329fa2ae9c64c4d5bd721c7`、SHA-256 `697bfac28373bdedd8e9bf32928b8d37682a7412e9bd803dbc52883bbf4544b8`）、
発散・move cap なし。

| net | avg_score 9000 / 9100 | 手/ゲーム | 8192 | ms/move |
| --- | ---: | ---: | --- | --- |
| train seed 4 | 67,497.8 / 66,838.0 | 3,036.0 / 2,987.3 | 0/20 | 0.94 / 0.89 |
| 同梱 net（seed 1） | **113,682.2 / 129,456.2** | 4,919.0 / 5,486.0 | **11/20 / 14/20** | 0.85 / 0.87 |

両 seed で約 40.6% / 48.4% 減、8192 0/20。事前条件「両 seed で大きく悪化しない」を満たさないので `eval-net.sh` / `pairw` の
正式判定には進まず**不採用**（seed 1 の強さは seed 4 で再現しない）。

### 128/32 の train seed 5 も同梱 net を大きく下回る（#254）

seed 5: 300k 完走、548,853,567 moves、3,089.6 s（51m29.6s）、最終ゲーム score 56,324、296,472 B
（md5 `2a8e37fa81284b6085fe532d5986b36a`）、5k〜295k の checkpoint と 300k を退避、発散・長ゲーム停止なし。
| net | seed | avg_score | 手/ゲーム | 8192 | ms/move |
| --- | ---: | ---: | ---: | --- | ---: |
| 同梱 128/32 | 9000 / 9100 | 113,682.2 / 129,456.2 | 4,919.0 / 5,486.0 | 11/20 / 14/20 | 0.87 / 0.85 |
| seed 5 / 300k | 9000 / 9100 | 55,401.6 / 55,363.8 | 2,509.0 / 2,504.7 | 0/20 / 0/20 | 1.05 / 0.93 |

**判定: 不採用**（両 seed で同梱の約半分、8192 0/20）。seed 1 の成功は train seed に敏感、という結論が補強された。

## 探索深度の再評価（#255、#267）

同梱 128/32 net を固定し、40 局（seed 9000–9019 と 9100–9119、各 20 局、非ペア):

| policy | 40 局 avg_score | 8192 | pooled ms/move |
| --- | ---: | ---: | ---: |
| 2-ply（`deep`） | 121,569.2 | 25/40 | 0.884 |
| 3-ply（`deep3`） | **137,341.6** | **33/40** | 19.326 |
| 終盤 4-ply（`phase_h4`） | 136,593.3 | 30/40 | 43.272 |

3-ply は 2-ply より +15,772.4（+13.0%）、8192 は +8 局、コストは 21.9 倍。`phase_h4` は 3-ply より 748.3 低く 8192 も 3 局
少ないのに 2.24 倍遅いので次候補にしない。**既定の 2-ply は置き換えない**（avg score 優先の用途では 3-ply）。

`deep4`（`ply=4`、cutoff 1e-2、node TT 付き、row 17）は seed 9000 の 1 局で `deep3` に 44,024 点差で負け（130,656 対
174,680）、ms/move 25.95 倍・leaves/move 24.53 倍。空き 4/6/8 の probe で `deep3` の 20.1/21.8/22.6 倍の葉を読む。10 局
スクリーニングは 3 時間 22 分で 1 局も終わらず 9 本を停止した。**不採用**（row 17 と `pairrow` は高速化検討用に opt-in で残す）。

## 探索の枝葉を削る評価（#258、#260）

downside risk（policy 13、40 局）: `deep` 121,569.2 / 25/40 / 0.95 ms → +risk 0.3 で 119,489.8 / 23/40 / 0.98。seed 9000 は
+3,738.8 だが seed 9100 は −7,897.6(sic)、40 局で **−2,079.4（−1.7%）**、8192 は 2 局減。**不採用**（#187 の `risk_weight=0.3` は
現行 net でも効かない）。

葉の直前の報酬を省く `omit_leaf_reward`（#260、opt-in の評価専用、既定 false）: 20 局ペアで −1,911.6 ± 12,264.9
（12 勝 8 敗、mean−2σ = −26,441.4）。非ペア 40 局は +1,597.6 だが seed 9100 で −14,710.0、8192 3 局減、ms/move 0.86 → 0.96。
**不採用**（スイッチのみ残す）。

## node TT: 内部探索値の再利用で 3-ply が 39.5% 速い（#263、#266）

`ValueCache` は葉（afterstate）だけを覚え、`Search.value` / `Search.chance` の内部ノード値は再訪されても毎回計算していた。
`Options.node_tt` に direct-mapped の `NodeCache` を足す。キーは盤面 16 バイト（`log2 + 1`、空き 0）、`value`/`chance` の別、残り深さ、
cprob（固定 spawn カットが効く深さだけ: `chance` は depth > 1、`value` は depth > 2。浅いノードは cprob 非依存で 1 キー共有）。
ハッシュはスロット選択だけ、衝突は 16 バイト全比較でミス。予算切れの値は保存せず、トップレベル探索ごとにクリア。1 スロット 48 B
（#260 の `direct_leaf` も同居）、desktop 4096 スロット = 192 KiB/thread、wasm 512 = 24 KiB。

#266 で `Options.node_tt` を `?bool = null` にし、**未指定なら 3-ply 以上で on、`false` で明示 off**。2-ply の `deep` は確保も probe も
せず結果はビット単位で不変。深さは探索開始後に決まるのでテーブル確保もその後。phase 行では 3-ply のパート（`h1`〜`h3` の深い側、
`h4` の 4-ply）が on、2-ply のパートは off。

10 局（seed 9000–9009）を row 4（明示 off）と 15（既定）で同じ spawn:

| row | policy | avg score | 手/ゲーム | 8192 | ms/move | leaves/move | node hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `deep3` TT off | 138,669.6 | 5,862.8 | 8/10 | 22.95（#263 では 23.87） | 19,278.4 | — |
| 15 | `deep3` 既定（TT on） | 138,669.6 | 5,862.8 | 8/10 | **12.80**（#263 では 14.45） | 10,675.8 | 20.0% |

probe 661,724,183 回 / hit 132,069,729 回。節約分を spawn カット 1e-3 に回す案（`pairtt`、A = TT 1e-2、B = 1e-3、20 局）は
10 勝 10 敗、−14,324.4 ± 11,574.6（mean−2σ = −37,473.5）、A 149,533.6 対 B 135,209.2、ms/move 69.47 対 96.12。**1e-3 は不採用**
（row 16 に候補として残す）。wasm は 540,957 B（29 ページ）→ 566,031 B（同ページ、data +24,628 B と code +446 B）。`clear` を
`@splat` にすると同サイズ定数が data にもう 1 つ入って 590,156 B になるため 1 エントリずつ代入する（約 24 KiB 減）。

テスト `node TT defaults on for 3-ply and never for 2-ply (#266)` が、2-ply 既定 = 明示 off と同じ手・同じ leaves/hits で probe 0、
3-ply 既定がクリア・probe・hit することを固定。

## 凍結 target net で bootstrap を止める `tnet` は短いゲームに流れる（#262）

`nn.Options.target_net`（既定 null = 従来とバイト一致）。`bootstrapValue` が `target_net orelse net` を引く。効く経路は 4 つ
（`td1` の 1 手 1 更新、`replay_k > 0` の online 更新（`tdBootstrap` を通らない）、backward/replay の末尾、priority の追加更新）。
終端の最終 afterstate は reward だけ。CLI variant `tnet` は `par` 必須・arg 13 = 何バッチごとに同期（既定 100）・scratch 開始。
`par` の各バッチは同じ target を読み、バッチ終了後にだけ live をコピーする。arg 13 の 0 は `error.InvalidTargetEvery`。
契約テスト: `target_net = null` で live と一致、lr=0 のゲームは live のみと完全一致、3 経路すべてで更新後の重みが変わる、
`TargetSync` は同期間隔の途中でコピーしない。

300k（128/32、lr 0.003、`par`、sqrt 8、seed 2、対照 `replay 30`）:

| 条件 | moves / train s | 最後のゲーム | 2-ply 9000 / 9100 | 判定 |
| --- | ---: | ---: | --- | --- |
| `replay 30`（対照） | 539,243,727 / 3,184.9 | 65,476 | 66,186.0 / 66,126.8（0/20） | 安定、裾なし |
| `tnet 30`、同期 100 | 79,205,713 / 694.2 | 3,156 | 7,213.4 / 7,784.8（0/20） | 崩壊（267 手/ゲーム） |
| `tnet 30`、同期 100（seed 1） | 80,097,021 / 627.9 | 3,480 | — | 崩壊 |
| `tnet 0`、同期 100 | 90,486,323 / 649.6 | 6,324 | — | 崩壊（302 手） |
| `tnet 0`、同期 1 | 108,272,536 / 753.4 | 5,480 | — | 崩壊（361 手） |
| `tnet 30`、同期 1、`batch_scale` 0.25 | 101,230,912 / 742.4 | 4,352 | — | 崩壊（337 手） |

`pairw` は同梱に対して **−112,444.4 ± 8,419.5**（0 勝 20 敗）。target を同梱 net に固定したまま（同期を実質無効化）5,000 ゲームだけ回すと
最後のゲーム 31,356 点 / 5,708,577 手で安定するので、`target_net` の読み出しと `TargetSync` は壊れていない。同期間隔・`replay_k`・
`batch_scale` を変えても 300k では同じ崩壊なので、原因は「greedy 1-ply の on-policy データで live から target を更新する閉ループ」。
target を固定した場合だけ値が保たれるが、それは学習ではなく同梱 net への回帰。**不採用**（opt-in variant として残す）。

## 対称変換を 1 手ごとにランダムに引く `rsym` / `rtgt` は発散・悪化（#271、#273）

`rsym`（`nn.Options.rsym`、既定 false）: `trainGamePerMove` の TD 更新で `p.after` を `T_k(p.after)`
（`k = rng.uintLessThan(usize, symmetry_count)`）に差し替える。`distill_only` は対象外、`base` は rng を引かず従来と同一。100k base は
134,574,330 moves / 767.2 s で完走（296,472 B）だが、同じ設定の `rsym` は `par` で 40/100000 局、serial で 8/100 局で
`diverged (non-finite or |w| > 1e6)`。初期化直後の net で 10 局続けると max|w1| が 0.2500 → 0.3170 → 0.8955（6 局目）と増え、非対称 probe の
V は 0.16–0.37 → 13.2–16.1 → 負 → 0.87–0.93。spawn 列だけ rsym と同じ対照は安定するので乱数列の変化が原因ではない。解釈: `r + V(s')` の
`V(s')` は方策が常に正準の向きで評価する値なので、net が対称になる前は回した入力の目標がその盤面の値にならず、正準の補正が 1/8 しか
入らない。lr 1/8 なら 5,000 局は完走する（smoke のみ）。**不採用**。

`rtgt`（`nn.Options.rtgt`、既定 false）: 1 手につき 1 回 `k` を引き、入力 `p.after` と目標の `next.after` を同じ `T_k` で回す
（`t2`/teacher は `spawned` を回す）。終端は入力だけ回し目標は reward。5k smoke 完走、100k は 100,000 局 / 28,086,194 moves / 208.8 s・
最終局 6,964 点 / max|w1| 0.61（base は 134,574,330 moves・3.25）。`pairw` 20 局 **−77,064.0 ± 5,507.7**（0 勝 20 敗）、非ペア 2-ply は
seed 9000 10,159.6 / seed 9100 12,042.8（base は 83,796.0 / 93,993.4）。`eval-net.sh` は REJECT（paired margin −88,079.4）。解釈: 同じ
`k` で回しても `V(T_k next.after)` は「正準盤面で方策が選んだ手の回転」の値で、回転盤面で方策が選ぶ最善手の値ではない（off-policy
バイアス）。値が縮み短い局（281 対 1,346 手）に流れた。契約テストは各 k で指定どおりの更新になることを固定しているので実装ミスではない。
**不採用**。回転盤面で search し直す `t2` 経路ならこのバイアスは入らないが未測定。

## h2 を 32→64 に広げると 2-ply が大きく下がる（#275）

h1=128 固定で h2 32→64、300k（`base`、lr 0.003、`par`、sqrt 8、`replay_k=30`）: 300,000 局 / 503,885,360 moves / 4,923.5 s /
最終局 76,448 点 / 329,752 B / SHA-256 `d1d063755f4a19ff933a10843173b3460fa16cdd3afffb8dcdf712de9727f9e0`、move cap・発散なし。
パラメータ 37,057 → 41,217（1.11 倍）、学習 61 分 → awake 82 分（1.34 倍）。A = 同梱 128/32 を `scripts/widen-net.py` で 128/64 に
ゼロ拡張（非ペア 113,682.2 / 129,456.2 を再現）:

| net | seed 9000 | 8192 | ms/move | seed 9100 | 8192 | ms/move |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| A: 128/32 ゼロ拡張 | 113,682.2 | 11/20 | 1.47 | 129,456.2 | 14/20 | 1.41 |
| B: 128/64、300k | **75,074.0** | 0/20 | 1.63 | **74,915.0** | 0/20 | 1.73 |

`pairw` 20 局: **−60,678.0 ± 8,993.0**（1 勝 19 敗、mean−2σ = −78,664.0）。**判定: REJECT**（score / reach / ms すべて fail。
ms は seed 9100 で +23%）。h2 は 32 が上限で、128/32 は現行レシピの局所最適。次の lever は報酬関数・局面表現（#184 / #185）。

## 空きマス potential による reward shaping（#277）

報酬系の次の候補として、空きマス数だけを使う potential-based reward shaping を実装した。
盤面 `b` の potential を `Phi(b) = shape_c * empties(b) / 15` とし、非終端の TD 目標を `reward + Phi(next.after) - Phi(p.after) + V(next.after)`、
終端を `reward - Phi(p.after)` にする（γ=1、終端 `Phi=0`）。`nn.Options.shape_c`（既定 0）、CLI は variant `shape` の arg 13 に係数
（既定と `-` は 1、0 は `base` と同じなので拒否、負値は有効）。`shape_c == 0` は生の報酬を返す分岐にして `reward + 0.0` を挟まないので
既存の `base`/`par` は bit-identical（5 局 checksum `3042928886` も不変、テストで固定）。potential shaping を実装していない経路
（replay、tnet、teacher、nstep、death、backward、priority、distill、rsym、rtgt、lambda、`td1`/`greedy1` 以外）は `validateOptions` が
`error.ShapeRequiresPlainTd1` で拒否する（afterstate を再学習する経路で生の報酬に黙って落ちると指定の意味が消えるため）。

## 空きマス potential shaping は 100k の一次判定で対照に負ける（#278）

100k・128/32・train seed 1・lr 0.003・`par`・sqrt 8・`replay_k` 30 で `shape_c=0.5`（`Phi = 0.5*empties/15`、1 空き 0.0333 は平均
score_delta/1000 の 0.03〜0.04 と同じ桁）:

| run | moves | 学習 | 最終局 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 対照 `base` | 134,574,330 | 777.1 s | 54,340 | `5cd1ffc6f54f7d81db07595ee366247a9ecea35da09e28f811ffa5d52fd09148` |
| 候補 `shape 0.5` | 122,551,827 | 743.5 s | 36,144 | `14ca7e0e1c218759759dcaec5c58e41af64049c223983ff384327361d3ead16b` |

候補は moves が 8.9% 少なく 1 局が短い。`pairw` 20 局 **−12,442.4 ± 6,702.5**（7 勝 13 敗、mean−2σ = −25,847.4）。非ペア 2-ply:

| net | seed 9000 | seed 9100 | 8192 |
| --- | ---: | ---: | --- |
| 対照 | 83,796.0 | 93,993.4 | 4/20・5/20 |
| 候補 | 75,298.0 | 80,757.2 | 1/20・2/20 |

**判定: Gate 1 不採用**（300k は回さない）。γ=1・終端 0 の potential shaping はエピソード収益を定数 `−Phi(start)` ずらすだけで最適方策は
変わらないが、TD 目標が状態ごとに `V − Phi` へ移り、空きマス数は候補手の afterstate 間で異なるので近似 V では greedy の順位が変わる。
学習中の 1 局が短くなり（134.6M → 122.6M moves）、評価でも 3,729→3,408 / 4,138→3,631 手と短命化して score と裾が落ちた。空きマス入力が
100k で差を失う #226 の観察とも整合し、この方向は 100k で打ち切る。

## 3-ply の spawn カットと深さ配分（#281、#256、#259、#257、#261）

`deep3` の chance node は各空きに 2/4 を置く枝（`ai/solver/nn.zig:692`、`ChanceBranch.p = cprob * spawn.p` 同 `:695`）。`depth > 1` で
`branch.p / empties < cutoff` の枝を落とし（同 `:999`）、残りを再正規化する。既定 cutoff 1e-2 は 4 タイル枝を空き 11 以上で刈る
（2 タイル枝は 16 空きでも刈られない。ただし `cprob < 1` の後続ノードでは 2 タイル枝も対象になり得る）。

| row | 変更 | 40 局 avg_score | 8192 | leaves/move（seed 9000 / 9100） | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| 15 | 既定（cut 1e-2、TT on） | 137,341.6 | 33/40 | 10,574 / 10,786 | — |
| 18 | cut 0.02 | 135,084.8 | 30/40 | 5,174 / 5,233（−51%） | 不採用 |
| 19 | cut 0.05 | 130,918.0 | 27/40 | 3,654 / 3,594 | 不採用 |
| 16 | cut 1e-3 | 137,251.7 | 32/40 | 15,349 / 14,699（+36〜45%） | 不採用 |

- #281（cut 0.02 / 0.05）: 40 局平均で row 18 は −2,256.8、row 19 は −6,423.6。common-spawn `pairrow` 20 局は −9,258.4 ± 15,571.7
  （8 勝 12 敗、mean−2σ = −40,401.8）。node hit は 19.9% → 16.7%。ms/move は負荷依存で、同一 run ログの seed 9000 は
  10.00 / 5.01（2.00 倍）、`pairrow` は 38.09 / 18.40（2.07 倍）——安定した 2.62 倍ではない。
- #256（cut 1e-3）: 40 局で差 −89.9 点（seed 9000 −7,315.2、seed 9100 +7,135.4 と逆転）、ms/move は約 1.53〜1.58 倍。
- #259（`phase_h1` = 空き 8 以上 2-ply / 未満 3-ply）: 20 局 130,887.8 / 13/20 対 row 15 の 140,438.2 / 17/20 で **−9,550.4 ± 10,052.8**
  （mean−2σ = −29,656.0）。leaves/move は −5.3% だけ、ms/move は 1.17 倍遅い。空き 8 以上の 2-ply パートはゲームの大半を占める
  空き 8 未満の 3-ply のコストに埋もれる。
  40 局（seed 9100 を追加）でも順位は変わらない:

  | policy | seed 9000 | seed 9100 | 40 局 avg | 8192 | leaves/move (9100) | ms/move (9100) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | row 2 `deep` | 113,682.2 | 129,456.2 | 121,569.2 | 25/40 | 969 | 1.45 |
  | row 5 `phase_h1` | 130,887.8 | 136,903.4 | 133,895.6 | 28/40 | 10,054 | 14.85 |
  | row 15 `deep3` | 140,438.2 | 134,245.0 | **137,341.6** | **33/40** | 10,786 | 16.56 |

  seed 9100 の 3 行は 1 プロセスで共通 spawn 列を共有して測った（`zig build nn -Doptimize=ReleaseFast -- 0 20 9100 0.003 1 base 30
  ai/solver/nn_weights.bin 1 - - - 2,5,15`）。3 行を 1 プロセスで回すと ms/move は単独 run より大きくなるので、行間の比だけを見る
  （h1 は deep3 の 0.90 倍）。40 局の差は h1 − deep3 = −3,446.0、seed 9100 の同時比較で +2,658.4、seed 9000 の `pairrow` で
  −9,550.4 ± 10,052.8 と符号が割れ、採用目安 `mean − 2σ > 0` を満たさない。leaves/move は 9100 で −6.8%、9000 の 20 局では −5.3%。
- #257（`phase_h2` / `phase_h3`、20 局 × 2 seed）: 4 局の一次判定（row 15 129,335.0 / 3/4 に対し h2 +880.0 ± 21,438.0、
  h3 +14,725.0 ± 27,938.9）を、同じ同梱 net・同じ seed・同じコードで 20 局ずつに広げた。seed 9000 は
  `zig build nn -Doptimize=ReleaseFast -- 0 20 9000 0.003 1 base 30 ai/solver/nn_weights.bin 1 - - - 2,15,6,7` の 1 プロセス、
  seed 9100 は h2/h3 をそれぞれ単独プロセスで測った（row 2 / row 15 の 9100 は #259 の 20 局値をそのまま使い、seed 9000 の
  row 2 / row 15 はこの run と一致した）。

  | policy | seed 9000 | seed 9100 | 40 局 avg | 8192 | leaves/move (9000) | ms/move (9000) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | row 2 `deep` | 113,682.2 | 129,456.2 | 121,569.2 | 25/40 | 996 | 0.89 |
  | row 6 `phase_h2` | 133,926.4 | 134,869.6 | 134,398.0 | 30/40 | 11,590 | 9.94 |
  | row 7 `phase_h3` | 148,547.6 | 138,872.0 | **143,709.8** | **33/40** | 12,701 | 11.34 |
  | row 15 `deep3` + TT | 140,438.2 | 134,245.0 | 137,341.6 | **33/40** | 10,574 | 9.69 |

  - h2 は 40 局で deep3 −2,944.0、8192 は 30/40 対 33/40 で既定に負ける。leaves/move +9.6%、ms/move 1.03 倍。
  - h3 は 40 局で deep3 +6,368.2（+4.6%）。seed 別でも +8,109.4 / +4,627.0 と同符号で、8192 は 33/40 で並ぶ。ただし採用線
    `mean − 2σ > 0`（≈ +15,000 点）には届かず、leaves/move +20%、ms/move 1.17 倍。共通 spawn の 1 ペア（seed 9000 の 1 局目）は
    deep3 174,680 対 h3 132,868（−41,812、36.97 / 46.73 ms/move）で、4 局の一次判定（+14,725.0）と符号が割れる。差は 40 局の
    ノイズ帯（±9,000〜13,000）の中なので、**既定は `deep3` のまま、h2/h3 は opt-in の eval row に残す**。
  - 20 局の `pairrow` は 1 ペア 8.8 分（A 268.7 s + B 260.2 s）で 20 ペア ≈ 3 時間かかり、しかも per-game diff の σ が ~40,000 なので
    この差を 2σ で切るには足りない。同じ工数なら非ペア 40 局を積むほうが安い。
- #261（`deep3_recover` = 刈った枝も子盤面から一段浅く評価して全 spawn を元の確率で加重、row 20）: seed 9000 の 20 局で
  +7,453.2（12 勝 8 敗、stderr 10,695.1、mean−2σ = −13,937.0）、leaves +0.5%、8192 は 17→16。旧測定の 40 局は +2,564 だが
  seed 9100 で −2,325.2、pooled ms +0.8%。旧 tape 比較 20 局は +17,141.2 ± 12,838.7（10 勝 9 敗 1 分、mean−2σ = −8,536.1）。
  **どれも採用目安 `mean − 2σ > 0` を満たさず既定にしない**（row 18/19/20 は再現用に opt-in で残す）。

既定 cutoff 1e-2、既定方針 `deep3` + node TT を維持。

## extended-TD 論文は全文が入手できず実装しない（#291）

Wang・Matsuzaki, "Refining Evaluation Functions for Game 2048 by Extended Temporal Difference Learning," *IEEE Transactions on Games*
18(1), 56–65 (2026), DOI 10.1109/TG.2025.3606517（IEEE Xplore document 11151781）。abstract と導入冒頭のみ公開で、続きは
"Sign in to Continue Reading"。Crossref の全文 PDF は取得できず、OpenAlex `best_oa_location: null`、Semantic Scholar の
`openAccessPdf.url` は空、arXiv と著者所属の検索でも著者版が見つからない。abstract は 2 つの refinement algorithm と NN / n-tuple の
スコア向上を述べるが、更新則・前提・評価条件は不明で、別実装の数値を期待値にはしない（#107 の optimistic initialization /
multi-stage TD、#231 の Adam と混同しないため方法を推測して実装するのは不適切）。

**判定: no-go**（全文または著者稿が入手できたら方法と互換性を確認して再検討）。

## replay 掃引を TD(λ) にする推測実装は既定にしない（#300）

このプロジェクトの refinement は 128/32 net の fine-tune で、末尾 `replay_k`（=30）局をやり直す掃引が本家の refinement に当たる。
その掃引は `trainAfter` による **TD(0)**（1 局ごとに独立の one-step 更新）だったので、1 本の accumulating eligibility trace での
**TD(λ)**（backward view）に置き換えた（#72 の `lambda` は online per-move の trace で、掃引側は未着手だった）。opt-in のみ:
`nn.Options.sweep_lambda`（既定 0 = 従来の TD(0) で bit-identical）、variant `lam`（scratch + `replay_k` + TD(λ)、arg 13 = λ、既定 0.7）、
`rlam`（同じものを resume）。

128/32・scratch・`par`・lr 0.003・`replay_k` 30・train seed 1、評価は row 1 = nn + 1 spawn lookahead、seed 9000..9099 の 100 局:

| λ | 20,000 games | 100,000 games（seed 1） | 100,000 games（seed 2） |
| --- | ---: | ---: | ---: |
| 0（TD(0)） | **13,196.0**（782.8 手） | 16,081.1（920.8） | 18,104.5（1,015.5） |
| 0.3 | 13,065.1 | 17,775.5 | 18,029.5 |
| 0.5 | 12,238.1 | 13,934.4 | — |
| 0.7 | 10,656.5 | — | — |
| 0.9 | 10,732.6 | — | — |

20k では λ を上げるほど単調に悪化（λ0.5 で −7.3%、0.7/0.9 で −19%）。100k の λ0.3 は seed 1 で +1,694.4、seed 2 で −75.0（平均 +809.7）で
λ0 の seed 間差 ±1,012 の内側。λ0.5 は 100k でも −13.3%（best tile 1024 止まり）。総 moves も λ とともに減る（20k: 7.31M → 5.51M）——
trace が末尾の短い trajectory に長く効き、伸びる方向に働かない。

副産物: **収束済み net の fine-tune（`rlam` = resume + `replay_k`）は λ に関係なく即発散**する（`par` で 2,552/20,000 局、serial で
5/20,000 局で `|w| > 1e6`）。online 側の bootstrap が自分の 1-ply `pick.value` で、stale な盤面に replay が乗る形になるため（`deepStep` の
注記と同じ機構）。「収束済み net を refinement する」という論文の前提自体が今の self-play bootstrap では測れず、測るには `tnet` の凍結
bootstrap か teacher label（`tch` / `dist`）で target を切る必要がある。

**判定: no-go**（`sweep_lambda` は既定 0 の opt-in、variant とテストだけ残す）。

## 4-ply の spawn を有限個サンプルすると同じ予算で deep3 に勝てない（#320）

#267 の full-width `deep4` は `deep3` の約 26 倍の葉を読み実測でも負けたので、評価関数と合法手を変えず **chance node の spawn だけを
有限個サンプル**して 4-ply を 3-ply と同じ時間予算に収める案を試した。

実装（`Search` / `Depth` / `Options`、row 21–26）:
- 各 chance node で branch を**重み付き・非復元抽出**。重みは `cprob × spawn.p`（2 タイル 0.9 / 4 タイル 0.1 を空きに均等）。
  Efraimidis–Spirakis の key `-log(u)/p` を全 branch で計算し小さい順に K 個を残す（`sampleMask`）。
- 残した branch の p の和で**残存 mass に再正規化**するので、kept 内の 2:4 は 0.9:0.1 のまま、スケールは full-width の平均と揃う。
  復元抽出ではないので K がそのまま葉の分岐数。
- 乱数は**ゲームの spawn tape を消費しない**。node の board hash・chance depth・`sample_seed`（既定 `0x9e3779b97f4a7c15`）から splitmix64 で
  作る per-branch 一様乱数なので同じ盤面・深さ・seed は常に同じサンプル → 再探索は同じ手、node TT の値と一致、`pairrow` の A/B は
  同じ tape で比較できる。`cprob` は key に混ぜない（全重みを同倍するだけでサンプル集合も再正規化値も変わらない）。
- `ValueCache` の key は不変（葉はサンプルに関係なく net の exact V）。`sample = 0`（既定）は bit-identical。
- `sample_min_depth`（既定 1 = すべての spawn 層）を 2 にすると**葉の直上の spawn 層だけ full width** になり、最後の 1 手が死ぬかを
  正確に残したまま深い層だけ間引ける（K=2 + min depth 2 が K=3・全層とほぼ同じコスト）。

スクリーニング（1 局 seed 9000、同梱 net）:

| 方策 | row | score | leaves/move | ms/move |
| --- | ---: | ---: | ---: | ---: |
| `deep3` + TT | 15 | 80,708 | 11,914.3 | 42.69 |
| 4-ply K=2 全層 | 21 | 71,420 | 2,938.7 | 10.47 |
| 4-ply K=3 全層 | 22 | 79,540 | 8,500.2 | 30.46 |
| 4-ply K=4 全層 | 23 | 168,352 | 16,161.9 | 57.80 |

K=3 が deep3 の 0.71 倍の葉・0.71 倍の時間で予算に最も近い（K=4 は 1.36 倍で超過）。以後 K=3 を本命、K=2 + min depth 2 を対照にした。

ペア測定（row 15 = A、row 22 = K=3 = B、同じ tape、12 局）: A 153,229 / 10 万点超 10/12、B 105,217 / 4/12、ms/move 比 B/A = 0.78。
平均差 **−48,012.3**（stderr 21,388.0、2 勝 10 敗、`mean + 2σ = −5,236 < 0` で 95% 水準でも deep3 より悪い）。row 25（K=2・
min depth 2）は 4 ペアで **0 勝 4 敗、−100,932 点**（sd 85,815）。

**原因**: サンプルが**死ぬ枝を消す**。空き 6 のノードは 12 branch で 4 タイル枝の重みは 0.1/6、K=3 の非復元抽出で 4 タイル枝が 1 本入る
確率は約 5%（6 本のどれかでも 25% 前後）。残り 3/4 のノードは 4 タイル spawn を見ないまま「安全」に再正規化され、さらに max が
ノイズの乗った推定の上振れを選ぶ（sparse sampling の optimism）。full width が守っていた「最後の 1 手が死ぬ確率」を層ごとに薄く
してしまう。B の負け方は揃っており、12 局中 8 局が 75,560–80,488 点（4096 止まり、3,500–3,700 手）に密集する。

**判定: 不採用**（row 21–26 は opt-in で残す）。次に試すなら K を増やすのではなく「4 タイル branch を必ず 1 本残す層化サンプル」か
「root の各手で同じサンプル集合を共有する common random numbers」で、比較ノイズと tail の欠落を分けて潰す。

## 探索の盤面を packed u64 にしても上限は 5%（#316）

`sample`（1 ms 間隔、macOS）で `nn.bestMoveWithCache` の top-of-stack を数えた。同梱 128/32 net、
`-Doptimize=ReleaseFast`、固定 4 局面（終盤セットは 16384 / 32768 を含む 4 局面）。

| 探索 | 局面 | samples | `nn.forward` | `core.GameState.slide` | `_platform_memset` | `_platform_memmove` | その他 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-ply | 序中盤 4 | 6,022 | 5,569 (92.4%) | 227 (3.8%) | 62 (1.0%) | 33 (0.5%) | 131 |
| 2-ply | 終盤 4 | 6,005 | 5,567 (92.7%) | 178 (3.0%) | 87 (1.4%) | 42 (0.7%) | 131 |
| deep3 | 序中盤 4 | 577 | 542 (93.9%) | 18 (3.1%) | 5 (0.9%) | 0 | 12 |

盤面のコピー・行変換・move は `slide` + `memset` + `memmove` で 5% 前後で、TT key の hash は上位に
現れない。探索を u64 の 4-bit 盤面にしても、葉の net は 16 マス配列を one-hot にするので pack/unpack の
変換は `forward` と同じ支配経路に残る。wasm も同じ Zig コードで、SIMD が無いぶん `forward` の比率は
上がる方向にしか動かないため、wasm の profile は判定を変えないと判断して省いた。

**判定: 不採用（試作しない）**。上限 5% のために探索・TT key・合法手・65536 以上の fallback を二重化する
代償に見合わない。node key を速くするなら、まず `Search` の key 計算そのものを測るのが先。

## 公開ページの平均スコア表示を現行 net に更新（#344）

`policy_info.score` の `nn2` / `nn3` は 2026-10-31 の ranking（seed 1000、5×20、旧 64/32 net）の値のままだった。
11/6 の 128/32 net 差し替え（#233）は重みと golden test を更新したが、この表示文字列は更新していなかった。
デプロイ済み wasm は 128/32 を埋め込んでいるので、ページの「平均 82,772 点 / 116,205 点」は現行 net より
低く見えていた。

同梱 128/32 net を固定した 40 局（seed 9000–9019 / 9100–9119、各 20 局、非ペア）の実測に合わせる:

| 表示 | policy | 40 局 avg_score | 8192 | ms/move |
| --- | --- | ---: | ---: | ---: |
| `nn2` | `deep`（eval row 2） | 121,569.2 | 25/40 | 0.884 |
| `nn3` | `deep3` + node TT（eval row 15） | 137,341.6 | 33/40 | 19.326 |

`nn2` / `nn3` は探索の深さが違うだけで、どちらも同じ net。ページの文字列を `平均 121,569 点` /
`平均 137,342 点` にし、README と静的ページのコメントの条件も「seed 9000–9019 / 9100–9119 の 40 局」に
揃えた。探索・学習・既定の方針は変えていない。

## n-tuple の全対称和 read-out は同じ 1M ゲームで 3 倍（#322）

n-tuple は D4 対称な盤面に対して変換のうち **canonical minimum の 1 つだけ**を index にしていた。8 変換すべての index を
合算する read-out を opt-in で実装し（`Symmetry = .canonical | .summed`、CLI `summed` / `widesum`、PR #342 = 5bef85f）、
同じゲーム数で学習し直して比べた。

- `Entries{canonical, summed}` を 16 セル走査 1 回で作る。`valueSummed` は 8 変換の index を重複込みで合算する。
- 学習は summed のとき各 index を `lr*delta/(8*n_tuples)` 動かす。1 パターンあたりの合計は `lr*delta` で canonical と同値なので、
  学習率スケールは変えていない（1 ゲームあたりに触る index は 8 倍）。
- 表のレイアウトは canonical と同じ（`wide` = 129.25 MiB）。学習済み重みの意味だけが変わる。TC(λ) / delayed TD(λ) は
  canonical index 前提なので併用を `assert` で拒否する。
- 推論コストは 1 葉あたりほぼ同じ。学習は **4.0 倍遅い**（50k ゲームで 18.7 s → 74.9 s、1M で約 65 分）。

学習 train seed 1・lr 0.01・1M ゲーム、評価は seed 9000/9100/9200/9300 の各 20 局（同じ variant で 1-ply と depth 2）:

| 学習 | eval | 9000 | 9100 | 9200 | 9300 | 平均 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wide`（canonical） | 1-ply | 13,945.4 | 16,713.8 | 14,229.2 | 15,361.4 | 15,062.5 |
| `wide` | depth 2 | 31,287.8 | 39,134.0 | 37,630.8 | 33,598.2 | 35,412.7 |
| `widesum` | 1-ply | 64,809.2 | 63,400.2 | 72,593.4 | 72,731.0 | 68,383.5 |
| `widesum` | depth 2 | 110,408.6 | 101,242.6 | 105,188.0 | 104,421.8 | 105,315.3 |

参照として `wide` 3M（同 seed）は 1-ply 17,107.2 / 16,847.6、depth 2 44,606.4 / 36,625.6（seed 9000/9100）。
3M は `widesum` だと約 5 時間かかり `docs/measurement.md` の 4 時間上限を超えるため、比較は 1M に揃えた。

- depth 2 で **35,412.7 → 105,315.3（3.0 倍）**、1-ply で 4.5 倍。8192 は seed 9000 の depth 2 で 9/20（対照 0/20）、
  4 seed すべてで best tile 8192（対照は 4096 止まり）。
- 4 seed の平均差は per-game stderr（2,000〜4,000）に対して 20σ 超。train seed 2 は未実施。
- 1 行の depth 2 は leaves/move 5,284.9・ms/move 0.43（同じ `wide` 1M の対照は 6,283.7・0.72。別プロセス計測なので
  比は目安）。手数が 4,800 手まで伸びるため 1 局の wall は長くなる。
- 解釈: 8 変換の平均は推論時の価値の分散を下げ、価値が盤面の向きに依存しなくなる。学習量（1 パターンあたりの delta）は
  変えていないので、伸びは「8 オービットすべてに更新が届く + 推論が平均になる」ぶん。

**判定: 採用（opt-in の `widesum` / `summed` として出荷、既定の canonical と同梱表は不変）**。既定を summed にするには
学習済み 129 MiB の表を同梱する必要があるので、量子化（#346）と蒸留（#347）の結果を見て別に決める。

## L 字 6 マスを 8 本目に足しても 1M ゲームでは seed 変動を超えない（#323）

`.wide` の 7 本は矩形だけで、L 字 6 マス（`{0,1,2,4,8,12}`）の orbit は 1 本も入っていない。8 本目として足す opt-in を実装し
（`Set.lshape`、CLI `lshape` / `lshaperesume`、PR #341 = f33a7f6）、`.wide` のファイルレイアウトは変えずに（`wide_count = 7`）
新しい表を 1 本追加した。表は `wide` の 129.25 MiB から **193.25 MiB（1.50 倍）**。

1M ゲーム・train seed 1・lr 0.01・TD(0)/canonical、評価は seed 9000/9100/9200/9300 の各 20 局（`wide` の対照は #322 の節と同じ run）:

| seed | `wide` 1-ply | `lshape` 1-ply | `wide` depth 2 | `lshape` depth 2 |
| --- | ---: | ---: | ---: | ---: |
| 9000 | 13,945.4 | 16,781.8 | 31,287.8 | 39,256.8 |
| 9100 | 16,713.8 | 15,691.0 | 39,134.0 | 40,705.6 |
| 9200 | 14,229.2 | 13,054.4 | 37,630.8 | 36,041.4 |
| 9300 | 15,361.4 | 13,675.6 | 33,598.2 | 42,889.8 |
| 平均 | 15,062.5 | 14,800.7 | 35,412.7 | 39,723.4 |

- depth 2 は平均 **+4,310.7（+12.2%）** だが per-seed の差は +7,969.0 / +1,571.6 / −1,589.4 / +9,291.6 と符号が揃わず、
  paired の stderr ≈ 2,505 で `mean − 2σ ≈ −699 < 0`。`docs/measurement.md` の採用線（≈ +15,000）には遠い。
- 1-ply は平均で −261.8（−1.7%）と改善なし。8192 到達は 4 seed × 20 局で 0（best tile も全て 4096）。
- 費用は表 1.50 倍・ゲームあたりの index アクセス 8/7 倍・1M ゲームの学習が 21 分（`wide` は 14 分、同じ機械の wall）。

**判定: 見送り**（`lshape` / `lshaperesume` は opt-in で残す）。#106 の「矩形 7 本が実質上限」という結論は、L 字を 1 本足しても
1M ゲームでは覆らない。次に試すならパターン追加ではなく、#322 の summed read-out のように**既にある 8 変換を使い切る**方向。

## temporal coherence は 1-ply を 18% 落とし depth 2 の伸びも seed 変動内（#288）

Sutton らの temporal coherence（TC）を opt-in の学習則として実装し（`Learner.temporal_coherence`、CLI `widetc` /
`widedeep_tc`、λ = 0.5 固定、PR #330 = 既にマージ済み）、既定の TD(0) と同じ条件で比べた。線形 TD の枠組みに
h-state の適格性トレースを足すだけで、パターン・表・read-out は変えていない。

1M ゲーム・train seed 1・lr 0.01、評価は seed 9000/9100/9200/9300 の各 20 局（`wide` の対照は #322 の節と同じ run）:

| seed | `wide` 1-ply | `widetc` 1-ply | `wide` depth 2 | `widetc` depth 2 |
| --- | ---: | ---: | ---: | ---: |
| 9000 | 13,945.4 | 11,833.4 | 31,287.8 | 34,208.0 |
| 9100 | 16,713.8 | 13,253.4 | 39,134.0 | 35,129.8 |
| 9200 | 14,229.2 | 12,893.6 | 37,630.8 | 42,311.8 |
| 9300 | 15,361.4 | 11,219.6 | 33,598.2 | 40,678.6 |
| 平均 | 15,062.5 | 12,300.0 | 35,412.7 | 38,082.1 |

- 1-ply は平均 **−2,762.5（−18.3%）** で 4 seed すべて負け（同符号）。greedy の自己対戦で「楽観的な値」が失われるぶん、
  即時 1 手読みの質が落ちる。
- depth 2 は平均 +2,669.4（+7.5%）だが per-seed 差は +2,920.2 / −4,004.2 / +4,681.0 / +7,080.4 と符号が揃わず、
  paired stderr ≈ 2,345 で `mean − 2σ ≈ −2,021 < 0`。
- `+1 spawn lookahead` も 30,732.8 → 22,938.6（seed 9000）で 4 seed とも 23,000 前後に張り付き、対照より明確に悪い。
- 費用は表も学習時間もほぼ同じ（1M ゲームで 19 分、`wide` は 14 分）。メモリ増は h-state の数だけで無視できる。
- 8192 到達は 0/80。

**判定: 不採用**（`widetc` / `widedeep_tc` は opt-in で残す）。λ = 0.5 固定・1M ゲームでは、greedy 運用の損失を depth 2 で
取り返せていない。λ や replay 長を同時に振るのはこの issue の範囲外（#106 の結論「学習則より read-out とパターン」と整合）。

n-tuple には `pairw` 相当の共通 tape harness が無いため paired 比較は seed ごとの 20 局で代替した。

## delayed TD(λ) は同じ 1M ゲームで depth 2 を 38% 伸ばす（#290）

TC(λ) の実装と同じ論文系譜にある delayed TD(λ)（履歴 h+1 個の TD 誤差を溜め、最も古い afterstate に λ 減衰付きの
和で 1 回だけ更新を当てる）を opt-in で実装し（`Learner.delayed_td_lambda`、CLI `widedelay`、λ = 0.5・horizon = 3 固定、
PR #331 = 既にマージ済み）、素朴な TD(λ) reference との一致テスト（`trainTdLambdaReference`）を付けてある。

1M ゲーム・train seed 1・lr 0.01・TD(0)/canonical、評価は seed 9000/9100/9200/9300 の各 20 局（対照 `wide` は #322 の節と同じ run）:

| seed | `wide` 1-ply | `widedelay` 1-ply | `wide` depth 2 | `widedelay` depth 2 |
| --- | ---: | ---: | ---: | ---: |
| 9000 | 13,945.4 | 16,865.6 | 31,287.8 | 46,272.6 |
| 9100 | 16,713.8 | 15,378.2 | 39,134.0 | 51,835.2 |
| 9200 | 14,229.2 | 14,468.0 | 37,630.8 | 43,111.6 |
| 9300 | 15,361.4 | 14,950.6 | 33,598.2 | 54,096.8 |
| 平均 | 15,062.5 | 15,415.6 | 35,412.7 | 48,829.1 |

- depth 2 は平均 **+13,416.4（+37.9%）**。per-seed 差 +14,984.8 / +12,701.2 / +5,480.8 / +20,498.6 は 4 seed とも同符号で、
  paired stderr ≈ 3,158、`mean − 2σ ≈ +7,101 > 0` で 95% 水準を超える。
- 1-ply は平均 +353.1（+2.3%）で悪化なし。8192 は seed 9100 の depth 2 で 1/20（対照 0/80 全体）。
- `+1 spawn lookahead` は 30,732.8（対照 seed 9000）に対し 24,290.2–31,233.2（平均 27,745）でやや弱い。
- 処理量: 同じ 200k ゲームを冷起動から回すと `wide` 161.2 s に対し `widedelay` 115.6 s（1.40 倍速）。ただしゲーム長が
  変わるので純粋な実装コストではない。実装は「state ごとに 1 回の集約更新」で TD(0) と同じ更新回数、追加メモリは
  h+1 = 4 個の `Indices`（無視できる）。1M ゲームの学習は 20 分（`wide` は 14 分、同じ機械の wall）。
- λ = 0.5・horizon = 3 は固定。λ と horizon の同時掃引はこの issue の範囲外。

**判定: 採用**（opt-in の `widedelay` として出荷、既定の TD(0) は不変）。#322 の summed read-out は canonical index 前提の
delayed 実装と併用できない（`assert` で拒否）ので、**併用解禁が次の最有力**（summed 単独で 3.0 倍、delayed 単独で 1.38 倍）。

## 行テーブル評価の packed 探索は 3-ply を 2.5 倍速くする（#349）

#316 の測定では NN 探索の 92〜94% が `nn.forward` で、盤面を u64 にしても上限は 5% だった。葉が net で
ある限り探索の作りは時間を動かさない。逆に葉を表引きにすれば探索側の速さがそのまま出る。#349 は
その確かめで、nneonneo の実装（3〜14-ply、葉は 8 回の表引き）と同じ形を作って比べた。

実装は `ai/solver/bitboard.zig`（u64 の 4-bit 盤、行 16-bit の merge delta を左右 2 表で 256 KiB、
行特徴を u64 に詰めて 512 KiB）と `ai/solver/bb_main.zig`（`zig build bb`）。評価は `evaluateW` の
行ローカル分解で、行内の順序罰・merge・単調性・合計は行表から、行間の上昇だけ境界項で足す。
スネーク経路の奇数行は行内の上昇も逆順に数えるので、特徴表は反転した行を引く（ここを first/last
だけ反転させると 1 行あたり数点ずれ、3-ply の平均が数百点変わる）。

一致の確認は 2 本: `evalState` が `evaluateStateW` と全 65536 行で一致すること（テスト）、1-ply が
参照 `expectimax` と同じゲームになること（同一 seed で score と最大タイルが一致、テスト）。同梱
128/32 net、seed 9000、40 局（10 runs × 4 局）、`-Doptimize=ReleaseFast`、逐次 1 本。

| 探索 | depth | 40 局 avg_score | 8192 | ms/move | 壁 moves/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `expectimax_tt`（参照、行スキャン葉） | 3 | 58,501.2 | — | 2.1 | 2,120 |
| packed + 行テーブル葉 | 3 | 62,123.2 | 4/40 | 0.8 | 5,510 |
| packed + 同梱 net 葉 | 2 | 92,007.3 | 14/40 | 5.0 | — |

葉 1 回は 30 ns（3,370 万回/秒/スレッド）。nneonneo の 1 move eval が約 100 ns なので、葉の速さは
同級になった。差は深さと評価にある。ただし 3-ply の差 +6.2% は 40 局の stderr（±7,500 前後）の中に
入るので、**強さは変わらず速さだけが成果**。

net 葉に差し替えると 2-ply で 5.0 ms/move、92,007 点で、`nn2`（0.884 ms/move、121,569 点）に
速度でも点でも負ける。探索の枝刈りが緩く葉数が 6 倍近いためで、探索を速くしても葉が net のままでは
#316 と同じ壁に戻る。

wasm へは出さない。表だけで 768 KiB（merge 256 + 特徴 512）、置換表 1 MiB で、初期 2.44 MiB・
`memory.grow` なしの予算に入らないうえ、点は 62,123 で `nn2` の 121,569 を下回る。

**判定: 採用しない（wasm の既定は `nn2` / `nn3` のまま）**。ただし「探索の速さは律速ではない」を
測って示したので、次の一手は学習（#346 / #347 / #350）に寄せる。

## 全 spawn 期待値の TD(0) は同じ 20k ゲームで 2-ply を 2 倍にする（#321）

`expspawn` は afterstate の target を「実 spawn の標本」から「全空きマス × {2: 0.9, 4: 0.1} の期待値」に
置き換える（`Options.expected_spawn`、`expectedSpawnValue` が `bestAfterValue` を平均する。spawn RNG は
消費せず、重みも固定）。盤面 1 枚について最大 32 分岐 × 最大 4 手の forward を回すので、1 手あたりの
計算は素直に 2 桁増える。

同条件（128/32、`par`、train seed 1、20k ゲーム、lr 0.001）で `base` と `expspawn` を学習し、
pairw 20 局（同一 spawn tape）と非ペア 2-ply 20 局で比べた:

| net | 1 ゲームの move 数 | 学習 wall | move/s | 非ペア 2-ply seed 9000 | seed 9100 | ms/move |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` 20k（対照） | 523.8 | 219.1 s | 47,800 | 30,655.0 | 31,771.2 | 1.35 / 1.38 |
| `expspawn` 20k | 863.9 | 2,890.5 s | 5,977 | 61,617.4 | 57,553.2 | 1.14 / 1.18 |

- pairw: `mean diff = +23,038.4 ± 5,787.7`（stderr）、better/worse = 17/3、`mean − 2σ = +11,463 > 0`。
  非ペア 2-ply も両 seed でほぼ 2 倍、8192 は 1/20（対照 0/20）。
- 学習コストは 13.2 倍（wall）、move あたり 8.0 倍。ゲームが長くなった分（1.65 倍の move 数）を除いた
  純粋な per-move コストは約 4.8 倍。
- **lr 0.003 では 2,081 ゲームで発散する**（`diverged at 2081/50000 games (non-finite or |w| > 1e6)`）。
  次元が 32 の one-hot 入力に対して target の分散が標本 TD より小さく、同じ lr では歩幅が過大になるため。
  lr 0.001 では 20k ゲーム発散せず、`base` 側も同 lr に揃えてある（対照の公平性のため）。
- 20k ゲームでは同梱 net（300k ゲーム、113,682.2）の 54% に届くところまで。コスト 13 倍を考えると
  300k ゲームで追いつくには 12 時間規模が要る。

**判定: 採用**（opt-in の `expspawn` として出荷、`ExpectedSpawnRequiresPlainTd1` で他の target 変更とは
排他のまま。既定は `base` の標本 TD(0)、lr 0.003 の既定 recipe も不変）。同じゲーム数で 2 倍という結果は
「target の分散を下げる方向が効く」ことの直接証拠なので、#347 / #350 の蒸留側でも target の平滑化として
参照する。配布 net を `expspawn` で作り直すのはコスト 13 倍なので、この issue の範囲では見送る。

## f32 重みと f32x4 の葉で wasm の探索を 1.4 倍速くする（#348）

`nn.forward` は葉 1 回あたり 48 KiB の f64 重みを読み、256 本の行内積を足す。f32 にすれば読む量が
半分になり、4 レーンに割れば依存の鎖も切れる。#348 はその 2 つを分けて測った。

実装は `nn.Net32`（f32 の同じ 128/32 構造）と `nn.forwardF32`（第 1 層は `@Vector(4, f32)` のレーン
別加算、第 2 層は積和 + `@reduce(.Add)`、`-Dnn-simd` のときだけ SIMD）、`2048-NF\0` の f32 ファイル
形式（`encodeF32` / `decodeF32` / `saveF32` / `loadF32`、`zig build nn-f32` で書き出しと計測）、
`Value.f32` と `solver.nnValue()`。学習は f64 のままで、重みだけを窄める。wasm では f32 ファイルを
埋め込まず初回利用時に f64 から窄める（埋め込むと +146 KB になるため）。

変換の誤差は重み 37,057 個で worst 6.996e-7（f32 の丸めだけ）。10,600 盤面の |ΔV| は平均 2.1e-6・
最大 1.1e-5、手が変わったのは 0（near tie 68 件でも 0）。40 局 8 万手では avg_score 46,695 対 47,106、
手数 80,024 対 80,025 で、80,024 手のうち 1 手だけ差が出た。

| 構成 | native ns/leaf | wasm nn2 ms/move | wasm nn3 ms/move | wasm サイズ |
| --- | ---: | ---: | ---: | ---: |
| f64（既定） | 4,405.8 | 7.91 | 287.78 | 574,807 B |
| f64 + `-Dnn-simd` | 3,943.8 | 8.28 | — | 572,942 B |
| f32 | 4,367.9 | 12.45 | — | 575,469 B |
| f32 + `-Dnn-simd` | 668.5 | 5.36 | 205.79 | 573,604 B |

wasm は node 24（V8）で 40 局 8 万手（nn2）と 2 局 501 手（nn3）、`scripts/wasm-nn-bench.mjs`。
f64 の SIMD は速くならない（3,943.8 ns と 8.28 ms/move は誤差）。f32 も SIMD 無しでは速くならない
（V8 が f64 の縮小版を同じように動かす）。**f32 と f32x4 が揃ったときだけ 1.4 倍**になる。

**判定: 採用**（wasm の既定を `-Dnn-f32 -Dnn-simd` にする。サイズは 1.2 KB 減、点は同じ、探索は
1.4 倍速い）。native の既定は f64 のまま（668 ns は `--release` の 8 コア機械で 1 スレッドの値で、
native の律速は探索ではなく評価本数だから）。f32 ファイルは形式の参照物として同梱し、テストで
f64 から窄めた結果と一致することを固定する。

## PopArt 型の target 正規化は同じ 50k ゲームで 2-ply を 58% 落とす（#289）

#240 / #241 で value の標準偏差が 1 桁ずれると学習が鈍ることが分かっていたので、target を running
平均・分散で正規化して出力を戻す PopArt を opt-in（`popart` / `popartresume`、sidecar は
`<weights>.popart`）で入れ、同じ 50k ゲームで比べた。学習は serial（PopArt の EMA は手をまたぐ状態
なので `par` と併用できない。対照も serial に揃えた）。

| 学習（50k、lr 0.003、seed 1） | moves | 学習 wall | 2-ply 9000 | 2-ply 9100 | 8192 | ms/move |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base`（対照） | 61,177,485 | 1,497.5 s | 103,715.0 | 97,263.8 | 10/20, 9/20 | 1.13 |
| `popart` | 41,005,310 | 1,133.8 s | 43,616.0 | 41,329.2 | 0/20, 0/20 | 1.35 |

正規化そのものは動いていて、target の μ=12.425 / var=32.816（σ=5.728）に対し value の μ=12.429 /
var=32.633（σ=5.713）で、分布は追従している。それでも手数が 4,496 → 2,120 に縮み、点は 58% 落ちた。
paired 20 局（seed 9000、A=`base` B=`popart`）は **0 勝 20 敗**、mean diff −47,429.2 ± 6,001.4
（`mean − 2σ = −59,432.0`）。

落ちた理由は target の分散ではなく、正規化で失われる「点そのものの大小」だと考えている。2-ply の
探索は子局面の値を比較するだけなので、単調変換は順位を変えない。ところが学習では、正規化した値を
戻す係数（σ と μ）が EMA で動くため、同じ局面の target が手を追うごとに違う値へ動く。終盤の 1 手
（score delta が 100 点規模）と序盤の 1 手（数点）を同じ lr で扱う情報が消え、深い局面の差が潰れる。

**判定: 採用しない**（`popart` / `popartresume` は実験として残すが、既定は `base`、lr 0.003・50k の
recipe も不変）。値のスケールを揃える目的は #321 の `expspawn`（target の分散を下げる方向で 2 倍）が
すでに満たしている。

## 4×6-tuple の n-tuple は 20M ゲームでも wide の 1M に届かない（#346）

`four6`（6 マスを 4 本、4 × 16^6 = 67,108,864 エントリ、int16 で 134 MiB）を `four6stage`
（lr 0.1、50% で ×0.1、75% で ×0.01、5M ゲームごとに checkpoint）で 4 時間回した。25M ゲームの
予定だったが、1,386 games/s では 4 時間で 20,030,000 ゲームに届いたところで止め、20M の
checkpoint を採点した（seed 9000 / 9100、各 20 局）。

| 学習ゲーム数 | 1-ply (9000) | depth 2 (9000) | 1-ply (9100) | depth 2 (9100) |
| --- | ---: | ---: | ---: | ---: |
| 1M（lr 0.01 線形、`wide` と同条件） | 6,533.8 | 25,656.6 | — | — |
| 5M | 11,446.2 | 30,642.6 | — | — |
| 10M | 10,815.2 | 35,883.8 | — | — |
| 20M | 12,266.4 | 37,784.8 | 11,906.2 | 39,184.6 |

対照 `wide` 1M（7 本 + 2x3 2 本 + L 字、lr 0.01 線形、4 seed 平均）は 1-ply 15,062.5 / depth 2
35,412.7（seed 9000 で 13,945.4 / 31,287.8）。1-ply は 5M から 20M で 11.4k → 12.3k（+7%）しか
動かず、100M へ外挿しても +3% 程度。250k 級は 20 倍以上遠い。depth 2 は 20M で wide 1M と同水準
（seed 9000 で +6,497、9100 で +51）だが、それは探索が埋める差で、表そのものは負けている。

速度は開始直後 4,854 games/s（表が空でキャッシュに載る）、15M〜20M で 1,300〜1,700 games/s、平均
1,386 games/s。100M は約 20 時間で 4 時間予算の 5 倍。f32 の表は 268 MiB、深さ 2 の評価中のピーク
RSS は 770 MiB（表 256 MiB + 探索の値キャッシュ）。

**判定: 採用しない**。6 マス 4 本では 1-ply の分解能が足りず、ゲーム数を 20 倍にしても 7〜8 本の
`wide` に追いつかない。`four6` / `four6stage` は実験として残す。

## n-tuple 1-ply を教師にした蒸留は 128/32 を 57% 落とす（#347）

2-ply self-play の 65,536 局面（max tile ≥ 4096、`dumpp2`、lr 3e-4 で 30 ゲーム・122,139 手。
lr 3e-3 は 1 ゲーム目で発散する）を `four6` の 10M 表で価値付けし（spawn 期待値で mean 6.269、
range [−2.680, 11.967]）、`dist` で 20,000 ゲーム・every 4 / div 3・lr 3e-3 蒸留した。

| 方策 | 9000 | 9100 | 8192 | ms/move |
| --- | ---: | ---: | ---: | ---: |
| 同梱 128/32 | 113,682.2 | 129,456.2 | 11/20, 14/20 | 1.96 |
| 蒸留 20k | 49,125.2 | 47,339.4 | 0/20 | — |

paired 20 局（A=同梱、B=蒸留）は **0 勝 20 敗**、mean diff −69,033.2 ± 8,651.8。

教師（n-tuple 1-ply）の実力は 1-ply で 12.3k、生徒の 2-ply は 113.7k。教師のほうが 1 桁弱く、
蒸留は生徒を弱い評価関数へ引き戻すだけだった。蒸留は「生徒より強い教師」が前提で、n-tuple の
1-ply は `four6` が 20M で 12k に届いても 2-ply の 128/32 には遠い。同じ dump を deep3 の探索価値で
価値付けし直すのが #350。

**判定: 採用しない**（`dist` / `tch` の経路は実験として残す）。

## deep3 の探索値を蒸留しても 128/32 は 40k 落ちる（#350）

2-ply self-play の 65,536 局面（`dumpp2`、max tile ≥ 4096）を同梱 net 自身の deep3 探索で価値付けし
（`tch` の既定モード、65,536 サンプル、6.6 ms/サンプル、431.7 s）、`dist` で 20,000 ゲーム・
every 4 / div 3・lr 1e-3 蒸留した（lr 3e-3 は 1,705 ゲーム目で発散）。

| 方策 | 9000 | 9100 | 8192 | ms/move |
| --- | ---: | ---: | ---: | ---: |
| 同梱 128/32 | 113,682.2 | 129,456.2 | 11/20, 14/20 | 2.07, 2.12 |
| deep3 蒸留 20k | 84,611.4 | 75,101.6 | 3/20, 1/20 | 2.77, 2.88 |

paired 20 局（seed 9000、A=同梱、B=蒸留）は **3 勝 17 敗**、mean diff −40,183.4 ± 10,883.7
（`mean − 2σ = −61,950.8`）。ms/move の +20% は手数が縮んだぶんの 1 手あたり固定費で、探索の計算量は
同じ。

#207 の `dist`（458,752 サンプル、同じ deep3 モード、lr 3e-3）は +1,184 ± 9,370 の引き分けだった。
サンプルを 1/7 にし lr を 1/3 にした今回は明確に負ける。原因は教師の水準ではなく値の意味の違いで、
deep3 の探索値は「spawn 済み局面を 3 手先まで読んだ V」、TD(0) の target は「この手の再帰で得た
afterstate の V」で、蒸留は生徒の V を探索値へ寄せて 2-ply の比較に使う校正を壊す。

**判定: 採用しない**。200k 局面・深さ 5 の教師は 1 サンプル 6.6 ms の 10 倍以上（深さ 5 は 65,536 で
4 時間を超える）で 4 時間予算に入らず、65,536 で負けた方向がサンプル 3 倍で逆転する根拠も薄い。

## 30k polish at the final lr (#autoresearch)

同梱 net を `resume` で 30,000 ゲーム、lr 0.0003（一定）で磨き直した（`base` と同じ per-move TD、
`par` sqrt 則、train seed 1、547 s）。ゲーム数・lr スケジュール・アーキテクチャは変えていない。

| eval (20 games) | shipped | polished |
|---|---|---|
| seed 9000 | 113,682.2 (11/20 8192) | **136,147.2** (17/20) |
| seed 9200 | 134,867.2 (15/20) | **144,437.4** (18/20) |
| seed 9300 | 125,321.2 (14/20) | **129,883.8** (16/20) |
| seed 9000, 40 games | 121,569.2 (25/40) | **135,032.0** (34/40) |

ms/move は 1.03 → 1.07（+4%）。1-ply も 60,864.6 → 62,810.4 と上がっており、#135 の
「微調整は 1-ply を上げ 2-ply を下げる」パターンとは違う。

注意: 同じ 30k/lr 3e-4 を train seed 4 でやると 134,058.2（−809、ほぼ同等）で、磨き直しの効果は
訓練軌道に依存する。採用したのは seed 1 の 1 個体で、レシピとしての再現性は未確認。ゴールデン
テスト（solver.zig の nn / nn2）と f32 ファイルは新しい net で記録し直した。

## 60k polish (second 30k pass)

同梱の 30k 磨き net に、同じ `resume` 30,000 ゲーム・lr 0.0003・train seed 1 をもう1周（566 s）。
4 シード 20 局の合計は 545,257.4 → **584,697.0**（+7.2%）、8192 到達 66/80 → 69/80。

| seed | 30k | 60k |
|---|---|---|
| 9000 | 136,147.2 | 135,419.4 |
| 9200 | 144,437.4 | **150,747.2** |
| 9300 | 129,883.8 | **144,828.0** |
| 9400 | 134,789.0 | **153,702.4** |

ms/move 1.06–1.15。9000 だけ微減なので、磨きは「まだ効くが飽和に近い」とみる。golden / f32 は
新しい net で記録し直した。

### 60k 磨き net のシード外検証

どの判断にも使っていない 2 シード（20 局）で、磨き前の 300k net と比較した。

| seed | 300k (0k) | 60k |
|---|---|---|
| 9500 | 118,945.0 (12/20) | **155,944.2** (19/20) |
| 9600 | 122,604.4 (14/20) | **151,678.0** (19/20) |

+31.1% / +23.7%。6 シード平均は 124.9k → 148.7k（+19%）。磨きの効果はシード選択の
産物ではなく、未知シードでも再現する。

### 同梱 net 更新後の表示スコア再測定（ページの嘘を消す）

60k 磨き net を同梱した後、`policy_info` の表示スコア（`nn` 69,717 / `nn2` 121,569 / `nn3` 137,342）は
旧 net の測定値のままだった。ブラウザのピッカーはこの文字列をそのまま出すので、実際より 19,694 点
低い強さを名乗っていた。文書化された手順（seed 9000、`resume`、`0,2` / `0,15`）で測り直した:

| policy | 旧表示 | 新表示 | 条件 |
|---|---|---|---|
| `nn`（1-ply） | 69,717 | **71,742** | 40 局 seed 9000 |
| `nn2`（2-ply） | 121,569 | **141,885** | 20 局 seed 9000 + 20 局 seed 9100 |
| `nn3`（3-ply + node TT） | 137,342 | **152,463** | 同上 40 局（9000: 157,429 / 9100: 147,497） |

nn2 の上振れは磨きの効果（seed 9000 は選定に使った 4 シードの 1 つ）。README と同じ 20+20 局
プロトコルに揃えた（9000 は 135,419.4、9100 は 148,349.8）。数字は「同梱 net で今測った値」に統一し、ページと実力の
乖離を残さない。

### 反復する兆候: 1-ply が上がり 2-ply が下がる平滑化（5例）

同梱 net の磨きとは別に、「価値関数を滑らかにする／平均する」系の変更は、5回とも同じ署名を出した:
1-ply は改善し 2-ply は悪化する。

| 変更 | 1-ply | 2-ply |
|---|---|---|
| 葉を8対称平均（`sym_leaf` #178） | — | 29,308.6（死 3/20） |
| 単一 run の SWA（avg4/8/16/32） | 64,190.6 → 84,518 | 134,867 → 118,982 / 112,633 / 84,518 / 74,840 |
| 低 lr で 300k（0.002 / 0.0028） | 54,423.8 / 55,792.6 | 101,526.4 / 113,191.8 |
| 60k の後に lr 1e-4 で 10k 尾 | 改善 | 9200 −8,287、9300 −3,598 |
| 別軌道 60k net との重み平均 | 71,091 / 66,575 / 68,412 / 75,099 | 132,255 / 151,912 / 140,198 / 145,380（合計 −2.6%） |

どれも「平均して滑らかにする」操作。1-ply は1手先のわずかな差を比較するので滑らかな `V` が効くが、
2-ply は**相手の最善手に対する2手目の差**を見るので、`V` の鋭い識別力（同じ盤面でも手番・空き数で
価値が跳ねる情報）が本質で、平滑化はそれを壊す。2-ply の改善は「平滑化」ではなく「識別力の獲得」
（60k 磨き）でのみ起きている。今後も 1-ply の改善を 2-ply の根拠にしない。

### 4シードで選んでも外れる: seed-2 磨きの教訓（#26 の追試）

60k 磨きを train seed 2 でもう一度やり、4 シード（9000-9400、選定に使用）で合計 609,407 と
同梱（584,697）を +4.2% 上回った。1-ply は −15% だったので採用しなかった（#26）。

その後、判断に一度も使っていない 9500 / 9600 で測ると **152,174.6 / 147,647.6** で、同梱の
**155,944.2 / 151,678.0** に両方負けた（−3,770 / −4,030）。**4 シードで選んだ差は選定効果**で、
外では反転する。20 局/シードの比較は ±4% の帯を持ち、4 シードでも +4% を確定できない。

以後、net の採否は「選定に使っていない 2 シードで両方勝つ」を必須にする（docs/measurement.md の
非ペア規則の強化）。1-ply の劣化は副次的だが、今回のように主指標の利得が選定効果のときは
判断材料として有効だった。

### ペア比較（`pairw`）で磨きの方向を確認

A = 同梱 60k 磨き net、B = 磨き前 300k net、`pairw` 10 局 seed 9000（同じ出現列を両者で再生）。
結果は mean diff（B − A）= **−9,852.0 ± 13,416.3**、B が勝ったのは 3/10。つまり**同梱 net が
+9,852 リード**で、方向は非ペアの +21,737（20 局 seed 9000）と一致する。10 局では 2σ = 26,833 に
届かず、docs の採否基準（mean diff − 2σ > 0）を満たすには 20 局必要。

注意: `pairw` のゲームは非ペア評価と同じ seed でも**同じ出現列ではない**（出現テープの消費順が
違う）。絶対値（155k〜176k）を非ペアの表と直接比べない。比べるのは差の符号と大きさだけ。

### 同梱 net の汎化（9 seed、すべて未使用 seed を含む）

20 局ずつ、seed 9000 / 9100 / 9200 / 9300 / 9400 / 9500 / 9600 / 9700 / 9800:

| seed | 磨き前 300k | 同梱 60k 磨き |
|---|---|---|
| 9000 | 113,682.2 | 135,419.4 |
| 9100 | 129,456.2 | 148,349.8 |
| 9200 | 134,867.2 | 150,747.2 |
| 9300 | 125,321.2 | 144,828.0 |
| 9400 | — | 153,702.4 |
| 9500 | 118,945.0 | 155,944.2 |
| 9600 | 122,604.4 | 151,678.0 |
| 9700 | — | 151,499.4 |
| 9800 | — | 144,591.8 |

9 seed 平均 148,528.9。比較可能な 6 seed では 124,146.0 → 147,827.8（**+19.1%**）。どの seed でも
135k を下回らず、8192 到達は 17〜19/20、2048 未満は 0/20。ベンチ seed 9200 は 9 seed のほぼ中央で、
最高値ではない（ベンチ用に選んだ seed ではない）。

### 終盤 3-ply（`phase_h6`）は 5 seed で利得なし＝棄却

`phase_h6` = 空き 4 以上は 2-ply、空き 3 以下は `phase_tight_cutoff` の 3-ply。同梱 net で測定:

| seed | 2-ply | h6 | 差 | 備考 |
|---|---|---|---|---|
| 9000 | 141,548.0 | 144,256.0 | +2,708.0 ± 8,531.5 | paired 20 局、10 勝 10 敗 |
| 9200 | 150,747.2 | 148,593.8 | −2,153.4 | 20 局、8192 18/20、3.29 ms/手（3.2 倍） |
| 9300 | 126,404.6 | 137,206.4 | +10,801.8 ± 10,050.1 | paired 20 局、12 勝 8 敗 |
| 9400 | 133,705.2 | 134,452.8 | +747.6 | paired 20 局 |
| 9800 | 145,885.6 | 149,881.6 | +3,996.0 ± 9,395.2 | paired 20 局、12 勝 8 敗 |

5 seed の平均 +3,220（=+2.2%）、いずれも 1.1σ 未満、代償は 3.2〜3.6 倍の ms/手。ベンチ seed では
わずかに負ける。**読みの深さはこれで打ち止め**（製品に 1 手あたりの時間予算を入れるなら別）。

注意: eval row の添字はずれた。4-ply の spawn sample 行が増えた後は h4 が 8、h6 が 9。
`pairrow` の A/B は args[11] / args[12] の**2 引数**で渡す（`2,9` の 1 引数ではない）。row 27 を
指定すると 4-ply spawn sample 3（26〜77 ms/手）を測ってしまう。

### ペア比較 20 局（`pairw`）: 磨きの利得 +16,403

A = 同梱 60k net、B = 磨き前 300k net（`git show 017da7f^:ai/solver/nn_weights.bin`、
sha256 `c98ea27c…`）、20 局 seed 9000、同じ出現列を両者で再生。

- mean diff（**B − A**）= **−16,403.2 ± 13,079.5**、B が勝ったのは 7/20、902.5 s
- 非ペア 20 局の差（135,419.4 vs 113,682.2 = +21,737）と同じ向き・同じ桁
- docs の採否基準（mean diff − 2σ > 0）を満たすには 20 局では足りない（1.25σ）。
  1 局あたりの差が −91k〜+45k と広いため、2σ には 60 局前後が要る。**採否は 9 seed の
  非ペア集計（+19.1%、どの seed も負けない）で判断するのが実用的**。

注意: 以前の 10 局ペア比較は B に lr 4e-4 の磨き net を渡していた（`/tmp/old300k.bin` が
その学習の出力で上書きされていた）。磨き前 net は git から取り直すこと。

### 磨きの train seed 分布（3 標本）: レシピに余力はない

同じ磨き前 net から 60k × lr 3e-4 を train seed 1 / 2 / 3 で回した 4 seed 合計:

| 標本 | 9000 | 9200 | 9300 | 9400 | 合計 | 8192 |
|---|---|---|---|---|---|---|
| seed 1（同梱） | 135,419 | 150,747 | 144,828 | 153,702 | **584,697** | 69/80 |
| seed 2 | 132,256 | 155,704 | 140,198 | 181,249 | 609,407 | — |
| seed 3 | 121,742 | 139,297 | 142,744 | 147,059 | 550,843 | 69/80 |

平均 581,649、標本 sd 29,300（±5%）。同梱 net は平均のすぐ上で、**磨き 1 回の当たり外れが
±5%**。seed 2 の +4.2% は上振れで、未使用 seed では負けた（#32）。つまり「磨きを何回か回して
最良を選ぶ」は選択バイアスで、分散を食い潰すだけで利得にならない。**このレシピは上限**。

補足: 300k base は 3 回の独立実行（553,202,634 moves）がすべて sha256 `c98ea27c…` で
**ビット一致**。同一 seed の再現誤差はゼロで、分散はすべて seed 選択に由来する。

### 容量は 128 が頂点（h1=160 は −38%）

`-Dh1=160` で同じレシピ（300k base + 60k 磨き、lr 3e-3 → 3e-4、par sqrt）を回した:

| seed | 9000 | 9200 | 9300 | 9400 | 合計 | 8192 |
|---|---|---|---|---|---|---|
| h1=160 | 86,882 | 93,071 | 89,585 | 92,080 | **361,618** | 19/80 |
| h1=128（同梱） | 135,419 | 150,747 | 144,828 | 153,702 | 584,697 | 69/80 |

−38%。base の自己対戦スコアは終盤 78k まで伸びるのに 2-ply は 93k 止まりで、**広げた分が
V の較正を壊す**（同じ lr/batch_scale では 160 幅に勾配が足りない）。96 も 192/256 も負けており、
**h1=128 が容量の頂点**。`zig build test` は既定幅 128 のみを検証するので影響なし。
