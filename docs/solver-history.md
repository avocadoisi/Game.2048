# Solver history: what worked

Record of solver variants, their scores, and the decisions made. Numbers come from the
runs in this log (depth 2 unless noted). Early numbers are from the first benchmarks;
the later table comes from `zig build solve` with `runs=10 games/run=20 depth=2 seed=1`.

## Evaluation: the timeline

1. **Corner bonus (expectimax, first version)**: score = empty cells x10 + corner bonus.
   depth 1: avg 3,673 (seed 1). depth 2: avg 7,606 (seed 1), 7,866 (seed 2). depth 3: avg 11,876 on 20 games (about 23 s).
2. **Row order + forced-down penalty**: order 1..16 row by row, largest tile at the bottom-right.
   depth 2: avg 18,596 (seed 1), 19,812 (seed 2). The forced-down penalty (-100 when only down is legal) was added here.
3. **Largest tile at top-left (row order)**: the direction flipped to match the top-left anchor.
   depth 2: 19,198 (seed 1), 19,352 (seed 2). depth 1: 3,343.
   - Bug found while flipping: the first non-empty cell was compared against 0, which penalized large first tiles.
     Fixed by starting the comparison at the first non-empty cell (`?f64`).
   - An earlier result (5,421 / 5,106 for the flipped direction) came from that bug. It was not a real weakness.
4. **Order penalty weight tuned (row order, x30)**. Sweep at empty weight 10, depth 2, mean of seeds 1 and 2:
   x3 about 10.5k, x10 19.3k, x30 22.5k (adopted), x50 20.3k, x100 21.1k.
   Empty weight 5 gave 20.9k and 30 gave 20.0k, so 10 was kept.
5. **Snake order (adopted)**: row 0 left to right, row 1 right to left, and so on.
   Same weights (empty 10, order penalty 30). depth 2: 26,716 (seed 1), 24,935 (seed 2).

## Tried and rejected (not merged)

- **Gap penalty** (empty cells between two tiles on the snake path, x20) together with a **down-move cost** (x50, applied to every move).
  bench seed 1: 20,199 avg, lower than the snake baseline (25,827).
- **Down-move cost only** (x50, gap 0): 24,226 avg, still lower than the baseline.
  - Lesson: the down cost is not useful at this weight. The forced-down penalty already covers the extreme case.
  - The gap penalty was not re-tuned; at x20 it was the main drop. A smaller weight is untested.

6. **Merge bonus x10 (adopted)**: `+10 × (count of adjacent equal non-empty pairs)`, added to the snake evaluation.
   Holdout seeds (200 games each, depth 2): seed 100 26,232 vs 24,627 baseline; seed 300 25,078 vs 24,140.
   - Tried: merge x5 (seed 100 25,428 / seed 300 23,983, not consistent), x15 (24,753), x30 (18,954, clearly worse).
   - Smoothness (sum of |log2 diff| between neighbours, x1 and x3): seed 100 25,015 and 24,866. Within noise, not adopted.
   - Caveat: the bonus counts pairs that have not merged yet, so a move that only forms a pair can score well.
     The old unit test `bestMove takes a merge over a move that does nothing` failed because of this. Its board was changed
     to a full board where only left/right are legal (the old comment wrongly said down did nothing).

## Benchmark (`zig build solve`, runs=10, games/run=20, depth=2, seed=1, ReleaseFast)

| Variant | avg score | avg moves | best tile | speed (games/s) | elapsed |
|---|---|---|---|---|---|
| Row order, order x30 (main before snake) | 20,744 | 1,129 | 4096 | 6.72 | 29.8 s |
| **Snake, order x30 (adopted)** | **25,827** | **1,366** | **4096** | **5.04-5.14** | **38.9-39.7 s** |
| Snake + down cost 50 | 24,226 | 1,311 | 4096 | 5.12 | 39.1 s |
| Snake + gap 20 + down cost 50 | 20,199 | 1,119 | 4096 | 6.42 | 31.2 s |
| Snake + merge x10 (seed 100 holdout) | 26,232 | 1,398 | 4096 | 2.86 | — |

Speed differs run to run (runs were sequential on one machine); treat it as a rough figure.

## Caveats

- Weights were tuned on seeds 1 and 2 (100 games each). The benchmark above uses seed 1 again,
  so its numbers are optimistic. Use a new seed range for the next check.
- depth 1 and depth 3 were not measured for the snake order.
- The evaluation is hand-set (`ai/solver/solver.zig`). An n-tuple table is the planned replacement.

## Next steps

- Re-check the snake order at depth 1 and 3, and on new seeds (e.g. `zig build solve -- 10 20 2 100`).
- Try merge x10 with a deeper search (depth 3) and a time budget.
- Try a smaller gap penalty (e.g. x2 or x5) and a down cost below x50 with the same benchmark.
- Record every new variant in this file with its benchmark line.

## Policy ranking (`zig build ranking`)
Served at `/api/ranking` from `src/web/ranking.json`. Regenerate with:
`zig build ranking -Doptimize=ReleaseFast -- 5 20 1000 > src/web/ranking.json`
(seed 1000 is not used for tuning.) Current result: expectimax (depth 2) 25,267 > greedy 3,424 > random 1,062 (avg score).

## Parallel candidates (ranking seed 1000, 5 runs x 20 games, avg score)
- expectimax_tt (depth 3 + transposition table): 41,216. Same moves as depth 3, about 4x faster (3.4 vs 15.2 s/game on a loaded machine).
- mcts (200 rollouts/move): 29,722 at about 2.0 s/game.
- expectimax (depth 2): 25,267.
- beam (width 32, 3 plies, spawn approximated as 2): 12,696 at about 0.07 s/game.
- greedy 3,424; random 1,062.
Speed rule: no policy slower than depth 3 (about 6.8 to 15 s/game depending on machine load).

## Deep learning: small MLP value net (`zig build nn`)

- Model: `ai/solver/nn.zig`, one hidden layer MLP over afterstates, trained by TD(0) in pure Zig (no dependencies).
- Training: 20,000 self-play games, learning rate 0.003 (default in `nn_main.zig`). Weights go to `ai/solver/nn_weights.bin` (tracked in git and embedded by `solver.zig`; format: 16-byte header + little-endian f64, see `nn.encode`/`nn.parse`).
- Eval on seeds 2000..2099 (100 games, `zig build nn -- 20000 100 2000 0.003`):

| Policy | avg score | avg moves | best tile | time (100 games) |
|---|---|---|---|---|
| NN 1-ply | 21,227 | 1,129 | 4096 | 0.6 s |
| NN + 1 spawn lookahead | 38,734 | 1,919 | 4096 | 36.2 s |
| expectimax depth 2 (baseline) | 25,004 | 1,328 | 4096 | 19.7 s |

- Learning-rate sweep, 20 games on seed 1000 (short run, noisy): 0.0003 gave 5,571 / 14,493; 0.001 gave 11,452 / 20,286; 0.003 gave 18,234 / 34,645.
- Caveats: one training run and one seed range only. The lookahead is not a like-for-like depth comparison with expectimax. It costs about 2x the baseline time. Re-check on more seeds before adopting.

## GA over evaluation weights (`zig build ga-rl`)

- Evolves the 4 snake-evaluation weights (empty, order_penalty, merge, forced_down_penalty) with tournament selection, uniform crossover, and clamped log-normal mutation. Args: `16 15 8 2 1 500 100` (population 16, 15 generations, games/fitness 8, depth 2, seed 1, holdout 500, 100 games).
- Holdout seeds 500..599 (100 games, depth 2): baseline 24,067, evolved 25,095 (+4%).
- Evolved weights: empty 7.028, order_penalty 43.174, merge 0, forced_down 0.
- Caveats: one GA run, one holdout range; merge and forced_down collapsed to 0, so the gain may be noise. Check on another seed range before adopting.

## Policies wired into the solver (`ga`, `nn`)

- `ga`: `solver.ga_weights` (3-decimal values from `zig build ga-rl`) with depth 2. `nn`: embedded `ai/solver/nn_weights.bin` (148 KB, committed), chance-mode net.
- Both are selectable in `/api/solve?algo=ga|nn` and in the UI dropdown (ranking.json regenerated with `zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`).
- Ranking, seed 1000 (5 runs x 20 games, avg score): expectimax_tt 41,216 / nn 36,885 / mcts 29,722 / expectimax 25,267 / ga 24,046 / beam 12,696 / greedy 3,424 / random 1,062.
- The GA weights are lower than plain expectimax on seed 1000. The +4% above holds only on seeds 500–599. Not a clear win.

## NN: longer training (200,000 games, lr 0.003 decaying to 10%)

- `zig build nn -- 200000 100 3000 0.003`: 3,316 s training (same 64/32 MLP, 148 KB weights, checkpoint every 5,000 games). Linear lr decay added.
- Eval seeds 3000..3099 (100 games): NN 1-ply 35,687 (best 4096); **NN + 1 spawn lookahead 66,600 (best 8192, 73.8 s)**; expectimax depth 2 24,839 (37.9 s). Before (20,000 games, seeds 2000..): 38,734.
- Ranking seed 1000 (5x20): nn 64,113 (was 36,885), above expectimax_tt 41,216. Slower than expectimax depth 2 (~2x per game), so it breaks the "no policy slower than depth 3" rule only in time per game of about 0.7 s; still faster than depth 3.
- Not measured: wider net (128/64, runs slower in training, see below), TD(lambda), n-tuple tables.

## Speed work (no change in play)

Goal: faster self-play for AI training. Every row below gives identical moves for the same seed
(checked with `zig build solve -- 2 20 2 1`: avg score 22135.1, avg moves 1219.0).

| Change | speed (games/s, depth 2, 40 games) |
|---|---|
| Baseline | 5.4 |
| `evaluate`: `@log2(value)` -> `@ctz(value)` (tiles are powers of 2, so the result is the same) | 7.3 |
| `core.mergeLine`: 65536-entry row table, fallback to the general path for non-powers-of-2 or >= 65536 | 8.3 |

- Tried and dropped: precomputed slide index table (`slide_lines`) in `core.slide`. No measurable gain (about 27 ns per slide both ways).
- Cost of the row table: about 512 KB in the binary, and about 15-20 s extra compile time in ReleaseFast.
- Remaining cost is mostly the solver (`evaluateState` runs up to 4 slides per call). A bitboard core would be the next big step, but it changes `Board` everywhere.

## Tried and not merged: packed nibble search (branch `exp/packed-search`)

Idea: search on rows packed as 4 nibbles (log2 of each tile), so a slide is one table lookup per row.
- Micro benchmark (slide only, random rows): about 3 ns per slide, against about 28 ns for `GameState.slide`.
- Full version (`core.Packed`, expectimax on packed boards, bit-identical results): `zig build solve -- 2 20 2 1`
  gives the same avg score 22135.1, but speed is about 5.5-5.9 games/s, the same as the merged version.
- Profile after the change: the heuristic (`evaluateStateP`, nibble extraction in the order loop) and
  `Packed.slide` dominate. The board-based heuristic was already cheap enough that the slide cost was not the limit.
- A pack-and-unpack wrapper around `GameState.slide` (no packed state kept) was slower: about 40 ns per slide.
- Conclusion: a real speedup needs `GameState` itself stored packed. That changes about 78 `.board` uses and
  makes tiles of 65536 and above need a separate path. Not done.

## Training focus: near-loss states (proposal)

Problem: `nn.trainGame` runs TD(0) over every move of every self-play game with the same weight.
Most moves are in the safe early and middle game, so the net sees few examples of boards close to
game over. Those boards decide the score, and their values are the least accurate.

Current behaviour (`ai/solver/nn.zig`, `trainGame`): each step calls `train(net, cache, reward + next_value, lr)`
in forward order, one update per move, no replay, no weighting by position.

### Candidate ideas (cheapest first)

1. **Diagnose before changing anything.** Log the per-move TD error `|target - V(s)|` grouped by
   empty-cell count (0..3, 4..7, 8+) and by distance to game over (last 10, 11..30, earlier).
   If the error and the value drop (`V(s_{t+1}) - V(s_t)`) are concentrated near game over, the
   focus idea is worth doing. If not, skip it. This needs only logging, no change in training.
2. **Backward sweep after each game.** Store the (afterstate, reward) pairs of the game and train
   from the last move back to the first. The terminal value propagates to earlier states in one
   pass, so the near-loss boards learn the real outcome sooner. Cost: a small buffer per game.
   Risk: changes the update order, so results differ from the current run; compare on the same seeds.
3. **Replay the last K moves of each lost game.** Keep the final K (e.g. 30) afterstates of each game
   and train on them again, in a separate pass with a smaller learning rate. This oversamples
   the states just before a loss without changing how games are played.
4. **Prioritized replay by TD error.** Keep a buffer of recent (state, target) pairs; sample with
   probability proportional to `|δ|^α` (α about 0.5), and always keep a fraction (e.g. 50%) uniform.
   Large `|δ|` is a good proxy for "the value just dropped". The uniform part keeps the training
   distribution from drifting too far from the one the policy visits.
5. **Restarts from hard boards.** Begin some games from saved boards with few empty cells and
   high tiles (taken from earlier games). Gives direct coverage of late-game states. Riskier:
   those boards may be unreachable under the policy, so keep the share small.

### Safety rules for any of the above

- Focusing on losing states is off-policy. Always mix in uniform samples (ideas 2–4 keep the
  normal per-move update as well).
- Compare with the same total number of moves processed, not the same number of games,
  because replay makes each game cost more time.
- Evaluate with the existing protocol: `zig build nn -- 20000 100 2000 0.003` on seeds 2000..2099,
  then a fresh seed range (e.g. 3000..3099) before deciding. Record every run in this file.
- Keep the diagnostic (idea 1) as a flag so it can be re-run after the training changes.

### Next steps

- Add the diagnostic log (idea 1) to `nn_main.zig` or behind a flag in `nn.zig`.
- If the error is concentrated near game over: implement idea 3 (cheapest) and measure.
  Then try idea 2 and idea 4 on the same seeds.
- Status (#28): all three are done and measured (#32 diagnostic, #33 replay/backward, #34 priority).
  Conclusion below: the error is spread out, so focusing on near-loss states does not pay off.

## Diagnostic run (#32)

Command: `zig build nn -Doptimize=ReleaseFast -- 2000 1 2000 0.003 diag` (2000 training games, seed 1; weights restored afterwards).

| empty | moves | steps | mean \|err\| | mean drop | share err |
|---|---|---|---|---|---|
| 0..3 | last 10 | 17979 | 0.2966 | -0.2193 | 7.9% |
| 0..3 | 11..30 | 20110 | 0.1582 | -0.0045 | 4.7% |
| 0..3 | earlier | 62048 | 0.1669 | 0.0224 | 15.3% |
| 4..7 | last 10 | 2021 | 0.1632 | -0.1198 | 0.5% |
| 4..7 | 11..30 | 19620 | 0.1165 | -0.0388 | 3.4% |
| 4..7 | earlier | 355867 | 0.1002 | -0.0061 | 52.6% |
| 8+ | earlier | 150594 | 0.0702 | -0.0136 | 15.6% |

- Reading: the last 10 moves before game over have about 3-4x the mean TD error of the early game (0.30 vs 0.07-0.10), and the value drops by about -0.22 on average there.
- But they are a small share of the total error (about 8%), because few steps are that close to game over. Most error sits in the 4..7 empty, earlier-move bucket (53%).
- So the near-loss states are badly fit per step, but focusing only on them would cover little of the error. Mix them in (e.g. #33 K-move replay, #34 priority with a uniform share) instead of replacing the uniform training.
- Caveat: one training run, one seed, 2000 games. Re-check with more games before deciding.

## Bench run 2026-10-25 19:31:46 +0900

- command: `zig build solve -- 1 2 2 1`
- commit: 248c1a5
- date: 2026-10-25 19:31:46 +0900
- exit: 0

```text
runs=1 games/run=2 depth=2 seed=1 games=2
avg score=20954.0 avg moves=1148.0 best tile=2048
speed: 0.54 games/s, 621 moves/s, elapsed 3.7 s
```

## Eval variance (#37)

Command (eval only, loads committed weights, writes nothing):

```
zig build nn -- 0 20 1000 0.003 5
```

Seeds 1000..1099 as 5 disjoint ranges of 20 games. Every policy plays the same seeds.
Each range gives one average; the table reports mean ± stderr over the 5 range averages
(stderr = sample sd / sqrt(5)).

| policy | avg score (mean ± stderr) | avg moves | best tile |
|---|---|---|---|
| nn 1-ply | 18851 ± 503 | 1028.1 | 2048 |
| nn + 1 spawn lookahead | 38572 ± 1608 | 1917.1 | 4096 |
| expectimax depth 2 | 24776 ± 1267 | 1319.8 | 4096 |

Reading the stderr:
- nn 1-ply vs expectimax depth 2: difference 5925. Stderr of the difference is about
  sqrt(503² + 1267²) ≈ 1363, so the gap is about 4.3 stderr (about 8 df). Significant at p < 0.01.
  Treat the expectimax lead as real for these weights and this seed set.
- nn + spawn lookahead vs expectimax: difference 13796, stderr about 2070 (≈ 6.7). Clearly significant.
- These stderr values assume independent ranges. Policies share seeds, so the true paired
  difference may be tighter; the independent estimate is conservative.
- Only 100 games per policy in total. Stderr is from 5 range means, so it is itself noisy.
  More ranges (e.g. 10+) or more games per range would tighten it.
- Single-seed-range runs with 20 games (the previous protocol) can not show this: one range
  alone gives no stderr. Use `seed_ranges` ≥ 5 for policy comparisons.
- Wall time: nn + spawn lookahead dominates (394 s for 100 games); 1-ply took 5 s.

## Fair comparison (#38)

Command: `zig build nn -Doptimize=ReleaseFast -- 0 20 1000 0.003 5` (eval only, committed weights, seeds 1000..1099 in 5 ranges of 20 games). Score per second = total score / CPU seconds for the 100 games.

| policy | avg score (mean ± stderr) | avg moves | elapsed (100 games) | score/s |
|---|---|---|---|---|
| nn 1-ply | 18851 ± 503 | 1028 | 0.6 s | 2,957,818 |
| nn + spawn lookahead | 38572 ± 1608 | 1917 | 42.2 s | 91,306 |
| expectimax depth 2 | 24776 ± 1267 | 1320 | 18.2 s | 136,202 |

- Per game, the lookahead is clearly stronger than expectimax depth 2 (about +13.8k).
- Per CPU second, the lookahead is about 33% slower than expectimax (91k vs 136k score/s), so the gain does not hold at equal time.
- nn 1-ply is far cheaper per second but much weaker per game. It is the cheapest option, not the strongest.
- Caveat: one set of weights, one seed range, 100 games per policy. The timing is a single run on a loaded machine.

## Parallel eval (#40)

Eval loops (`zig build nn` and `zig build ranking`) now run games on threads via `ai/solver/parallel.zig` (`std.Thread.spawn`, `min(cpu count, 8)` threads). Each game keeps its own PRNG seeded by its seed, so results do not depend on the thread count. `ranking` parallelizes per run (one PRNG per run, games in order inside a run, the same stream as before).

- Machine: 8 logical CPUs (`hw.ncpu` 8), up to 8 threads (nn: 100 games in 8 chunks; ranking: 5 runs, so 5 threads).
- Command: `zig build nn -Doptimize=ReleaseFast -- 0 20 1000 0.003 5` (eval only, committed weights).

| policy | serial elapsed | parallel elapsed | speedup |
|---|---|---|---|
| nn 1-ply | 0.6 s | 0.1 s | 6x (too short to measure well) |
| nn + spawn lookahead | 44.2 s | 9.9 s | 4.5x |
| expectimax depth 2 | 13.1 s | 2.7 s | 4.9x |
| sum of the three | 57.9 s | 12.7 s | 4.6x |

- Whole command wall time (includes build): 58.1 s serial, 29.4 s parallel. `zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`: 7m00s serial, 2m17s parallel.
- Identical results: nn output lines (avg score ± stderr, avg moves, best tile) match the serial run exactly: 18851.1 ± 503.4 / 1028.1 / 2048; 38572.0 ± 1607.6 / 1917.1 / 4096; 24775.9 ± 1266.8 / 1319.8 / 4096. The ranking JSON is byte-identical (`cmp`) to the serial output for `5 20 1000`.
- Tests: `parallel.map` equals a serial loop (`parallel.zig`), nn games match the serial loop over 8 seeds (`nn.zig`), and `bench.run` matches a serial loop over 8 games (`bench.zig`).
- Caveat: `score/s` in `zig build nn` is now total score per **wall** second, not per CPU second as in the fair-comparison table above. Use the serial numbers for per-CPU-second comparisons.

## Near-loss focus: conclusion (#28)

Closed after #32 (diagnostic), #33 (replay / backward sweep, PR #48) and #34 (priority replay, PR #50).

- Measured: the last 10 moves before game over carry 3-4x the per-step TD error of the early game
  (0.2966 vs 0.0702-0.1002 mean |err|) and the value drops by about -0.22 there. But those steps are
  only about 8% of the total error; 53% sits in the 4..7 empty, earlier-move bucket (#32).
- Measured at equal training budget (20000 games, training seed 1; eval on seed range 2000 and fresh
  ranges 3000..6000, 20 games each): base 18,677 ± 717, replay K=30 14,973 ± 838,
  priority (m=32, alpha 0.5, 50% uniform) 19,488 ± 1,490, backward 2,116 ± 80 (#33, #34).
- So no candidate beats the uniform per-move TD(0) update. Priority comes closest: +811 against a
  combined stderr of about 1,650 (0.5 sigma), and it processed 14.02M training moves vs base's 13.97M
  (+0.3%), so that comparison is effectively at equal moves. Replay is about 3.4 sigma worse, and the
  backward sweep collapses. The 8% error share explains it: oversampling near-loss states mostly
  takes fit away from the mid-game states where half the error lives.
- Candidate 5 (restart games from hard boards) is not implemented. The same 8% argument applies, with
  an added risk: boards saved from earlier games may be unreachable under the current policy, so the
  net would train on a state distribution it never plays. With every cheaper candidate at or below
  baseline, restarting from hard boards is not worth the cost.
- Kept: the diagnostic flag (`diag`) stays in `nn_main.zig` so the error split can be re-checked after
  any future training change. The replay/backward/priority options stay off by default, with the
  default path pinned by the recorded-checksum test in `ai/solver/nn.zig` ("default trainGame weights
  match the recorded checksum (5 games, seeds 11/12)").
- Not done: no equal-moves sweep for replay (it reached 92% of base's moves), K/m/alpha not swept, and
  one training seed only. A wider sweep would need several training seeds before it could overturn the
  conclusion.
## Training parallelization (#35)

Self-play training (`trainGame`) runs in parallel batches. With the opt-in `par` argument the CLI
takes batches of `N = min(cpu count, 8)` games (clamped to the games left). Game `i` uses seed
`train_seed + i` and trains its own copy of the batch-start net; the next net is
`net_start + (1/N) * sum_i (net_after_i - net_start)`, summed in game index order. The sum order is
fixed, so the weights do not depend on the thread count; the serial path (no `par`) is unchanged.
`diag` works with `par` (per-game diagnostics are merged into one table).

- Command: `zig build nn -Doptimize=ReleaseFast -- 2000 1 2000 0.003` (serial) vs
  `zig build nn -Doptimize=ReleaseFast -- 2000 1 2000 0.003 1 par` (parallel), 8 logical CPUs,
  batches of 8 games, 250 batches. Weights restored with `git checkout -- ai/solver/nn_weights.bin`.
- Training time: 9.1 s serial, 1.6 s parallel (5.7x). The parallel run applies the lr of the batch
  start to the whole batch, so it is a slightly different lr schedule from the serial loop.
- Reproducibility: game `i` must match a serial run of the same games with the same seeds. The unit
  test `parallel training batch matches a serial run of the same game seeds` (`ai/solver/nn.zig`)
  runs a 4-game batch on threads and compares the resulting weights byte-for-byte with a serial
  loop over seeds 77..80 that averages the same per-game nets. `zig build test` passes. Two
  parallel CLI runs of the command above also write byte-identical weight files.
- Caveat: the batch average is not the same update as applying the 8 games one after another to a
  single net; it is the aggregation chosen for determinism. `par` is opt-in, so existing commands
  and recorded results above keep the serial behaviour.


## Replay and backward sweep (#33)

Options in `ai/solver/nn.zig` (`trainGameOpts`, `Options`); default off, and the default path is pinned by the recorded-checksum test above.
- `backward`: no per-move update during the game (weights frozen while playing); afterwards every stored afterstate is trained from last to first, target = reward + V(next) with V recomputed.
- `replay`: normal per-move updates, then the last K=30 afterstates are re-trained once, oldest first, at 0.5 x lr with fresh targets.
- CLI: `zig build nn -- <train> <eval> <seed> <lr> <-|diag> <base|backward|replay> <K> <weights_path>`. Variant runs write to a temp path, so `ai/solver/nn_weights.bin` is not touched.

Command per variant: `zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - <variant> 30 /tmp/nnb/<variant>.bin`, then eval-only runs (`-- 0 20 <seed> 0.003 - base 30 /tmp/nnb/<variant>.bin`) on seed ranges 2000, 3000, 4000, 5000, 6000 (20 games each, 100 games per variant).
Mean ± stderr is over the 5 range averages (n=5). The built-in `seed_ranges` argument only covers contiguous ranges (`eval_seed + r*eval_games + g`), so the 1000-apart ranges were run individually and the stderr was computed by hand.

| variant | train games | train moves | nn 1-ply avg score (mean ± stderr) | nn + 1 spawn avg score (mean ± stderr) |
|---|---|---|---|---|
| base | 20000 | 13,969,906 | 18,677 ± 717 | 39,715 ± 1,103 |
| backward | 20000 | 6,272,449 | 2,116 ± 80 | 2,116 ± 80 (same policy, no lookahead gain) |
| replay K=30 | 20000 | 12,811,345 | 14,973 ± 838 | 33,241 ± 740 |

Per range, nn 1-ply avg score: base 19,110 / 16,668 / 19,592 / 17,423 / 20,589; backward 2,288 / 2,101 / 1,821 / 2,196 / 2,176; replay 16,664 / 13,497 / 12,655 / 16,871 / 15,177 (seeds 2000, 3000, 4000, 5000, 6000 x 20 games). Expectimax depth 2 on seeds 2000..2019 was 21,840 in every run (same policy, reference only).

- Reading: neither candidate beats the baseline. Replay is about 20% lower than base (difference 3,704 vs combined stderr 1,103, about 3.4 sigma on the 1-ply policy). Backward collapses to about 2,100 (policy dies after about 200 moves). The collapse is not tuned: only lr 0.003 was tried, with the sweep updates applied to frozen-policy games.
- Moves: backward played far fewer training moves because its policy is weak; replay reached 92% of base's moves. The comparison is at equal games (20000), and the moves are listed above. Equal-moves comparison was not run.
- Caveats: one training seed (1, fixed in the code), K=30 and replay lr 0.5x not swept, backward not swept over lr. The near-loss error share (#32) is 8% of the total, so replaying it at this weight may just shift the fit away from the mid-game states. Next: if replay is kept, try a smaller replay share (K=10, lr x0.25) before deciding.

## Priority replay (#34)

Option `priority` in `ai/solver/nn.zig` (`Priority`, `Options.priority`; CLI variant `priority`). Default off; the default path is pinned by that recorded-checksum test.
- Normal per-move TD(0) updates run for every move, as in base. Each update is also stored in a ring buffer (1024 entries: afterstate, reward, next afterstate, terminal flag, |TD error|). The buffer persists across games.
- After each game, `m = 32` extra updates. Each draws an entry: uniform with probability 0.5, otherwise with probability proportional to `(|δ| + 1e-3)^0.5`. The target is recomputed with the current weights, and the drawn entry's |δ| is refreshed.
- Tests: sampling share checked against the analytic mixture (fixed seed, tolerance 0.01); priority pass changes the weights.

Command per variant: `zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - <variant> 30 /tmp/nnp/<variant>.bin`, then eval-only runs (`-- 0 20 <seed> 0.003 - base 30 /tmp/nnp/<variant>.bin`) on seed ranges 2000, 3000, 4000, 5000, 6000 (20 games each, 100 games per variant). Mean ± stderr over the 5 range averages (n=5), as in #33.

| variant | train games | train moves | nn 1-ply avg score (mean ± stderr) |
|---|---|---|---|
| base | 20000 | 13,969,906 | 18,677 ± 717 |
| priority (m=32, α=0.5, 50% uniform) | 20000 | 14,016,958 | 19,488 ± 1,490 |

Per range, nn 1-ply avg score: base 19,110 / 16,668 / 19,592 / 17,423 / 20,589; priority 18,278 / 14,513 / 23,450 / 20,639 / 20,559.

- Reading: no measurable gain. The difference is +811 against a combined stderr of about 1,650 (about 0.5 sigma). The priority spread is wider (seed range 3000 is the lowest of all runs, 14,513; seed 4000 the highest, 23,450). The base number reproduces #33 exactly (same moves and per-range scores), so the session is comparable.
- Moves: priority played about 0.3% more training moves than base at equal games (14.02M vs 13.97M), so the extra updates did not change how games were played much.
- Caveats: one training seed (1, fixed in the code), m=32 and α=0.5 not swept, 100 eval games per variant. Differences this size would need more training seeds to call either way. The issue's safety rules are met: the uniform share is kept at 50%, and the normal per-move update is unchanged.

## Priority replay across training seeds (#56)

#34 left the only open question: one training seed cannot separate a +811 difference from noise. The training seed is now a CLI argument (positional 9, after `weights_path`, default 1), so the comparison can be repeated per training seed.

Protocol: train `base` and `priority` for each training seed 1..5 (`zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - <variant> 30 /tmp/nnseed/<variant>_s<s>.bin <s>`), then evaluate every weights file on the fixed eval seed ranges 2000, 3000, 4000, 5000, 6000 (20 games each). Score is the nn 1-ply average score. Both variants stay on the serial per-game path: `par` trains the plain per-move variant only.

Per training seed (mean of the 5 eval ranges):

| training seed | base | priority | delta |
|---|---|---|---|
| 1 | 19,611 | 17,413 | -2,197 |
| 2 | 18,011 | 17,107 | -905 |
| 3 | 17,870 | 22,388 | +4,518 |
| 4 | 20,431 | 19,528 | -903 |
| 5 | 17,846 | 19,957 | +2,111 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **19,279 ± 959** | **+525 ± 1,224** (t = 0.4) |

- Reading: priority wins 2 of 5 training seeds and loses 3. The paired delta is +525 against a stderr of 1,224, so the earlier +811 was a single-seed artefact, not a gain. Verdict unchanged: `priority` stays opt-in and off by default.
- The spread across training seeds (base 17.8k-20.4k, priority 17.1k-22.4k) is about 1.5x the difference between variants, which is why one seed could not decide it. Base seed 1 reproduces #34 (19,611 vs 18,677, both inside one stderr).
- Cost: 10 training runs of 20,000 games, 300-410 s each on 8 cores (serial). `par` reaches 2,000 games in 1.5 s against 8.6 s serial, but it cannot run the variant path, so these runs are slow by construction. `par` now also accepts `weights_path`, so a parallel run no longer has to overwrite the committed `ai/solver/nn_weights.bin`.
- Caveats: still 5 training seeds and 100 eval games per variant; m=32 and α=0.5 were not swept.

## Mid-game focus (#59)

#32 put 52.6% of the total TD error in the "4..7 empty cells, earlier move" bucket and only 7.9% in the last 10 moves before game over, so the next candidate was to sample the extra post-game updates from that mid-game bucket instead of the near-loss one.

Mechanism: `Options.midgame` (CLI variant `midgame`). Same shape as `priority` — the per-move TD(0) update is untouched, the ring buffer (1024 entries) is reused, and after each game `m = 32` extra updates are drawn **uniformly from the entries with 4..7 empty cells** (falling back to the whole buffer while that bucket is empty), with the target recomputed from the current weights. Default off; the default path is pinned by the recorded-checksum test (base weights byte-identical, `md5 0a8e6bf01e256adf2fc481317c5072c9` at 2000 games before and after).

Protocol identical to #56: train each variant for 20000 games at 5 training seeds, evaluate on the fixed seed ranges 2000, 3000, 4000, 5000, 6000 (20 games each), nn 1-ply average score. Base column is the #56 measurement re-evaluated with the same binary.

| training seed | base | midgame | delta |
|---|---|---|---|
| 1 | 19,611 | 13,311 | -6,300 |
| 2 | 18,011 | 21,087 | +3,076 |
| 3 | 17,870 | 18,056 | +186 |
| 4 | 20,431 | 15,700 | -4,731 |
| 5 | 17,846 | 17,488 | -358 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **17,128 ± 1,290** | **-1,625 ± 1,710** (t = -1.0) |

- Reading: no gain, and the spread is much wider than base (13.3k-21.1k against 17.8k-20.4k). Two seeds lose 6.3k and 4.7k, which is the same failure mode as `replay` (#33): replacing the uniform sample with a narrow bucket takes fit away from the rest of the state distribution.
- Verdict: `midgame` stays opt-in and off by default. The error-share argument (#32) says where the error is, not that concentrating updates there helps — the last three attempts (`replay`, `priority`, `midgame`) all point the same way.
- Next candidates, in order of expected value: mix the mid-game bucket with a uniform share (the shape that came closest in #34), more training games at the same protocol, and an lr sweep.
- Cost: 5 training runs, 145-358 s each (serial; `par` still cannot run variant paths).

## Training games: 20,000 vs 40,000 (#61)

Same protocol as #56 (5 training seeds, eval on seed ranges 2000, 3000, 4000, 5000, 6000 x 20 games, nn 1-ply average score), no code change: only `train_games` moved from 20,000 to 40,000. The linear lr decay is a function of the fraction trained, so the 40k runs also stretch the schedule to twice as many games.

| training seed | 20k | 40k | delta |
|---|---|---|---|
| 1 | 19,611 | 30,770 | +11,160 |
| 2 | 18,011 | 30,283 | +12,272 |
| 3 | 17,870 | 33,257 | +15,387 |
| 4 | 20,431 | 33,314 | +12,883 |
| 5 | 17,846 | 25,906 | +8,060 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **30,706 ± 1,352** | **+11,952 ± 1,195** (t = 10.0) |

- Reading: doubling the games is worth about 12,000 points of nn 1-ply score at this stage of the curve, far above the seed noise. The curve has not flattened at 40,000 games.
- **The committed weights were not replaced**: they were trained with 200,000 games (see "NN weights" above) and score 40,282 on these same five ranges (38,641 / 34,894 / 34,196 / 50,210 / 43,467), well above any 40k net. Comparing a 40k net against them is not a fair fight, which is the point of the table: the protocol in #33/#34/#56/#59 uses 20,000-game nets, so those variant comparisons are all made far down the curve.
- Cost: 5 runs x 40,000 games, 514-559 s each (serial). Re-evaluating the 20k and committed weights with the same binary is free.

## Mid-game mixed replay (#63)

#59 lost the uniform share entirely and dropped 1,625 points. #34's design ("50% uniform + 50% |TD error| weighted") came closest to base of all candidates so far, so the two were combined: keep the uniform share, and bias the other half towards the bucket that holds 52.6% of the error (4..7 empty cells, #32).

Mechanism: `Options.midmix` (CLI variant `midmix`). Per-move TD(0) update untouched; after each game `m = 32` extra updates, each one either uniform over the whole ring buffer (probability 0.5) or `(|δ| + 1e-3)^0.5` weighted **among entries with 4..7 empty cells** (falling back to a uniform whole-buffer draw while that bucket is empty), target recomputed from the current weights. Default off; default path byte-identical (`md5 0a8e6bf01e256adf2fc481317c5072c9` at 2000 games).

Protocol identical to #56/#59: 20000 games, 5 training seeds, eval on seed ranges 2000, 3000, 4000, 5000, 6000 (20 games each), nn 1-ply average score; base column from #61.

| training seed | base | midmix | delta |
|---|---|---|---|
| 1 | 19,611 | 20,775 | +1,164 |
| 2 | 18,011 | 21,868 | +3,857 |
| 3 | 17,870 | 21,637 | +3,767 |
| 4 | 20,431 | 14,644 | -5,787 |
| 5 | 17,846 | 17,883 | +37 |
| **mean ± stderr (n=5)** | **18,754 ± 534** | **19,362 ± 1,377** | **+608 ± 1,762** (t = 0.3) |

- Reading: 4 of 5 seeds are positive and the mean is the best of all mid-game variants tried, but the spread (14.6k-21.9k against base's 17.8k-20.4k) is three times the effect, so this is not a gain either. It is however the first variant that is not worse on average, which fits the #32/#34 story: keep the uniform share and mix in the mid-game bucket rather than replacing it.
- Verdict: stays opt-in and off by default; committed weights unchanged.
- Cost: 5 runs x 20,000 games, 194-353 s each.

## Model update: fine-tuning the committed weights (#66)

`train_games > 0` always started from a random init, so the only way to improve the committed net was 200,000 games from scratch (#62). New CLI variant literal `resume` (positional 6) loads `weights_path` first and then trains `train_games` more games on the plain per-move path: `zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.001 - resume 30 /tmp/x.bin 1`. The default path is unchanged (recorded-checksum test, `md5 0a8e6bf01e256adf2fc481317c5072c9`).

Protocol: start from `ai/solver/nn_weights.bin` (200,000 games, training seed 1), fine-tune, evaluate on the #56 in-protocol ranges (2000..6000) and on held-out ranges (7000..10000) that are never used to choose a candidate. All scores are nn 1-ply average score, 20 games per range unless noted.

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

- Chosen: **+20k lr 1e-3 then +20k lr 3e-4** (40,000 extra games, 1,028 s + 628 s serial). It wins in-protocol and held-out, and the pattern "high lr first, low lr to settle" beats the same budget at a single lr.
- Final check on fresh ranges **11000..15000 with 40 games each** (never used before): committed 38,402, candidate 42,609 (+4,207, about +11%), candidate ahead on 4 of 5 ranges and tied on the fifth.
- Canonical bench `zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`: nn 64,112.9 -> **65,409.9** (+1,297, best tile 4096 -> 8192); every other policy unchanged, so the delta is the weights alone. `src/web/ranking.json` regenerated.
- New committed weights: `ai/solver/nn_weights.bin`, `md5 e501bae176f5cf2ff11a25c6f99e75a0` (old `6c9a164b5e05519ca2298c26e30431c4`).
- Caveat: fine-tuning cannot pass the point the 200k run already reached, so the gain is bounded. The single-lr and longer schedules were only tried at training seed 1 (seed 2 fine-tunes were clearly worse), so a small seed effect cannot be excluded.

## 200,000 games from scratch across seeds (#62)

Same protocol, `train_games` 200,000, training seeds 1 and 2, compared against the committed net.

| training seed | in-protocol (2000..6000) | vs committed (40,282) |
|---|---|---|
| 1 | 40,282 | 0 (byte-identical weights, `md5 6c9a164b...`) |
| 2 | 38,116 | -2,166 |

- Seed 1 reproducing the committed file byte-for-byte confirms the committed weights are exactly `200000 games, lr 0.003, seed 1`, and that the pipeline is deterministic (this is the check #62 asked for).
- Seed 2 lands below it, so retraining from scratch across seeds does not improve the model. The win came from fine-tuning (#66) instead, at 40,000 games (about 1/5 of the cost).
- Cost: 2 x 200,000 games, 4,118-4,254 s each.

## Symmetry augmentation + TD(lambda): diverges (#72)

Two ways to get more out of each self-play game: reuse every position in all 8 dihedral orientations (rotation/reflection of the 4x4 board), and propagate TD errors backwards along the whole game with eligibility traces instead of one step.

Mechanism (opt-in, `nn.Options`): `symmetry` applies each per-move TD(0) update to all 8 dihedral transforms of the afterstate with the same target (deterministic, no RNG); `lambda` switches the update to TD(λ) with accumulating traces (`trace = λ * trace + grad V`, `net -= lr * (out - target) * trace`, λ = 0 reduces to today's TD(0)). CLI variants `sym` and `symlam` (symmetry + λ = 0.7). `base` and every earlier variant stay byte-identical; the recorded-checksum test is untouched.

Correctness: one `sym` step equals eight explicit `forward(transformBoard(k, board))` + `train` steps (tested to 1e-15), λ = 0 matches the plain TD(0) step, the 8 transforms are distinct and closed under composition, and `forward` on a transformed board equals `forward` with transformed weights. So the implementation does what it says; the outcome is what fails.

20,000 training games per seed, eval ranges 2000/3000/4000/5000/6000 x 20 games, committed `nn` 1-ply protocol:

| variant | lr | per-seed mean score | mean ± stderr | weights |
|---|---|---|---|---|
| base (reference, #56) | 0.003 | — | 18,754 ± 534 | fine |
| `sym` | 0.003 | 2,116 / 3,472 / 2,116 / 2,772 / 3,035 | 2,702 ± 264 | 2 of 5 nets contain NaN/Inf |
| `sym` | 0.000375 (lr/8) | 4,274 / 6,828 / 2,762 | 4,621 ± 1,187 | finite but collapsed (V ≈ -51 for every board) |
| `symlam` | 0.000375 | 2,116 / 4,958 / 2,116 | 3,064 ± 947 | 2 of 3 nets are entirely NaN |

- Reading: eight updates per visited position is not eight times the data, it is eight correlated steps against a target computed from the pre-step net, and TD(0) with function approximation cannot absorb that — at the normal lr the weights blow up to NaN within 20,000 games, at lr/8 they collapse to a near-constant value function (dead ReLUs), and λ = 0.7 makes it worse because traces multiply the same over-update. Dividing the lr keeps the total step size but not the optimisation dynamics: the game schedule (lr decay per game, freshest games last) is compressed 8x.
- Verdict: not adopted, both options stay off by default (no default-path change, checksum test unchanged). A version that could work would accumulate the 8 transformed gradients from one net snapshot and apply a single averaged step (true batch augmentation) instead of eight sequential steps, and clip the gradient norm; that is a different, larger change, not a knob.
- Cost: 2 settings x 5 seeds and 1 setting x 3 seeds x 20,000 games, ~150-800 s per run.
## n-tuple network runner (#70)

The classic 2048 SOTA value function, as a standalone runner: `ai/solver/ntuple.zig` holds flat f32 pattern tables (`[16^k]f32`) indexed by the same `levelOf` cell encoding as the MLP, `ai/solver/ntuple_main.zig` trains them by TD(0) on afterstates and evaluates them.

- Patterns (one table each, shared across the 8 dihedral transforms of the board): row 0, row 1, 2x2 top-left, 2x2 middle column, main diagonal (4-tuples), 2x3 top-left (6-tuple). The table index is the minimum over the 8 transforms, so a rotated or mirrored board gets the same weights (TDD test).
- Index cost: a compile-time `cell -> (table, shift)` slot table, so one 16-cell scan yields all 48 transform accumulators; no table rescan per board.
- Update: `table[p] += lr * (r + V(next) - V(after)) / n_tuples`, reward = score delta / 1000 (same scale as `nn`).
- File: 8-byte magic + version + endianness + pattern count + per-table lengths + f32 LE payload.
- CLI: `zig build ntuple -- <train_games> <eval_games> <eval_seed> <lr> <base|chance> <table_path> [train_seed]`; `train_games = 0` loads the table. Eval prints the same lines as `nn`, labelled `ntuple`. `chance` is spawn-lookahead self-play and is much slower than `base`.
- Not wired into `solver.Policy` / `ranking` / `server` yet (adoption is a follow-up PR).

Smoke (Zig 0.17, Debug, `zig build ntuple -- 200 20 2000 0.01 base /tmp/t.bin 1`): 200 games, 37,341 moves, 0.2 s training. Eval seeds 2000..2019: `ntuple 1-ply` 2,439, `ntuple + 1 spawn lookahead` 8,710, expectimax depth 2 21,840. 2,000 games (same command, 10 eval games) gives 3,118 / 10,213 on seeds 2000..2009, so the tables do learn.

Table file: 68,419,628 bytes raw, 106,259 bytes with `gzip -9` after 200 games (182,822 bytes after 2,000). The dense 6-tuple table dominates the raw size; it is almost all zeros early, so the gzip target holds for the smoke file but grows as training fills it. If that breaks the < 1 MB static-hosting target later, drop the 6-tuple until a quantized or sparse payload is in place.

### Scaling the n-tuple tables (training run, #70)

Serial training runs with the initial lr 0.01 (linear decay to 10%), training seed 1, table size 68.4 MB raw (5 four-tuples + one 2x3 six-tuple):

| train games | moves | time | 1-ply | + 1 spawn lookahead |
|---|---|---|---|---|
| 200 | 37,341 | 0.2 s | 2,439 | 8,710 |
| 20,000 | 6,218,061 | 3.7 s | 4,426 | 11,385 |
| 200,000 | 86,039,320 | 51.0 s | 7,783 | 15,100 |
| 1,000,000 | 492,306,926 | 288.0 s | 8,832 | 17,093 |

- Reading: still climbing steeply at 1M games and about 60 games/s, but the level is far below the MLP (`nn` 72,205 with 2-ply). The classic n-tuple result needs tens of millions of games plus a much lower per-sample lr; at this scale the MLP is simply better per unit time.
- Table size: 68.4 MB raw, 106 KB gzipped at 200 games and 183 KB at 2,000 games, so a sparse/gzip payload is fine for static hosting as long as the six-tuple stays (it dominates the raw size).
- Verdict: not adopted as a policy yet. Follow-ups tracked in #73: 2-ply search on the n-tuple leaf, a `light` table set (5 four-tuples, 1/17 the raw size) and training resume so long sweeps are possible.

## Deeper search: nn2 (2-ply) (#69)

The `nn` policy is "1-ply + one spawn lookahead". Every extra ply is expensive (branching 4 moves x ~6 empty cells x 2 tile values), but the MLP's afterstate value is cheap enough to afford one more level, so the question was simply how much score one more ply buys.

Mechanism: `nn.Mode.deep` — my move -> exact spawn over all empty cells x {2, 4} (4 with p = 0.1) -> my move -> spawn -> leaf = afterstate value + score_delta / reward_scale. Deterministic (no RNG), with a bounded value cache, exposed as `solver.Policy.nn2` (ranking depth 2) and therefore selectable via `/api/solve?algo=nn2`. `nn` is unchanged and remains the default; the recorded-checksum test is untouched.

Comparison with the committed weights (200,000 + 40,000 games), eval ranges 2000, 3000, 4000, 5000, 6000 (20 games each), ReleaseFast:

| range | nn (1-ply + spawn) | nn2 (2-ply) | ms/move nn | ms/move nn2 |
|---|---|---|---|---|
| 2000 | 64,052 | 72,220 | 0.10 | 0.97 |
| 3000 | 64,231 | 73,277 | 0.10 | 1.07 |
| 4000 | 62,665 | 72,320 | 0.10 | 0.81 |
| 5000 | 63,276 | 73,410 | 0.04 | 0.47 |
| 6000 | 67,640 | 69,797 | 0.05 | 0.48 |
| **mean** | **64,373** | **72,205** | **0.08** | **0.76** |

- Reading: +7,832 (+12.2%) for about 10x the time per move. Absolute cost is still under 1 ms per move and ~1,200 leaves per move, so it is affordable in the browser; expectimax depth 2, for comparison, needs 0.04-0.08 ms/move and scores ~22-25k.
- Verdict: adopted as an extra policy (`nn2`); `nn` stays the default for speed, `nn2` is the quality option. `ranking.json` is regenerated separately once all pending policy changes land.
- Canonical bench `zig build ranking -Doptimize=ReleaseFast -- 5 20 1000` after the policy landed: nn2 **71,611.4** (best tile 4096) > nn 65,409.9 > expectimax_tt 41,216.4 > mcts 29,722.0 > expectimax 25,266.8 > ga 24,046.1 > beam 12,695.9 > greedy 3,423.8 > random 1,061.5; `src/web/ranking.json` regenerated with these numbers.
- Cost: 5 x 20 games, 63 s per range for nn2.

## Row-LUT board representation: no gain (#71)

Idea: pack the board into one `u64` (4 bits per cell, tile level 0..15, the same encoding `nn.zig` uses) and generate moves with 65,536-entry row tables for left/right plus a transpose for up/down, so a slide is four table lookups instead of a per-row scan. Implemented as `ai/solver/fastboard.zig` (`fromBoard`/`toBoard`, `slidePacked(board, dir) -> {board, score, changed}`, `emptyMask`, `setCell`) and wired into `solver.zig`'s expectimax (`maxValuePacked`/`chanceValuePacked`, with the old path kept as the test oracle).

Correctness: 1,000 pseudo-random boards x 4 directions reproduce `core.GameState.slide` exactly (board, score delta, `changed`), round-trip `toBoard(fromBoard(b)) == b`, and the searched values/best moves match the core-slide search. The scores in the table below are byte-identical, so the LUT is right — it just is not faster.

Measured with the committed weights/weights-independent expectimax, ReleaseFast, seed 2000:

| mode | moves/s (branch) | moves/s (main) | avg score | avg moves |
|---|---|---|---|---|
| expectimax depth 2, 20 games | 10,845 | 12,707 | 26,153.6 | 1,373.8 |
| expectimax depth 3, 10 games | 280 | 301 | 42,718.8 | 2,084.0 |

- Reading: 15% slower at depth 2 and 7% slower at depth 3, same scores. A slide is 4 LUT lookups plus a transpose for half the directions, while `core.slide` is already a tight 4-row scan, and every recursive leaf pays a `toBoard` conversion because the evaluation function still takes a `core.Board`. The move generation was never the bottleneck: chance-node recursion (6-8 spawns per node) and the position evaluation dominate.
- Verdict: not adopted, code dropped. It would only pay off with a packed-native evaluation function (no `toBoard` at leaves) and a packed chance-node spawn, i.e. rewriting the whole search around the representation — a separate, much larger change than a faster slider.
- Cost: 2 x 20 + 2 x 10 games, ~3 min build + 80 s measuring.

## n-tuple: deep search, light tables, resume (#73)

The n-tuple value function (`ai/solver/ntuple.zig`) stores one weight per (pattern, cell-state) pair and learns it with TD(0) self-play. Issue #70 left it at the runner stage with two limits: only 1-ply and 1-spawn lookahead, and a 68 MiB raw table that is dominated by one 2x3 6-tuple pattern.

Added:
- `Mode.deep` — my move -> exact spawn over empty cells x {2, 4} -> my move -> spawn -> leaf = afterstate value + score_delta / 1000 (same leaf rule as `nn`'s 2-ply `nn2`). Deterministic, leaf counts reported in the eval line (`leaves`, `leaves/move`, `ms/move`).
- `Set.light` — only the five 4-tuple patterns, 327,680 entries, **1.3 MiB raw** instead of 68 MiB, so the table could actually ship.
- `resume` / `lightresume` — load the table, then train more games from it (the load validates the file length, so a full-set file cannot be loaded into a light-set table: `error.TableSizeMismatch`).

Measured, 20 games per policy on eval seed 2000, ReleaseFast:

| training | 1-ply | + 1 spawn | depth 2 | ms/move (depth 2) |
|---|---|---|---|---|
| light, 2,000 games | 2,444 | 7,502 | 15,915 | 0.39 |
| light, 500,000 games | 7,134 | 17,987 | **25,760** | 0.24 |
| expectimax depth 2 (no table) | — | — | 21,840 | 0.02 |

- Reading: greedy self-play training is cheap (500k games in 124 s, 202M moves), but a 1-ply table stays weak (7,134). Almost all of the strength comes from search depth: the same table jumps to 17,987 with one spawn and to 25,760 with a second own move, i.e. depth-2 n-tuple already beats expectimax depth 2 while still running at 0.24 ms/move (~4.6k leaves/move). Search, not more table entries, is where the gain is.
- Against the MLP on the same protocol (eval seed 2000, 20 games): `nn` 64,052, `nn2` 72,220, depth-2 n-tuple 25,760 — roughly 2.5x below `nn`, so the n-tuple table is not a replacement for the learned MLP value function; it stays an experiment/runner (and a possible cheap browser-side fallback if the 1.3 MiB table ever matters more than strength).
- Verdict: adopted as the experiments' new best non-MLP option; `light` + `deep` is the default sweep configuration from here (1.3 MiB table, evaluable in seconds). Not wired into `algo` yet: `ranking.json` regeneration and a `Policy.nntuple` connection are separate follow-ups.
- Cost: 500k-game training 124 s + depth-2 eval 6.5 s per 20 games.

## Static WASM build (#68)

The web UI used to need the Zig server (`zig build run -- serve`): the board lived in server state and every policy call went through `/api/solve`. `zig build wasm -Doptimize=ReleaseFast` now produces a static site in `zig-out/web/` — `index.html` plus `game2048.wasm` — that plays and solves entirely in the browser, no backend, so it can be hosted on GitHub Pages or any static file server.

Mechanism: `src/wasm.zig` exposes a small C ABI (`init(seed_lo, seed_hi)`, `move(dir)`, `solve(algo, depth)`, `board_ptr()`, `score()`, `best_tile()`, `algo_count()`, `algo_name(i)`) over the same `core`/`solver` code the server uses; the nn weights are `@embedFile`-ed, so all nine policies (random, greedy, expectimax, mcts, beam, expectimax_tt, ga, nn, nn2) work offline. `src/web/index.static.html` is a separate static UI because it never talks to `/api/*`. `scripts/wasm-smoke.mjs` (node, no dependencies) builds a game, applies moves, and calls every policy, failing non-zero on a bad direction or an unchanged board.

Two fixes were needed to make the module run in a browser:
- `nn.ValueCache.get` indexed with `key & (capacity - 1)` where `usize` is 32-bit on wasm32; the slot is now explicitly `usize`.
- The 8,192-entry value cache (128 KiB) lives on the stack of `deepMove` and overflowed the wasm stack (`RuntimeError: memory access out of bounds` on the first `nn2` call). The capacity is now comptime-selected: 1,024 entries on wasm, 8,192 elsewhere, so native `nn2` performance is unchanged.

Measured: wasm module 745,167 bytes, smoke test passes for all nine policies and ten applied moves.

## Truncated-rollout MCTS with the NN leaf: nnmcts (#82)

`mcts` plays whole random games out to a terminal state and scores them with the raw game score, so it only knows what a random player accumulates. `nnmcts` keeps the same root UCT but stops each rollout after 8 random plies and scores the truncated board with the trained MLP: a simulation's reward is `score gained along the rollout / 1000 + net value of the final board`, where 1000 is the `reward_scale` the net was trained on, so a truncated leaf and a terminal position are comparable. A dead board adds 0 future value, which keeps terminal positions on that same scale.

Mechanism (`ai/solver/solver.zig`): `nnmctsMove` builds the four root arms from slide legality (null on a dead board) and per simulation runs `selectArm` — the UCT rule extracted verbatim from `mctsMove` — then adds one reward from `nnmctsRollout`. The rollout takes uniformly random legal slides and spawns after each ply except the last, so the leaf board is a post-slide, pre-spawn afterstate: the input shape the net was trained on. The first version evaluated the post-spawn board instead and scored 8,745 (200 sims x 8 plies) against 17,023 for the same budget once the leaf matched the training shape — the leaf's board shape matters more than the rollout length.

Wiring: `Policy.nnmcts` is appended after `nn2`, so `algo=nnmcts` over HTTP, `algo_count`/`algo_name` in the wasm ABI, the static wasm UI and `scripts/wasm-smoke.mjs` pick it up without further changes. `src/web/ranking.json` is deliberately not regenerated, so the server UI table still lists the nine adopted policies.

Measured (ReleaseFast, committed weights only, eval seed 1000, 10 games per row, serial play):

| sims x plies | avg score | ms/move |
|---|---|---|
| 200 x 2 | 17,085 | 0.27 |
| 200 x 8 | 17,023 | 0.91 |
| 200 x 16 | 13,717 | 1.19 |
| 500 x 16 | 20,054 | 2.82 |
| 1,000 x 2 | 17,115 | 1.65 |
| 1,000 x 4 | 23,646 | 3.16 |
| 1,000 x 8 (shipped) | 20,137 | 5.41 |
| 1,000 x 16 | 23,113 | 4.12 |
| 2,000 x 8 | 27,346 | 9.61 |
| 5,000 x 2 | 20,026 | 2.91 |

References on the same seed: `mcts` 29,722 (committed `ranking.json`, 5 x 20 games), `nn` 64,601 at 0.34 ms/move and its 1-ply greedy variant 38,616 at 0.008 ms/move (same seed, 10 games).

- Truncation plus the net leaf does not beat plain `mcts`: 2,000 simulations reach 27,346, i.e. `mcts`'s 29,722 at roughly 7x the per-move cost, and the shipped 1,000 x 8 point lands at 20,137 for 5.4 ms/move (~6.9 s/game against `mcts`'s ~1.3 ms/move).
- Random play drives the board off the distribution the net was trained on, so after 8 random plies the leaf value is noisy and the arm means collapse toward each other. The same leaf used the way it was trained — argmax over 1-ply afterstates plus one spawn sample — is worth 64,601 at 0.34 ms/move: 3x the score for 1/16 of the time per move. The net reads afterstates; random play before the leaf only destroys information.
- More simulations or deeper rollouts did not produce a consistent improvement at the tested budgets; even the best point (2,000 x 8) stayed below plain `mcts` and `nn` while costing more.
- Verdict: not adopted for `ranking.json` or as the default policy; kept as an experimental policy (`Policy.nnmcts`, `algo=nnmcts`, wasm) because it documents the limit of pairing random rollouts with this value net.
- The canonical 5 x 20 same-seed `zig build ranking` run for `nnmcts` was started but skipped — it did not finish within the working timeout — so the decision rests on the 10-game table above, which carries a few thousand points of noise per row. Existing `mcts` and `nn`/`nn2` behaviour is unchanged: the UCT and best-arm rules were extracted verbatim with the same RNG call order, and both NN policies still call the same `nn.bestMove`.
- Wasm: three separate 148 KB per-move net copies (one per NN arm) overflowed the wasm stack — `solve` failed with `RuntimeError: memory access out of bounds` even for `algo=0`, since the copies live in `choose`'s frame. The three NN arms now share one copy; the module grew from 745,167 to 753,911 bytes (286,245 gzipped) and the smoke test passes for all ten policies and ten applied moves.
## NN-value beam with sampled spawns (#83)

Idea: `beam` uses the hand-written snake/merge evaluation and assumes a 2 on every empty cell; `nn`/`nn2` use the learned MLP but only walk one or two own moves. `nnbeam` puts the learned value under a small beam: fixed width 8, horizon 3 own moves, and 2 sampled spawns between plies instead of an exact spawn expectation.

Mechanism (`ai/solver/solver.zig`):
- `nnBeamSearch(net, state, rng, horizon)` keeps two `[8]NnBeamNode` arrays (`state`, `first_move`, `reward`, `value`) and reuses the existing `insertBeam`-style insertion. Rewards are `score_delta / nn.reward_scale`; leaves are `nn.forward` on the afterstate, i.e. the same leaf rule as `nn`. `nnBeamBest` is the `nnbeam_horizon` wrapper the policy calls.
- Spawns are sampled only *between* plies: `sampleSpawn` picks a uniformly random empty cell and places 2 (90%) or 4 (10%). The last ply inserts the afterstate itself, without a spawn, matching `nn2`, which ends on `chance(..., 0)` — the net was trained on afterstates before spawn, so a post-spawn leaf asks it about a board it never saw. A full board keeps the afterstate as the leaf; no legal root move returns `null`, so the dead-board semantics match the other policies.
- `first_move` is copied from the ply-0 direction, so the returned move is always legal on the root board (a test asserts `canSlide` and that the deeper direction never leaks out).
- Randomness comes only from the caller's `std.Random`, so the same seed replays the same move; a test plays 12 random-move boards and compares two identically seeded calls.
- `depth` is ignored (the horizon is fixed); `Policy.nnbeam` is registered in `choose`, in `ranking`'s policy list/description, and therefore in the server, the wasm `algo_count`/`algo_name` export, and the static UI dropdown.

WASM: routing `nnbeam` through its own `nn.parse` arm pushed `choose`'s stack frame over the wasm stack — `random` failed too, because a function's frame is allocated on entry. The three NN policies now share one `nn.parse` per call (behavior unchanged: same net, one policy executes), and all ten policies pass the smoke test.

Measured, ReleaseFast, one run per policy, same seed and policy order (`zig build ranking -Doptimize=ReleaseFast -- 1 20 2000`, 3m47s):

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

- Reading: `nnbeam` scores exactly 2.03x `beam` (25,480 vs 12,585) at 0.124 ms/move, i.e. still cheap, so the sampled beam plus the learned leaf does lift the shallow `beam` baseline. It stays slightly below `expectimax` depth 2 (26,154) and far below `nn` (63,486): three plies of width 8 with 2 spawns each evaluate ~64 leaves per move, but every leaf is an afterstate the beam reached through one sampled spawn per gap, so the search commits to one spawn outcome instead of averaging it. That is the structural reason a sampled beam trails the 1-ply-plus-expectation `nn`.
- Verdict: kept as an experimental policy (`algo=nnbeam`), not a default; `nn` stays the fast option and `nn2` the strong one. Either widening the beam or averaging the spawns would recover strength, but at `nn2`-class cost, so it stays a bounded experiment.
- Correction: the first version of this section reported seed-1000 10-game numbers (23,875 / 28,493 / 16,513) from an implementation whose *last* ply also inserted a post-spawn child, so it asked the net about spawn boards at the leaf. Those numbers describe a different leaf rule and were dropped rather than mixed with the table above; the code now ends the last ply on the afterstate, and the focused test (`nnbeam evaluates the afterstate before spawn at the final ply`) pins the rule at horizon 1: the returned leaf board must equal the afterstate of the returned move.
- `src/web/ranking.json` is not regenerated here (the canonical 5 x 20 run costs ~5 min and the policy is experimental); the file still lists the nine earlier policies.
## n-step TD: worse than TD(0) at n = 4 (#84)

TD(0) trains each afterstate from the next afterstate's value, i.e. from a value estimate one move old. A finite n-step return (`reward_t + ... + reward_{t+n-1} + V(s_{t+n})`) propagates a terminal reward back n moves per update and is still fully online, unlike the failed backward sweep (#33) which re-swept the whole game after it ended.

Mechanism (opt-in, `nn.Options.nstep`): 0 = off, 1 = TD(0), n > 1 = n-step. `NStepWindow` holds the pending steps; each new step's `V` closes the target of the step leaving the window (`sum of its n rewards + V`), and at game over `nextFinish` flushes the last min(n, game length) steps with the rewards alone (`V(terminal) = 0`). The moves come from the same 1-ply policy and the rng is used only by `state.move`, so the trajectory matches TD(0) exactly while the updates are identical (`n = 1` is byte-identical to TD(0), tested). CLI variant `nstep` (n = 4), standalone: other options are ignored and the weight file format is unchanged. Default weights and every existing variant stay byte-identical (recorded-checksum tests untouched).

Correctness: the window closes a step with exactly its n rewards plus the closing value and flushes truncated returns in order (unit tests); `nstep = 1` reproduces TD(0) scores and weights byte-for-byte; `nstep = 4` is deterministic and finite; `nstep = 0` equals the default path. So the implementation does what it says; the outcome is what fails.

20,000 training games, training seed 1, eval ranges 2000/3000/4000/5000/6000 x 20 games (100 games, same seeds for both nets), ReleaseFast, committed `nn` eval protocol:

| variant | lr | train moves | train time | 1-ply | + 1 spawn | 2-ply (deep) |
|---|---|---|---|---|---|---|
| base TD(0) | 0.003 | 13,346,693 | 145.0 s | 19,175 ± 895 | 30,918 ± 816 | 43,048 ± 1,549 |
| `nstep` (n = 4) | 0.003 | 10,150,588 | 100.6 s | 13,055 ± 641 | 23,755 ± 296 | 29,886 ± 679 |
| `nstep` (n = 4) | 0.00075 (lr/4) | 8,335,318 | 89.1 s | 9,019 ± 231 | 15,253 ± 415 | 20,351 ± 584 |

(± is the stderr over the 5 eval ranges; expectimax depth 2 on the same seeds is 25,004 ± 1,108, same for every run.)

- Reading: at the normal lr n = 4 loses 6,120 points on 1-ply (−32%) and 13,162 on 2-ply (−31%), far outside the stderr. It is not a step-size problem: lr/4 is worse again (−53% on 1-ply, −42% on 2-ply), so the n-step target itself is the problem, not the size of the update.
- The same game budget also yields fewer training moves (10.15M vs 13.35M, −24%): the policy being learned plays shorter games, so n-step training compounds its own regression instead of seeing more data. The faster wall time is that, not a gain.
- Why: the update for a state is applied n−1 moves after it was visited, against a net that has already moved on, so the bootstrapped part of the return is stale and the target's scale is n rewards instead of one. TD(0)'s one-move-old target is exactly what this net, lr schedule and `reward_scale` were tuned for. Same failure mode as the backward sweep (#33) and TD(λ) (#72): value-side tricks that lag behind, or multiply, the online update.
- Verdict: not adopted. `nstep` stays an opt-in experiment variant, `nstep = 0` is the default, no default or committed weights change. A version that could work would accumulate the n-step targets over a batch and apply one averaged step against a frozen snapshot, and normalise the target scale — a different, larger change, not a knob.
- Limits: one training seed and one budget (20,000 games) at n = 4 only; lr/4 was the only extra diagnostic. The drop is far larger than the eval stderr, so more seeds would only sharpen the size of the loss, not its sign.
- Cost: 3 x 20,000 games (145 / 101 / 89 s training) + 4 policies x 5 ranges x 20 eval games per run (2m51s-4m52s wall each, dominated by the 2-ply eval).
## Bench run 2026-10-27 00:09:04 +0900

- command: `/usr/bin/time -p zig build ranking -Doptimize=ReleaseFast -- 1 2 1000`
- commit: 83af5aa
- date: 2026-10-27 00:09:04 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":2,"entries":[{"policy":"nn2","depth":2,"games":2,"avg_score":67424.0,"avg_moves":3091.0,"best_tile":4096,"latency_ms_per_move":1.911,"games_per_sec":0.17,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"nn","depth":0,"games":2,"avg_score":66624.0,"avg_moves":2992.5,"best_tile":4096,"latency_ms_per_move":0.197,"games_per_sec":1.69,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"expectimax","depth":2,"games":2,"avg_score":60220.0,"avg_moves":2804.0,"best_tile":4096,"latency_ms_per_move":0.080,"games_per_sec":4.46,"description":"2手先まで探索し、ランダムな出現（2か4）を平均して評価する。評価は蛇行順と合体ペアの手作り関数"},{"policy":"ga","depth":2,"games":2,"avg_score":46584.0,"avg_moves":2287.0,"best_tile":4096,"latency_ms_per_move":0.068,"games_per_sec":6.43,"description":"GAで評価の重みを進化させた期待値探索（depth 2）。GA内のholdout(seed 500〜)では+4%だが、ranking seed 1000では素の expectimax より低い"},{"policy":"expectimax_tt","depth":0,"games":2,"avg_score":37282.0,"avg_moves":1825.0,"best_tile":4096,"latency_ms_per_move":0.686,"games_per_sec":0.80,"description":"depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）"},{"policy":"mcts","depth":0,"games":2,"avg_score":29732.0,"avg_moves":1533.0,"best_tile":2048,"latency_ms_per_move":1.018,"games_per_sec":0.64,"description":"UCTで候補手を選び、ランダムな対局を繰り返して評価する"},{"policy":"beam","depth":0,"games":2,"avg_score":17038.0,"avg_moves":943.5,"best_tile":2048,"latency_ms_per_move":0.061,"games_per_sec":17.45,"description":"幅32で3手先まで探索し、各空きマスに2が出現すると仮定して評価する"},{"policy":"greedy","depth":0,"games":2,"avg_score":3020.0,"avg_moves":251.0,"best_tile":256,"latency_ms_per_move":0.000,"games_per_sec":9428.40,"description":"1手先だけ見る。滑らせた盤面の評価が最も高い手を選ぶ（出現や探索はしない）"},{"policy":"random","depth":0,"games":2,"avg_score":1368.0,"avg_moves":140.0,"best_tile":128,"latency_ms_per_move":0.000,"games_per_sec":30207.83,"description":"合法手からランダムに1手選ぶ（比較の基準）"}]}
real 19.65
user 19.54
sys 0.16
```

## Bench run 2026-10-27 00:09:36 +0900

- command: `/usr/bin/time -p zig build ranking -Doptimize=ReleaseFast -- 1 2 1000 nn2`
- commit: 83af5aa
- date: 2026-10-27 00:09:36 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":2,"entries":[{"policy":"nn2","depth":2,"games":2,"avg_score":67424.0,"avg_moves":3091.0,"best_tile":4096,"latency_ms_per_move":1.911,"games_per_sec":0.17,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"}]}
real 11.97
user 11.89
sys 0.13
```

## NN weights parsed once per thread, not once per move (#87)

`choose` re-parsed the embedded ~148 KB weight block into a `Net` value on every move of every NN policy (one shared copy per move since the wasm-stack fix). Now a `threadlocal` cache parses the weights once per thread and every move passes a `*const Net` by reference, so parallel ranking / HTTP / WASM each keep one copy per thread with no shared mutable state. The `parse_count` seam in `nn.zig` pins the behaviour in a test (a fixed-seed nn/nn2 game parses exactly once), and the golden tests pin the move sequences.

Correctness: same seed 1000 / 5 games give identical scores and move counts before and after (nn2 71,880.0 / 3,267.2, nn 69,640.8 / 3,151.2, nnbeam 21,096.8 / 1,143.8, nnmcts 12,864.8 / 763.4); `zig build test` and the wasm smoke (11 policies, 10 applied moves) pass on the rebased branch.

Timing (`ranking`, ReleaseFast, `latency_ms_per_move` from the #89 instrument):

| policy | before (b389ba5) | after | delta |
|---|---|---|---|
| nn2 | 1.8864 | 1.8658 | -1.1% |
| nn | 0.1995 | 0.1955 | -2.0% |
| nnbeam | 0.1233 | 0.1197 | -2.9% |
| nnmcts | 2.5574 | 2.5930 | +1.4% (noise) |

(seed 1000, 5 games.) A second pair at seed 2000 / 10 games reproduces the direction: nn 0.1961 -> 0.1947 (-0.7%), nnbeam 0.1210 -> 0.1190 (-1.7%). Same-code repeat runs span nn 0.1902-0.1947 and nnbeam 0.1175-0.1190, so every before run sits above every after run for nn and nnbeam: the 1-3% gain is small but outside the run-to-run noise band. nn2's -1.1% is inside the band and nnmcts is flat.

- Reading: the removed work is one 148 KB `Net` copy per move; against nn/nnbeam's 0.1-0.2 ms/move that is the visible 1-3%. For nn2 the 2-ply search dominates and hides it. The copy also lived in `choose`'s frame, the last wasm-stack footprint of the NN arms (per-arm copies used to overflow the wasm stack).
- Verdict: adopted. Behaviour is byte-identical, the gain is small but consistent for the fast policies, and the change is the enabling step for reusing the parsed net inside a search.
- Limits: one machine; one A/B pair per seed plus the repeats above; `latency_ms_per_move` is the serial per-move cost from the #89 instrument. `src/web/ranking.json` untouched.

## Bench run 2026-10-27 01:26:14 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000 nn,nn2,nnmcts,nnbeam`
- commit: b389ba5
- date: 2026-10-27 01:26:14 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.8864,"games_per_sec":0.16,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"nn","depth":0,"games":5,"avg_score":69640.8,"avg_moves":3151.2,"best_tile":4096,"latency_ms_per_move":0.1995,"games_per_sec":1.59,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"nnbeam","depth":0,"games":5,"avg_score":21096.8,"avg_moves":1143.8,"best_tile":2048,"latency_ms_per_move":0.1233,"games_per_sec":7.09,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5574,"games_per_sec":0.51,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"}]}
```

## Bench run 2026-10-27 01:25:05 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000 nn,nn2,nnmcts,nnbeam`
- commit: 1380349
- date: 2026-10-27 01:25:05 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.8658,"games_per_sec":0.16,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"nn","depth":0,"games":5,"avg_score":69640.8,"avg_moves":3151.2,"best_tile":4096,"latency_ms_per_move":0.1955,"games_per_sec":1.62,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"nnbeam","depth":0,"games":5,"avg_score":21096.8,"avg_moves":1143.8,"best_tile":2048,"latency_ms_per_move":0.1197,"games_per_sec":7.31,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5930,"games_per_sec":0.51,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"}]}
```

## Bench run 2026-10-27 01:28:19 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 10 2000 nn,nnbeam`
- commit: b389ba5
- date: 2026-10-27 01:28:19 +0900
- exit: 0

```text
{"seed":2000,"runs":1,"games_per_run":10,"entries":[{"policy":"nn","depth":0,"games":10,"avg_score":66471.2,"avg_moves":3016.5,"best_tile":4096,"latency_ms_per_move":0.1961,"games_per_sec":1.69,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"nnbeam","depth":0,"games":10,"avg_score":25959.2,"avg_moves":1337.3,"best_tile":4096,"latency_ms_per_move":0.1210,"games_per_sec":6.18,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"}]}
```

## Bench run 2026-10-27 01:28:27 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 10 2000 nn,nnbeam`
- commit: 1380349
- date: 2026-10-27 01:28:27 +0900
- exit: 0

```text
{"seed":2000,"runs":1,"games_per_run":10,"entries":[{"policy":"nn","depth":0,"games":10,"avg_score":66471.2,"avg_moves":3016.5,"best_tile":4096,"latency_ms_per_move":0.1947,"games_per_sec":1.70,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"nnbeam","depth":0,"games":10,"avg_score":25959.2,"avg_moves":1337.3,"best_tile":4096,"latency_ms_per_move":0.1190,"games_per_sec":6.28,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"}]}
```

## Search workspaces reused per thread and per game (#88)

- What changed: `nn2`'s `ValueCache` (built fresh inside every `Search`, i.e. every move) and `expectimax_tt`'s 8192-entry TT (allocated and cleared on every `bestMoveTT` call) are now reused workspaces. `nn.sharedValueCache()` and `ttWorkspace()` hand out one threadlocal workspace each (lazy-cleared on first use), `nn.bestMoveWithCache` lets a caller own the cache, `nn.playWithStats` keeps one cache per game, and `nnmcts` reuses `sharedValueCache()` across rollouts and moves. Entries validate the full board plus depth and kind (`ValueCache` keys on the 64-bit board hash and drops on mismatch), so a warm workspace changes speed only and can never return another position's value.
- Correctness: same seed 1000 / 5 games give identical scores and move counts before and after (nn2 71,880.0 / 3,267.2, expectimax_tt 51,376.0 / 2,426.2, nnmcts 12,864.8 / 763.4). Four new tests cover fresh-vs-warm move identity over 30 moves (nn) and 2x40 moves across two games (TT, also asserting one shared workspace), plus slot-collision behaviour for both caches (a colliding entry is dropped, never returned as a wrong value). `zig build test` and the wasm smoke (755,150 bytes wasm, 11 policies, 10 applied moves) pass.
- Timing (`ranking`, ReleaseFast, `latency_ms_per_move`, seed 1000 / 5 games; before = 3 runs on 8986c30, after = 2 runs on 3ac831d):

| policy | before | after | delta |
|---|---|---|---|
| nn2 | 1.8399-1.8668 | 1.7395-1.7427 | -5.5% |
| expectimax_tt | 0.6965-0.7058 | 0.6613-0.6620 | -5.0% |
| nnmcts | 2.5701-2.5920 | 2.5348-2.5362 | -1.4% |

Every after run sits below every before run: the bands do not overlap for any of the three policies.
- Cache statistics (temporary counters, seed 1000, 200 moves per case): `nn2` with the shared cache hit 88.2% (2,418,915 hits / 2,743,720 lookups) and evaluated 1,624.0 leaves/move vs 1,745.6 with a fresh cache per move (-7.0% leaf evaluations: 324,805 vs 349,113). `expectimax_tt` with the shared TT made 1,261,389 probes (6,306.9/move) vs 1,313,235 with a fresh table per move (-3.9% node expansions); hit rate 55.7% vs 54.5%.
- Memory: the same arrays moved from per-call stack to per-thread TLS — TT 8192 x 32 B = 256 KiB per thread, `nn2`'s `ValueCache` 2^13 x 16 B = 128 KiB per thread (2^10 x 16 B = 16 KiB on wasm). What disappeared is per-move init traffic: the 128 KiB `@splat` reset plus the 8,192-entry `valid = false` clear loop now run once per thread instead of once per move.
- Reading: nnmcts only shares `ValueCache` (it has no TT) and gains the least; expectimax_tt gains from both the removed clear and cross-move hits (a new root is inside the previous search's tree); nn2's gain is mostly the removed per-move init since the leaf saving is only 7%.
- Verdict: adopted. No behaviour change, all three policies improve, every after run below every before run.
- Limits: one machine; hit-rate figures come from temporary counters, not a permanent instrument; `latency_ms_per_move` is the serial per-move cost from the #89 instrument. `src/web/ranking.json` untouched.

## Bench run 2026-10-27 01:50:00 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000 nn2,nnmcts,expectimax_tt`
- commit: 8986c30
- date: 2026-10-27 01:50:00 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.8399,"games_per_sec":0.17,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"expectimax_tt","depth":0,"games":5,"avg_score":51376.0,"avg_moves":2426.2,"best_tile":4096,"latency_ms_per_move":0.6965,"games_per_sec":0.59,"description":"depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5701,"games_per_sec":0.51,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"}]}
```

## Bench run 2026-10-27 02:03:28 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000 nn2,nnmcts,expectimax_tt`
- commit: 3ac831d
- date: 2026-10-27 02:03:28 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.7395,"games_per_sec":0.18,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"expectimax_tt","depth":0,"games":5,"avg_score":51376.0,"avg_moves":2426.2,"best_tile":4096,"latency_ms_per_move":0.6620,"games_per_sec":0.62,"description":"depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5348,"games_per_sec":0.52,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"}]}
```
