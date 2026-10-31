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
`diag` works with `par` (per-game diagnostics are merged into one table). #99 extends batching to the
stateless variants (`par` in arg 10): see `par for stateless variants (#99)` at the end of this file.

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

## Fine-tune extra rounds: overfit past the #66 peak (#97)

The 20k to 40k learning curve was still climbing (#66), so the question was whether more `resume` rounds on top of the current committed weights (200k + the #66 40k) keep paying. Same protocol as #66: `resume` from `ai/solver/nn_weights.bin`, training seed 1, nn 1-ply, in-protocol ranges 2000..6000 and held-out ranges 7000..10000, 20 games per range.

| candidate (from committed) | in-protocol | held-out | vs committed (held-out) |
|---|---|---|---|
| committed (200k + 40k, seed 1) | 43,187 | 45,342 | — |
| +20k, lr 3e-4 (a2) | 43,083 | 41,936 | -3,406 |
| +20k lr 1e-3 then +20k lr 3e-4 (a1r2) | 42,974 | 41,290 | -4,052 |

- The held-out column is the decider (the #66 adoption rule): both candidates are flat in-protocol (spread ~200 against per-range noise of several thousand) but lose 3.4k-4.1k on the ranges that never select a candidate. The exact recipe that won +8,109 when applied once loses -4,052 when applied again — fine-tuning has passed its peak and is now fitting the training distribution.
- Verdict: not adopted, `ai/solver/nn_weights.bin` unchanged (md5 `e501bae176f5cf2ff11a25c6f99e75a0`). More rounds on this recipe are spent; further gains need new training signal (search-based self-play, or the batched-update line in #98/#99), not more of the same.
- Cost: 3 x 20,000 games serial (`resume`), 460 s / 319 s + 345 s / 460 s, plus 3 nets x 9 ranges x 20 eval games.

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

## Batched symmetry update: symbatch (#98)

Follow-up of #72: reuse each position in all 8 dihedral orientations without the eight correlated steps that made `sym` diverge.

Mechanism (opt-in `nn.Options.sym_batch`, CLI variant `symbatch`): for one afterstate and its target `t` (computed exactly as the `sym` path does: its scaled reward plus V of the next afterstate, 0 after game over) it runs `forward` + `gradOf` on the 8 transformed boards **while `net` is untouched**, so all eight gradients describe the same pre-step net, averages them elementwise (`mean_k (V_k - t) * grad V_k`), clips the averaged gradient's global L2 norm to `nn.sym_batch_clip = 1.0`, and applies one SGD step `net -= lr * clip(mean_k ...)`. Deterministic, no RNG. `base` and every earlier variant stay byte-identical; the recorded-checksum tests are untouched.

Why the clip value: measured averaged-gradient norms over one self-play game are 0.04..0.26 on a freshly initialised net but 4..300 late in training on the committed net (TD errors grow with the value scale, ~36..40 there). A unit-norm clip is therefore a no-op at the start (where lr is largest) and only binds late, where it caps one update's total weight displacement at `lr = 0.003` against a committed net mean |w| of 0.44 — the blow-up path of #72 (weights moving by their own magnitude per move, 8 times per position) is cut off instead of pre-empted by a smaller lr.

Correctness: one `symbatch` step equals the averaged gradient applied once (tested to 1e-15), differs from the eight sequential `sym` steps, and on a large TD error (t = 100) the applied step's norm is exactly `lr * sym_batch_clip` while the unclipped averaged gradient is more than 10x the clip, so the clipping branch is exercised.

Measurement (#72 protocol: 20,000 training games, training seed 1, eval ranges 2000/3000/4000/5000/6000 x 20 games, `nn` 1-ply):

| variant | lr | mean over ranges | per-range scores | training moves (games) |
|---|---|---|---|---|
| base (reference, #56) | 0.003 | 18,754 ± 534 (5 seeds) | — | — |
| `sym` (reference, #72) | 0.003 | 2,702 ± 264 | — | — |
| `sym` (reference, #72) | 0.000375 | 4,621 ± 1,187 | — | — |
| `symbatch` | 0.003 | 1,314 | 1,292 / 1,337 / 1,404 / 1,309 / 1,227 | 2,568,519 (~128/game) |
| `symbatch` | 0.000375 (lr/8) | 794 | 732 / 883 / 766 / 788 / 803 | 1,858,935 (~93/game) |

- Reading: the averaged-snapshot step does not diverge (no NaN/Inf, unlike `sym`), but it collapses anyway — both nets play 90-130-move games that die at tile 128-256, below even the collapsed `sym` rows and an order of magnitude below `base`. The likely mechanism is the clip: the measured averaged-gradient norms reach 4-300 late in training, so `sym_batch_clip = 1.0` caps those updates at 1/4-1/300 of their natural size exactly when the value function would mature, and lr/8 shrinks the early updates too — so the lr/8 row is worse, not better. The #72 failure mode (eight correlated over-updates) is gone, but TD(0) with a bootstrapped target gets no benefit from the extra transforms and the throttled step cannot recover.
- Verdict: not adopted; `symbatch` stays an opt-in experiment variant, `sym_batch = false` is the default, no default-path change (recorded-checksum tests untouched).
- Next candidates if this line is retried: a clip sweep (`sym_batch_clip` at 10 or 100, or no clip), and/or a target averaged over the same 8 transforms so the regression is self-consistent.
- Cost: 2 x 20,000 games (435 s + 335 s training), plus 5 ranges x 20 eval games per net.
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

## Order penalty follows the game phase, looser after 4096 (#92)

Issue #92 proposed tuning the snake-order penalty by game phase: keep the mid-game relaxed
(empties and merge pairs first) and tighten from the 4096 tile on, so the endgame builds the
merge chain that reaches 8192. The mechanism shipped: `Weights.order_penalty_late`, used by
`evaluateW` once the board's largest tile reaches 4096 (level >= 12). Default equals
`order_penalty`, so pre-#92 behavior is unchanged unless a weight sets it; `ga_weights` keeps
one gene for both phases.

The measurement inverted the hypothesis. expectimax_tt depth 3, 50 games per cell, seeds
1000-1049 (`TEMP measure order phase`, average score / reach 4096 / reach 8192 / dead below
2048):

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

Tightening after 4096 lost to the constant in every variant; loosening won, best at 25.
Second seed set (2000-2049, 50 games, same cells) confirms: 43.174/25 avg 38,383.7 (>=4096
16) vs A 38,025.2 (>=4096 13). Across both sets 43.174/25 averages 40,307 vs A 38,005 (+6.1%)
with 30 vs 22 runs reaching 4096. Same-seed determinism verified by repeating the cell
(50-game sets reproduce exactly).

The 5-game `ranking` snapshot on seed 1000 moved the other way for expectimax (41,107 -> 19,964)
and beam (14,369 -> 12,329). Re-measured at 50 games per cell those are small-sample noise:
expectimax depth 2 is 24,297.7 (C5) vs 24,403.5 (A) on seeds 1000-1049 and 25,385.7 vs
24,220.1 on 2000-2049; beam 12,346.7 vs 11,373.8; greedy 3,195.5 vs 3,109.8; mcts is
evaluation-independent (30,709.7 both). No evaluation-weight policy regresses at 50 games.

Adopted as `baseline = .{ .order_penalty = 43.174, .order_penalty_late = 25 }` (commit follows).
NN policies are unaffected (learned weights, not `Weights`).

Reading: the mid-game wants the snake order enforced (43.174 > 30, same as the old GA weights),
but after the 4096 anchor is secured the penalty should relax, not tighten — holding the chain
rigidly costs the merges that grow 4096 into 8192. The 8192 tail (2 runs vs 1) is suggestive,
not conclusive at 100 games.

Verdict: adopted (expectimax_tt +6.1% avg, more 4096+ runs, no measured regression).

Limits: one machine, 100 games per config; the 5-game ranking table in the run blocks below is
too small to compare weight sets (expectimax flipped 41,107 -> 19,964 there and back at 50
games); `ga_weights` keeps the single `order_penalty` gene for both phases, so GA evolution is
unchanged; an intermediate sweep between 25 and 43.174 was not run.


## par for stateless variants (#99)

`trainBatch` used to hard-code `trainGame`, so only the plain per-move variant could batch. It now
delegates to `trainBatchOpts(.., opts)` (`ai/solver/nn.zig`), which runs each game with
`trainGameOpts(.., opts)`: same `net_start + (1/N) * sum_i (net_after_i - net_start)` aggregation,
now over variant-trained per-game nets. `trainBatch` stays as the default-options wrapper, so the
plain path and its recorded checksum are unchanged.

- CLI: arg 6 `par` keeps its meaning (base + parallel). The new positional arg 10 `par` batches the
  variant named in arg 6: `zig build nn -Doptimize=ReleaseFast -- 20000 20 2000 0.003 - sym 30 /tmp/nnb/sym.bin 1 par`.
  Arg 10 was an unused trailing slot (any 10th positional forced the usage error), so every existing
  command line keeps its exact meaning. The training log prints the composed label (`variant=sym+par`).
- Works: `base`, `backward`, `replay`, `sym`, `symlam`, `nstep`, and `resume` (the batch starts from
  the net `startNet` loaded out of `weights_path`).
- Not supported: `priority`, `midgame`, `midmix`. Their 1024-entry ring buffer is cross-game state
  shared by pointer and the batch mean would change their semantics, so the CLI prints the usage text
  and refuses; `trainBatchOpts` asserts the buffer is null.
- Reproducibility: `parallel training batch with variant opts matches a serial run of the same game
  seeds` (`ai/solver/nn.zig`) runs 4-game batches with `symmetry` and with `replay_k = 4` on threads
  and compares the weights byte-for-byte with the serial reconstruction over seeds 77..80. Two CLI
  runs each of `sym` + `par` and `replay` + `par` wrote identical weight files. `zig build test` passes.
- Measured speedup (8 logical CPUs, batches of 8 games): `sym` 2,000 games 16.5 s serial -> 2.7 s
  with `par` (**6.1x**), and the #66 `resume` fine-tune +20,000 games lr 3e-4 460 s serial -> 90.9 s
  with `par` (**5.1x**). The speedup of #35's `base` (5.7x) carries over to the heavier variants.
- Measured score (does batched aggregation change the outcome?): `resume` +20,000 games lr 3e-4,
  training seed 1, the exact recipe of the #97 serial run `a2`, run as `resume` + `par`:
  in-protocol (2000..6000, 20 games/range) 42,736 vs serial 43,083, held-out (7000..10000) 43,388
  vs serial 41,936. The two update orders land within the eval noise of each other on both range
  sets, so batching trades nothing measurable for the ~5x speedup. (Both stay below the committed
  net's 45,342 held-out, which is the #97 overfit finding, not a batching effect.)


## Bench run 2026-10-27 02:43:24 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:43:24 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=37985.2 reach4096=9 reach8192=1 dead_below2048=10 max_tile_seen=8192
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:44:46 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:44:46 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=39981.5 reach4096=14 reach8192=0 dead_below2048=8 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:46:19 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:46:19 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=35069.0 reach4096=9 reach8192=0 dead_below2048=10 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:47:30 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:47:30 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=39221.7 reach4096=14 reach8192=0 dead_below2048=8 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:48:47 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:48:47 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=38056.3 reach4096=13 reach8192=1 dead_below2048=11 max_tile_seen=8192
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:51:29 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:51:29 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=40437.0 reach4096=14 reach8192=1 dead_below2048=8 max_tile_seen=8192
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:52:46 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:52:46 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=42230.7 reach4096=14 reach8192=2 dead_below2048=8 max_tile_seen=8192
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:54:26 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:54:26 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=39996.6 reach4096=14 reach8192=0 dead_below2048=8 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:55:44 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:55:44 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=37956.9 reach4096=16 reach8192=0 dead_below2048=8 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:57:01 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:57:01 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 1000-1049: avg_score=39750.6 reach4096=16 reach8192=0 dead_below2048=8 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:58:41 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:58:41 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 2000-2049: avg_score=38972.6 reach4096=15 reach8192=0 dead_below2048=10 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 02:59:57 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 02:59:57 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 2000-2049: avg_score=37841.6 reach4096=15 reach8192=0 dead_below2048=10 max_tile_seen=4096
OK
All 1 tests passed.
```

## Bench run 2026-10-27 03:01:09 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 6d8ed10
- date: 2026-10-27 03:01:09 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase (#92)...[order-phase] 50 games seeds 2000-2049: avg_score=38025.2 reach4096=13 reach8192=1 dead_below2048=10 max_tile_seen=8192
OK
All 1 tests passed.
```

## Bench run 2026-10-27 03:13:41 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order phase v2 --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 255bef0
- date: 2026-10-27 03:13:41 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase v2 (#92)...[order-phase-v2] expectimax_tt d3 50 games seeds 1000-1049: avg_score=42230.7 avg_moves=2074.2 reach4096=14 dead_below2048=8
[order-phase-v2] expectimax   d2 50 games seeds 1000-1049: avg_score=24297.7 avg_moves=1301.8 reach4096=5 dead_below2048=28
OK
All 1 tests passed.
```

## Death reward learns moves-to-death, but the fragility penalty does not pay (#101)

Issue #101 asked whether a net trained to die fast can measure board fragility, and whether
penalizing fragile boards in `evaluateW` raises the score or the survival rate. Two pieces
shipped. `nn` gains an opt-in `death` variant: per-move reward is -1/reward_scale instead of
the score delta, so the value learns V ≈ -(moves to game over)/1000. Training uses the
backward full-return sweep, not one-step TD — with -1 per move the TD target is identical on
every board and the net collapsed to a constant (observed, then fixed; the naive scale also
blew up to NaN, hence the /reward_scale unit). `evaluateW` gains `Weights.fragility` +
`Weights.death_net`, adding `fragility * (0 - V)`; both default to inert, so every existing
policy is bit-identical unless a caller opts in. The CLI variant is `nn ... death`.

Probe (TEMP harnesses, removed): 1k and 5k death games, then V vs true moves-left on
expectimax_tt depth 3 games (seeds 3000+). In-distribution (the death policy's own games) the
net works: pearson(V, ml) = -0.447 (1k) → -0.606 (5k). On real games it does not:
pearson(V, ml) = 0.023 (1k) / -0.001 (5k) overall, and inside every empties bucket |r| <= 0.12
with mixed signs. The ml <= 100 tail keeps the signal: n=2000, r = -0.230 overall, empties 1-6
r = -0.137 / -0.030 / -0.216 / -0.056 / -0.197 / -0.496, terciles monotone. So V ranks
near-death boards correctly and saturates OOD — exactly the training horizon.

A/B of the penalty (50 games x 2 seed sets, expectimax depth 2, death net trained 5k games):

| fragility | avg (1000-1049) | avg (2000-2049) | dead <2048 (both sets) |
|---|---|---|---|
| 0 (off) | 24,297.7 | 25,385.7 | 52/100 |
| 300 | 19,420.6 | 24,644.3 | 58/100 |
| 1000 | 26,354.1 | 25,614.8 | 50/100 |
| 3000 | 26,345.5 | 24,204.2 | 49/100 |

The λ=1000 bump did not replicate: a paired run (same seeds, off vs on) over 200 games on
fresh sets 1000/2000/3000/4000 gives per-set diffs +2056.4 / +229.1 / -1306.3 / -1490.8,
total -127.9 ± 1477.3, and deaths move the wrong way (93 → 104 of 200). That is noise with a
hint of harm.

Verdict: not adopted. `fragility` stays an opt-in field (tested, default 0 = current
evaluation), nothing sets it, committed weights unchanged. The death variant itself works as
advertised within its horizon and stays as `nn ... death`.

Reading: the tail correlation (r ≈ -0.23..-0.50 at ml <= 100) is real but too thin to rank
mid-game moves, and OOD saturation makes the term nearly constant across candidate boards —
it cannot distinguish moves where it matters. A usable fragility signal would need training
on the real policy's game lengths, not on self-play deaths.

Limits: one machine; the sweep is 4 λ x 100 games plus a 200-game paired run; the probe's
ml <= 100 cut is post-hoc; death nets were trained 1k/5k games only; the A/B plays expectimax
depth 2 (the `Weights` consumer), not nn/nn2.

## Bench run 2026-10-27 03:15:22 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order phase v2 --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 255bef0
- date: 2026-10-27 03:15:22 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase v2 (#92)...[order-phase-v2] expectimax_tt d3 50 games seeds 1000-1049: avg_score=37985.2 avg_moves=1904.5 reach4096=9 dead_below2048=10
[order-phase-v2] expectimax   d2 50 games seeds 1000-1049: avg_score=24403.5 avg_moves=1304.8 reach4096=5 dead_below2048=26
OK
All 1 tests passed.
```

## Bench run 2026-10-27 03:17:36 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order phase v3 --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 255bef0
- date: 2026-10-27 03:17:36 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase v3 (#92)...[order-phase-v3] expectimax_tt d3 50 games seeds 2000-2049: avg_score=38025.2 reach4096=13 dead_below2048=10
[order-phase-v3] expectimax d2 50 games seeds 2000-2049: avg_score=24220.1 reach4096=4 dead_below2048=25
[order-phase-v3] beam 50 games seeds 2000-2049: avg_score=11373.8 reach4096=0 dead_below2048=47
[order-phase-v3] greedy 50 games seeds 2000-2049: avg_score=3109.8 reach4096=0 dead_below2048=50
[order-phase-v3] mcts 50 games seeds 2000-2049: avg_score=30709.7 reach4096=5 dead_below2048=13
OK
All 1 tests passed.
```

## Bench run 2026-10-27 03:20:18 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order phase v3 --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 255bef0
- date: 2026-10-27 03:20:18 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase v3 (#92)...[order-phase-v3] expectimax_tt d3 50 games seeds 2000-2049: avg_score=38383.7 reach4096=16 dead_below2048=10
[order-phase-v3] expectimax d2 50 games seeds 2000-2049: avg_score=25385.7 reach4096=3 dead_below2048=24
[order-phase-v3] beam 50 games seeds 2000-2049: avg_score=12346.7 reach4096=0 dead_below2048=44
[order-phase-v3] greedy 50 games seeds 2000-2049: avg_score=3195.5 reach4096=0 dead_below2048=50
[order-phase-v3] mcts 50 games seeds 2000-2049: avg_score=30709.7 reach4096=5 dead_below2048=13
OK
All 1 tests passed.
```

## Bench run 2026-10-27 03:34:23 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure order phase v3 --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 255bef0
- date: 2026-10-27 03:34:23 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure order phase v3 (#92)...[order-phase-v3] expectimax_tt d3 50 games seeds 2000-2049: avg_score=38383.7 reach4096=16 dead_below2048=10
[order-phase-v3] expectimax d2 50 games seeds 2000-2049: avg_score=25385.7 reach4096=3 dead_below2048=24
[order-phase-v3] beam 50 games seeds 2000-2049: avg_score=12346.7 reach4096=0 dead_below2048=44
[order-phase-v3] greedy 50 games seeds 2000-2049: avg_score=3195.5 reach4096=0 dead_below2048=50
[order-phase-v3] mcts 50 games seeds 2000-2049: avg_score=30709.7 reach4096=5 dead_below2048=13
OK
All 1 tests passed.
```


## Bench run 2026-10-27 03:09:30 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000`
- commit: 255bef0
- date: 2026-10-27 03:09:30 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.7518,"games_per_sec":0.17,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"nn","depth":0,"games":5,"avg_score":69640.8,"avg_moves":3151.2,"best_tile":4096,"latency_ms_per_move":0.1922,"games_per_sec":1.65,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"expectimax_tt","depth":0,"games":5,"avg_score":50344.8,"avg_moves":2391.2,"best_tile":8192,"latency_ms_per_move":0.7048,"games_per_sec":0.59,"description":"depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）"},{"policy":"ga","depth":2,"games":5,"avg_score":27538.4,"avg_moves":1415.2,"best_tile":4096,"latency_ms_per_move":0.0822,"games_per_sec":8.59,"description":"GAで評価の重みを進化させた期待値探索（depth 2）。GA内のholdout(seed 500〜)では+4%だが、ranking seed 1000では素の expectimax より低い"},{"policy":"mcts","depth":0,"games":5,"avg_score":24463.2,"avg_moves":1301.2,"best_tile":2048,"latency_ms_per_move":1.0935,"games_per_sec":0.70,"description":"UCTで候補手を選び、ランダムな対局を繰り返して評価する"},{"policy":"nnbeam","depth":0,"games":5,"avg_score":21096.8,"avg_moves":1143.8,"best_tile":2048,"latency_ms_per_move":0.1200,"games_per_sec":7.28,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"},{"policy":"expectimax","depth":2,"games":5,"avg_score":19964.0,"avg_moves":1093.0,"best_tile":2048,"latency_ms_per_move":0.0957,"games_per_sec":9.56,"description":"2手先まで探索し、ランダムな出現（2か4）を平均して評価する。評価は蛇行順と合体ペアの手作り関数"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5548,"games_per_sec":0.51,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"},{"policy":"beam","depth":0,"games":5,"avg_score":12328.8,"avg_moves":741.2,"best_tile":1024,"latency_ms_per_move":0.0703,"games_per_sec":19.19,"description":"幅32で3手先まで探索し、各空きマスに2が出現すると仮定して評価する"},{"policy":"greedy","depth":0,"games":5,"avg_score":3110.4,"avg_moves":258.4,"best_tile":256,"latency_ms_per_move":0.0014,"games_per_sec":2665.72,"description":"1手先だけ見る。滑らせた盤面の評価が最も高い手を選ぶ（出現や探索はしない）"},{"policy":"random","depth":0,"games":5,"avg_score":966.4,"avg_moves":110.2,"best_tile":128,"latency_ms_per_move":0.0017,"games_per_sec":5060.51,"description":"合法手からランダムに1手選ぶ（比較の基準）"}]}
```

## Bench run 2026-10-27 03:11:35 +0900

- command: `zig build ranking -Doptimize=ReleaseFast -- 1 5 1000`
- commit: 6d8ed10
- date: 2026-10-27 03:11:35 +0900
- exit: 0

```text
{"seed":1000,"runs":1,"games_per_run":5,"entries":[{"policy":"nn2","depth":2,"games":5,"avg_score":71880.0,"avg_moves":3267.2,"best_tile":4096,"latency_ms_per_move":1.7520,"games_per_sec":0.17,"description":"TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）"},{"policy":"nn","depth":0,"games":5,"avg_score":69640.8,"avg_moves":3151.2,"best_tile":4096,"latency_ms_per_move":0.1919,"games_per_sec":1.65,"description":"TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）"},{"policy":"expectimax_tt","depth":0,"games":5,"avg_score":51376.0,"avg_moves":2426.2,"best_tile":4096,"latency_ms_per_move":0.6913,"games_per_sec":0.60,"description":"depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）"},{"policy":"expectimax","depth":2,"games":5,"avg_score":41107.2,"avg_moves":2019.4,"best_tile":4096,"latency_ms_per_move":0.0854,"games_per_sec":5.80,"description":"2手先まで探索し、ランダムな出現（2か4）を平均して評価する。評価は蛇行順と合体ペアの手作り関数"},{"policy":"ga","depth":2,"games":5,"avg_score":27538.4,"avg_moves":1415.2,"best_tile":4096,"latency_ms_per_move":0.0823,"games_per_sec":8.59,"description":"GAで評価の重みを進化させた期待値探索（depth 2）。GA内のholdout(seed 500〜)では+4%だが、ranking seed 1000では素の expectimax より低い"},{"policy":"mcts","depth":0,"games":5,"avg_score":24463.2,"avg_moves":1301.2,"best_tile":2048,"latency_ms_per_move":1.0845,"games_per_sec":0.71,"description":"UCTで候補手を選び、ランダムな対局を繰り返して評価する"},{"policy":"nnbeam","depth":0,"games":5,"avg_score":21096.8,"avg_moves":1143.8,"best_tile":2048,"latency_ms_per_move":0.1211,"games_per_sec":7.22,"description":"NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）"},{"policy":"beam","depth":0,"games":5,"avg_score":14368.8,"avg_moves":817.8,"best_tile":2048,"latency_ms_per_move":0.0674,"games_per_sec":18.14,"description":"幅32で3手先まで探索し、各空きマスに2が出現すると仮定して評価する"},{"policy":"nnmcts","depth":0,"games":5,"avg_score":12864.8,"avg_moves":763.4,"best_tile":1024,"latency_ms_per_move":2.5553,"games_per_sec":0.51,"description":"UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）"},{"policy":"greedy","depth":0,"games":5,"avg_score":3300.8,"avg_moves":272.2,"best_tile":256,"latency_ms_per_move":0.0004,"games_per_sec":8437.64,"description":"1手先だけ見る。滑らせた盤面の評価が最も高い手を選ぶ（出現や探索はしない）"},{"policy":"random","depth":0,"games":5,"avg_score":966.4,"avg_moves":110.2,"best_tile":128,"latency_ms_per_move":0.0005,"games_per_sec":18373.91,"description":"合法手からランダムに1手選ぶ（比較の基準）"}]}
```


## Bench run 2026-10-27 10:34:47 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure fragility --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 7ed57f9
- date: 2026-10-27 10:34:47 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure fragility probe (#101)...#101 probe: 43057 afterstates, expectimax_tt d3 games (seeds 3000+), death net 1000 games
#101 overall: pearson(empties, ml) = 0.278, pearson(V, ml) = 0.023
#101 empties= 1 n=  1292 pearson(V, ml)=-0.007 tercile_ml=845.5/867.3/884.5
#101 empties= 2 n=  2757 pearson(V, ml)=0.070 tercile_ml=808.4/931.3/957.3
#101 empties= 3 n=  5063 pearson(V, ml)=0.089 tercile_ml=961.9/1011.3/1136.3
#101 empties= 4 n=  7244 pearson(V, ml)=0.120 tercile_ml=1051.4/1129.0/1334.4
#101 empties= 5 n=  7920 pearson(V, ml)=0.078 tercile_ml=1175.2/1276.7/1365.2
#101 empties= 6 n=  7431 pearson(V, ml)=0.037 tercile_ml=1327.7/1406.8/1430.1
#101 empties= 7 n=  5393 pearson(V, ml)=-0.016 tercile_ml=1578.4/1512.4/1584.1
#101 empties= 8 n=  3245 pearson(V, ml)=-0.006 tercile_ml=1724.7/1644.4/1731.7
#101 empties= 9 n=  1614 pearson(V, ml)=-0.033 tercile_ml=1917.8/1791.4/1900.1
#101 empties=10 n=   657 pearson(V, ml)=-0.104 tercile_ml=2280.3/2049.6/1991.8
#101 empties=11 n=   227 pearson(V, ml)=-0.044 tercile_ml=2088.6/2065.2/2075.5
#101 empties=12 n=   119 pearson(V, ml)=0.011 tercile_ml=2242.3/2157.5/2486.9
OK
All 1 tests passed.
```


## Bench run 2026-10-27 10:38:38 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP death in-distribution --dep core --dep parallel -Mroot=ai/solver/nn.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mparallel=ai/solver/parallel.zig`
- commit: 7ed57f9
- date: 2026-10-27 10:38:38 +0900
- exit: 0

```text
1/1 nn.test.TEMP death in-distribution (#101)...#101 in-dist: n=1706 pearson(V, ml)=-0.447
OK
All 1 tests passed.
```


## Bench run 2026-10-27 10:45:17 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure fragility probe robust --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 7ed57f9
- date: 2026-10-27 10:45:17 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure fragility probe robust (#101)...#101 probe-5k: 43057 afterstates
#101 overall-5k: pearson(empties, ml) = 0.278, pearson(V, ml) = -0.001
#101 empties= 1 n=  1292 pearson(V, ml)=-0.059 tercile_ml=932.2/894.4/771.2
#101 empties= 2 n=  2757 pearson(V, ml)=0.059 tercile_ml=825.1/928.9/942.9
#101 empties= 3 n=  5063 pearson(V, ml)=0.076 tercile_ml=968.1/1030.1/1111.2
#101 empties= 4 n=  7244 pearson(V, ml)=0.125 tercile_ml=1041.9/1133.1/1339.8
#101 empties= 5 n=  7920 pearson(V, ml)=0.084 tercile_ml=1177.6/1281.7/1357.8
#101 empties= 6 n=  7431 pearson(V, ml)=0.053 tercile_ml=1314.2/1395.1/1455.3
#101 empties= 7 n=  5393 pearson(V, ml)=-0.012 tercile_ml=1540.1/1542.0/1592.7
#101 empties= 8 n=  3245 pearson(V, ml)=0.004 tercile_ml=1696.2/1655.0/1749.6
#101 empties= 9 n=  1614 pearson(V, ml)=-0.046 tercile_ml=1950.4/1761.7/1897.3
#101 empties=10 n=   657 pearson(V, ml)=-0.112 tercile_ml=2267.5/2089.5/1964.7
#101 empties=11 n=   227 pearson(V, ml)=-0.057 tercile_ml=2203.4/1954.8/2071.2
#101 empties=12 n=   119 pearson(V, ml)=0.032 tercile_ml=2325.4/2250.3/2319.6
OK
All 1 tests passed.
```


## Bench run 2026-10-27 10:48:08 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP death in-dist robust --dep core --dep parallel -Mroot=ai/solver/nn.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mparallel=ai/solver/parallel.zig`
- commit: 7ed57f9
- date: 2026-10-27 10:48:08 +0900
- exit: 0

```text
1/1 nn.test.TEMP death in-dist robust (#101)...#101 in-dist-5k: n=1385 pearson(V, ml)=-0.606
OK
All 1 tests passed.
```


## Bench run 2026-10-27 10:55:16 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure fragility probe tail --dep core --dep nn -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: 7ed57f9
- date: 2026-10-27 10:55:16 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure fragility probe tail (#101)...#101 tail-5k: n=2000 pearson(V, ml)=-0.230
#101 tail empties= 1 n=   336 pearson(V, ml)=-0.137 tercile_ml=36.2/32.6/29.3
#101 tail empties= 2 n=   516 pearson(V, ml)=-0.030 tercile_ml=45.2/44.1/38.6
#101 tail empties= 3 n=   514 pearson(V, ml)=-0.216 tercile_ml=56.8/54.4/43.9
#101 tail empties= 4 n=   357 pearson(V, ml)=-0.056 tercile_ml=62.0/63.2/61.1
#101 tail empties= 5 n=   191 pearson(V, ml)=-0.197 tercile_ml=70.7/70.3/67.1
#101 tail empties= 6 n=    77 pearson(V, ml)=-0.496 tercile_ml=79.5/71.5/57.4
OK
All 1 tests passed.
```


## Bench run 2026-10-27 13:08:07 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure fragility A/B --dep core --dep nn --dep parallel -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: da3364c
- date: 2026-10-27 13:08:07 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure fragility A/B (#101)...#101 fragility lam=0 seed=1000 avg=24297.7 moves=1301.8 reach4096=5 reach8192=0 dead=28
#101 fragility lam=0 seed=2000 avg=25385.7 moves=1360.2 reach4096=3 reach8192=0 dead=24
#101 fragility lam=300 seed=1000 avg=19420.6 moves=1082.9 reach4096=2 reach8192=0 dead=34
#101 fragility lam=300 seed=2000 avg=24644.3 moves=1331.2 reach4096=2 reach8192=0 dead=24
#101 fragility lam=1000 seed=1000 avg=26354.1 moves=1388.7 reach4096=5 reach8192=0 dead=23
#101 fragility lam=1000 seed=2000 avg=25614.8 moves=1365.2 reach4096=5 reach8192=0 dead=27
#101 fragility lam=3000 seed=1000 avg=26345.5 moves=1393.9 reach4096=5 reach8192=0 dead=22
#101 fragility lam=3000 seed=2000 avg=24204.2 moves=1294.8 reach4096=4 reach8192=0 dead=27
OK
All 1 tests passed.
```


## Bench run 2026-10-27 13:29:16 +0900

- command: `/nix/store/bg372xp0r29mh7qigkbx1gb3j50nnmcl-zig-0.17.0/bin/zig test -OReleaseFast --test-filter TEMP measure fragility A/B v2 --dep core --dep nn --dep parallel -Mroot=ai/solver/solver.zig -OReleaseFast -Mcore=src/core.zig -OReleaseFast --dep core --dep parallel -Mnn=ai/solver/nn.zig -OReleaseFast -Mparallel=ai/solver/parallel.zig`
- commit: da3364c
- date: 2026-10-27 13:29:16 +0900
- exit: 0

```text
1/1 solver.test.TEMP measure fragility A/B v2 (#101)...#101 v2 set 1000: paired mean diff = 2056.4
#101 v2 set 2000: paired mean diff = 229.1
#101 v2 set 3000: paired mean diff = -1306.3
#101 v2 set 4000: paired mean diff = -1490.8
#101 v2 total n=200 avg0=25400.4 avg1000=25272.5 paired diff=-127.9 stderr=1477.3 dead0=93 dead1=104
OK
All 1 tests passed.
```

## 楽観的初期化 (OI) の検証 (#107, 2026-10-28)

TD 学習の停滞を破る手法として、文献 (Guei et al. 2021) の楽観的初期化を `nn` の TD(0) afterstate
学習に移植し、`base` と同条件で比べた。**結論: この MLP 構成では不採用**。

### 実装

`nn.scaleInit(net, k)` で初期重み (w1 / w2 / w3) を k 倍してから学習する。`nn` の variant として
`oi` (k=4) と `oi16` (k=16) を追加した（既定の学習経路・committed weights は不変）。
出力バイアス `b3` を楽観的に持ち上げる方式は採らなかった: 定数を足しても全 afterstate が同じだけ
ずれるので、手の argmax は変わらず、学習者に見えるデータが変化しない（no-op）。

### プロトコル

`zig build nn -Doptimize=ReleaseFast -- <games> 10 2000 0.003 2 <variant> 30 <tmp weights> <train_seed> par`

- 訓練ゲーム数 4k / 20k / 100k / 400k、train_seed 1 と 2、lr 0.003（既存の線形減衰のまま）
- 評価は train と無関係な seed 2000.. の 20 局 × 2 レンジ、`nn 2-ply (deep)` の avg score
- 同じ row は同一 seed・同一ゲーム数で、初期重みのスケールだけが違う

### 結果（avg score、nn 2-ply）

| 訓練ゲーム数 | base seed1 | base seed2 | oi (×4) seed1 | oi (×4) seed2 | oi16 (×16) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,000 | 12,151 | 11,719 | 6,841 | 7,750 | 10,333 |
| 20,000 | 21,279 | 15,212 | 8,813 | 13,069 | 10,333 |
| 100,000 | 30,145 | 25,669 | 14,113 | 11,069 | 10,333 |
| 400,000 | 76,977 | — | 26,952 | — | — |

expectimax depth 2 は同じ評価 seed で 36,410（base は 100k で追い抜く）。

### わかったこと

- **×16 は学習が発散する。** 保存された重みの最大絶対値が 1e+291、NaN が 4,374/18,561 個。
  NaN になると比較がすべて偽になり手の選択が固定されるため、訓練ゲーム数も train_seed も
  変えて同じスコア (10,333) になる。1-ply と 1-spawn lookahead も同値で、価値関数が定数に潰れた。
- **×4 は発散しないが学習が大幅に遅い。** 100k で base の 1/2、400k でも 27.0k 対 77.0k。
  曲線はまだ上昇中なので最終性能が同じになる可能性は残るが、少なくとも「少ないゲームで同じ
  スコアに到達する」という OI の主張は逆方向に出た。
- OI の効果は「初期値の分散を上げて自己対戦の探索を促す」ことに尽きるが、この MLP は既定の
  ランダム初期化ですでに分散を持ち、スケールを上げると TD(0) の更新が不安定になるだけだった。

### 残作業

- **MS-TD (段階別評価) は未着手。** 空きマス数などの段階ごとにパラメータ集合を持つ必要があり、
  重みファイル形式と `Cache`/`train` の変更を伴う。効果も「学習が遅くなる」という報告がある
  (Matsuzaki 2017) ため、別 issue で設計から始める。

## n-tuple の systematic パターン集合 (#106, 2026-10-29)

### 背景

n-tuple ネットワークは 6-tuple を 6 本使っていた（`{0,1,2,3,4,5}`, `{0,1,2,4,5,6}`,
`{0,1,2,5,6,7}`, `{0,1,4,5,6,9}`, `{0,1,5,6,9,10}`, `{0,4,5,9,10,14}`）。これは Szubert & Jaśkowski の
systematic な集合ではなく、文献の 1M ゲーム結果 (1-ply 100,178) には遠く及ばない。
足りない要因はパターン集合か学習率かを切り分けるため、まず学習率だけを掃引した。

### 学習率だけを動かす（既存 6 パターン、1M ゲーム、seed 1、20 局 × seed 2000）

| lr | 1-ply | +1 spawn | depth 2 |
| ---: | ---: | ---: | ---: |
| 0.01 | 8,831.8 | 17,093.4 | 30,750.4 |
| 0.0025 | 6,953.4 | 16,435.0 | 25,763.4 |

低い学習率の方が悪い。文献の α=0.0025 は 2 段階 (afterstate) TD とセットの値で、
そのまま持ってきても効かない。**パターン集合が本質**と判断して次を実装した。

### 追加したパターン

2x3 矩形の対称軌道を列挙すると、12 個の位置は 2 軌道に分かれる。

- 8 位置の軌道（既存の `{0,1,2,4,5,6}` が代表）
- 4 位置の軌道（中央寄り）→ `{4,5,6,8,9,10}`

`indices` が 8 対称変換の最小を取る実装なので、同一軌道の別位置を足しても同じ表の重複になる。
よって**追加は 1 本だけ**で、7 本（`.wide`）が実質的な上限。`Set = .light | .full | .wide` を追加し、
既存の `ntuple_weights.bin`（6 本）は `.full` としてそのままロードできる。

| set | パターン数 | entries | raw |
| --- | ---: | ---: | ---: |
| `.light` | 5 | 14,253,888 | 54.4 MiB |
| `.full` | 6 | 17,104,896 | 65.3 MiB |
| `.wide` | 7 | 33,882,112 | 129.3 MiB |

7 本目で entries が 2 倍になるのは、中央の 6-tuple が位置あたりの組み合わせ数が多いため。
`gzip -9` で 2,963,242 B (2.9 MB) まで縮むが、static 配信で 1 MB 未満に収めるには量子化が要る。

### 結果（wide、20 局 × seed 2000、評価は train と無関係な seed）

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

expectimax depth 2 は同じ評価 seed で 36,410（1-ply の wide はすでに 5 倍近い差で上回る）。

### わかったこと

- **パターン 1 本の追加で 1M ゲームの depth 2 が 30,750 → 39,344 (+28%)。** 同じ計算量で
  expectimax depth 2 を超える。学習率の掃引より効果が大きい。
- ゲーム数を 5M に増やすと 50,103（+27%）。まだ頭打ちが見えていない。
- 学習率は 0.01〜0.02 が良く、0.0025 は明確に悪い（34,693）。文献の α はパターン集合と
  セットで意味を持つ値で、単独では持ち込めない。
- seed 差は 5M で 50,103 対 45,586（±5%）あり、この規模の比較は複数 seed が要る。
- 1 ゲームあたりの手数が 2,400 を超える（depth 2 の評価）。8192 到達はまだ出ていない。

### 残作業

- `algo=ntuple` の API / WASM への接続はしていない。wide の表は 129 MiB で、WASM に同梱するには
  量子化（f32 → int8 など）と 1-ply での運用が前提になる。`nn2` の 72,205 にはまだ届かないため、
  接続は学習が頭打ちになってから判断する。
- 学習時間は 1M で 554 s、5M で 4,197 s（4 並列時の実測、単独なら ~2,000 s）。パターンを増やすより
  ゲーム数を増やす方がまだ効く。
- ただし 5M→10M では depth 2 が 47,374（3 seed 平均）→ 51,951 で +10%、1-ply は 17,304 → 20,014 で
  +16%。伸びは鈍化している。depth 2 は 50k 前後で頭打ちが見え始めた（9,300 s の学習で 8192 到達も無し）。
  次に効かせるならパターンの追加（`.wide` が 7 本で上限）ではなく、量子化して 2-ply を安く回すか、
  afterstate の 2 段階 TD に切り替えるかのどちらか。

## expectimax_id の統計 (#104, 2026-10-30)

`docs/evalab.md` の反復深化の節は 50 局（5 × 10）だったので、標本を増やして確かめた。

| policy | 予算 | 局数 | avg score | avg moves | best | 到達深さ | games/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expectimax（重み付き, depth 3） | — | 200 | 52,735.7 | 2,528.6 | 8192 | 3 | 0.18 |
| expectimax_id | 2 ms | 100 | **75,202.3** | 3,429.3 | 8192 | 8（上限） | 0.07 |

seed 9000。depth 3 は `zig build evalab -- "baseline" 4 50 3 9000`、id は
`zig build solve -Doptimize=ReleaseFast -- 10 10 3 9000 expectimax_id`（予算は `solver.zig` の
`id_budget_ms`）。**+43%** で、50 局のときの +34%（1 ms）と整合する。

注意: `baseline` は `#117` で `.monotonic = 32` を採用済み（`ai/solver/solver.zig:52`）。
したがって `monotonic=32` を明示しても `baseline` と同じ結果になる（no-op）。

## MS-TD（段階別評価）の検証 (#107, 2026-10-30)

### 実装

段 = afterstate の空きマス数で 7 分割（`0..2`, `3..4`, `5..6`, `7..8`, `9..10`, `11..12`, `13..16`）。
段ごとに独立した MLP を持ち、TD(0) の更新は遷移先 afterstate の段のネットだけに当てる。
探索からは `nn.Value = union(enum) { single: *const Net, staged: *const Staged }` を通して透過に使えるので、
1-ply / 2-ply の探索コードは共有のまま staged 価値でも動く。重みファイルは専用の magic を持ち、
既存の単一ネット形式（`nn_weights.bin`）はそのままロードできる。

### プロトコル

- 学習: `zig build nn -Doptimize=ReleaseFast -- <games> 10 2000 0.003 2 {base,mstd} 30 /tmp/w.bin 1`
- 評価: 20 局（10 局 × 2 レンジ、seed 2000 から）で全 variant が同一 seed。`par` は使わない（下記）
- 評価対象: 1-ply / 1 spawn lookahead / 2-ply (deep) / expectimax depth 2（参照）

### 結果（avg score）

| 訓練 | 1-ply | +1 spawn | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: | ---: |
| base 100k | 44,901 | 69,546 | 87,824 | 1,064 s |
| mstd 100k | 47,427 | 73,045 | 89,079 | 1,414 s |
| mstd 400k | **2,288** | 2,288 | **10,333** | 1,447 s |

expectimax depth 2 は同じ評価 seed で 36,410（100k の時点で both が追い抜く）。

### わかったこと

- **100k では base と同等**（89.1k 対 87.8k、stderr ±4.9k / ±1.1k）。段を分けた分の容量増は
  この規模では誤差に埋もれる。
- **400k では崩壊する。** 1-ply 2,288、2-ply 10,333。学習ログの総手数は 147.6M = 369 手/ゲーム
  （100k では 143.1M = 1,431 手/ゲーム）なので、100k 以降に方策が短手数に落ち、以降の自己対戦が
  学習信号を失っている。段の境界をまたぐブートストラップ（γ=1、報酬は終端のみ）は段ごとに
  独立した価値スケールを持つため、段間の整合が崩れると一気に発散しやすい。
  `lr * (1 - 0.9 * frac)` の減衰を入れても抑えられなかった。
- 対策案（未検証）: 段の境界をまたぐときに目標を切り離す（段内ブートストラップに限定する）、
  または段ごとの学習率・γ を分ける。ただし MS-TD の利得自体が 100k で見えていないので、
  この MLP では追う価値が薄い。

### 結論

この MLP + TD(0) の構成では **MS-TD は採用しない**。100k では同等、400k では不安定化する。
実装は variant `mstd` として残す（`zig build nn -- ... mstd ...`）。

## 副作用として判明: `par` バッチ学習は 3 倍遅い (2026-10-30)

MS-TD の比較用に `par` 無しの base を回したところ、同じ 100k ゲーム・lr 0.003・seed 1・同一評価 seed で
大きな差が出た。

| variant | 1-ply | +1 spawn | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: | ---: |
| base + `par` | 8,832 | 17,093 | 30,145 | 187 s |
| base（par なし） | 44,901 | 69,546 | 87,824 | 1,064 s |

`par` は 8 ゲームぶんの勾配を平均し norm clip して 1 ステップにするので、1 ゲームあたりの
実効学習率がおよそ 1/8 になる。**nn の学習は既定で `par` を使わない（または lr をバッチ数倍にする）**のが正しく、
これまでの nn 系の計測（OI の A/B など）は par 前提だったため絶対値が控えめに出ている
（同一条件どうしの相対比較はそのまま有効）。速度は 187 s 対 1,064 s（5.7 倍）なので、
速度が要る場面では lr を上げた par を使う選択肢が残る（未検証）。

## n-tuple 表の量子化と `algo=ntuple` の接続 (#129, 2026-10-30)

`.wide` の n-tuple 表は 33,882,112 entries × f32 = 129.3 MiB あり、そのままでは同梱できない。
int16 に量子化して 64.6 MiB にし、`solver.Policy.ntuple` として API から使えるようにした。

### 形式

- `ai/solver/ntuple.zig` の `Quantized`: int16 の本体 + パターンごとのスケール（f64 × 7）
- 量子化は `q = round(w / scale)`、`scale = max|w| / 32767`（表ごと）。復元は `w ≈ q * scale`
- ファイルは magic `2048-NQ\0` + version + endian + パターン数 + 表長 + スケール + i16 本体
  （既存の f32 形式 `2048-NT\0` はそのまま読める）
- 変換は `zig build ntuple-quant -- <in.bin> <out.q16.bin>`。`zig build ntuple` も出力パスが
  `.q16.bin` で終われば量子化して保存する
- 実際のステップはパターンあたり 1.7e-4（値の範囲が ±6 程度なので、丸め誤差は無視できる）

### 精度（10M ゲーム wide 表、20 局 seed 2000、量子化表を f32 に戻して同じ評価コードで測定）

| 表 | 1-ply | +1 spawn | depth 2 |
| --- | ---: | ---: | ---: |
| f32（129.3 MiB） | 20,013.8 | 32,243.4 | 51,950.6 |
| int16（64.6 MiB） | 20,060.8 | 28,912.0 | 53,592.2（8192 到達） |

差は 20 局のばらつき（±10%）に埋もれる。**量子化による劣化は観測できない。**

### 接続

- `Policy.ntuple` を enum の末尾に追加。`solver.choose` は `ntuple.bestMoveQuantized`（1-ply）を呼ぶ
- 表は**遅延ロード**: そのポリシーが初めて選ばれたとき、スレッドごとに 1 回だけ
  `ai/solver/ntuple_wide_q16.bin` を `std.heap.page_allocator` に読む（64.6 MiB）
- `/api/solve?algo=ntuple` は表が無ければ 503 と理由を返す（`solver.ntupleAvailable()`）
- WASM はファイルを読めないので `src/wasm.zig` の `algo_count()` が `-1` して一覧に出さない
  （enum の末尾に置いたのはこのため）
- `ranking.zig` の既定リストには入れていない（64.6 MiB の表を CI/計測で読ませない）

### 副産物: WASM ビルドが main で壊れていた

`#104` の `clockNow()` が `std.Io.Threaded.global_single_threaded.io()` を呼ぶため、
wasm32-freestanding では `std.Io.Threaded` の `getrandom` / `IOV_MAX` 解決に失敗して
`zig build wasm` が通らなくなっていた（`#104` の PR 以降、誰も wasm をビルドしていなかった）。
`clockNow` を freestanding 用のスタブ（固定タイムスタンプを返す）に分岐させて修復した。
当初ここには「その場合 `expectimax_id` は深さ 2 で止まる」と書いていたが、これは**間違いだった**。
スタブは 0 を返すので `elapsedMs()` が常に 0 になり、`while (... elapsedMs(start) < budget_ms)` が
常に真 → `id_max_depth = 8` まで深化して 1 手 80 s 以上かかっていた（#145）。いまは
`shouldDeepen()` が「時計が進むターゲット」を条件に入れているので、wasm では `id_min_depth = 2` に固定される。

### 限界

- 1-ply のみ。`chance` / `deep`（n-tuple 版の 1 spawn / 2-ply）は `solver` 側からは未接続
- 1-ply で約 20,000 なので `nn2`（72,205）には遠く、`ranking.json` には入れていない
- 64.6 MiB を git 管理している（既存の `ntuple_weights.bin` 68.4 MiB と合わせて約 133 MiB）

## n-tuple の TD 目標: afterstate + max backup (#131, 2026-10-31)

現行の TD 目標は「実際に選んだ手の次の afterstate の価値」＝on-policy なサンプル目標。
n-tuple の文献（Szubert ら）で使われる **max backup**、つまり出現後の盤面で
`r + max_a V(afterstate(s', a))` を目標にする方式に変えて、同じ総ゲーム数で比較した。

### 実装

- `ntuple.Target = enum { on_policy, max_backup }`、`trainGameWith(net, rng, lr, mode, target)`
  （`trainGame` は `trainGameWith(..., .on_policy)` のままで挙動不変。同じ seed で同じ表になることをテストで固定）
- `.max_backup` の目標は `p.reward + bestAfterValue(net, state)`（`state` は出現後の盤面）
- variant 名 `maxb` / `maxbchance` / `maxbdeep`（いずれも `.wide` + 指定の target）

### 結果（20 局、評価 seed 2000、lr 0.01、`.wide`）

| 学習 | 1-ply | depth 2 | 学習時間 |
| --- | ---: | ---: | ---: |
| on-policy 1M, seed 1 | 13,795.0 | 39,344.0 | 554 s |
| on-policy 1M, seed 2 | 14,097.8 | 41,389.2 | 613 s |
| maxb 1M, seed 1 | **19,347.4** | 43,102.0 | 720 s |
| maxb 1M, seed 2 | 13,607.4 | 37,849.2 | 758 s |
| on-policy 5M, seed 1 | 15,777.6 | 50,103.2 | 4,197 s |
| **maxb 5M, seed 1** | **17,567.6** | **52,953.4** | 6,436 s |

- 1M の平均: 1-ply 13,946 → 16,477（+18%）、depth 2 40,367 → 40,476（±0%）
- 5M の同一 seed 比較: 1-ply +11%、depth 2 +5.7%（seed 間のばらつき ±5% と同程度）
- 学習は 1.2〜1.5 倍遅い（`bestAfterValue` が 1 手あたり 4 回の表引きを足す。ゲームも長くなる）

### 解釈

- **1-ply の価値は明確に良くなる**。サンプルされた手ではなく最善手の価値で更新するので、
  表が「今の自分の方策」より一歩先を評価するようになる。
- **2-ply 探索の利得は縮む**（maxb 5M は 52,953 / 1-ply 17,568 = 3.0 倍、on-policy は 50,103 / 15,778 = 3.2 倍）。
  価値自体が先読みを含むぶん、追加の探索で得られるものが減る。
- 合計では depth 2 で +5.7% だが seed のばらつきに埋もれる。**既定の学習目標は変えない**
  （`maxb` 系は variant として残す）。
- 学習時間あたりでは on-policy が有利（6,436 s で +5.7% は割に合わない）。

### 使い方

```bash
zig build ntuple -Doptimize=ReleaseFast -- 5000000 20 2000 0.01 maxb /tmp/table.bin 1
```

## 既定 nn 重みの学習レシピ修正 (#133, 2026-10-31)

`#107` の MS-TD 検証中に、`par` バッチ学習が実効学習率を約 1/8 に薄めていることが分かった
（同一 100k ゲーム・lr 0.003 で `par` あり 30,145 / なし 87,824）。それを受けて、
**`par` なしで 100k ゲーム学習した `nn`** を `expectimax_id` と同じプロトコル
（100 局 = 10 ゲーム × 10 seed レンジ, seed 9000–9009, 2-ply）で測り、既定重みと比べた。

| 重み | 1-ply | 2-ply | avg moves | 出典 |
| --- | ---: | ---: | ---: | --- |
| 既定 `nn_weights.bin`（200k ゲーム, 旧レシピ） | 41,611 ± 2,201 | **72,397 ± 832** | 3,282.6 | 再測 |
| 100k / par なし / train seed 1 | 36,562 ± 2,198 | **86,864 ± 2,503** | 3,865.8 | 本節 |
| 100k / par なし / train seed 2 | 39,711 ± 1,895 | **93,852 ± 3,173** | 4,119.0 | 本節 |
| `expectimax_id` 2 ms（`#104` の記録） | — | 75,202 | 3,429.3 | `#104` |
| `expectimax` depth 2（同条件の手作り評価） | — | 34,000 ± 1,616 | 1,745.1 | 再測 |

別の評価 seed（9100–9109）での追試: train seed 2 = 89,622 ± 1,674、train seed 1 = 81,540 ± 2,449。
seed 2 が両方の評価で上なので採用した。2 seed 平均は **90,358**。

### 解釈

- **既定の `nn` / `nn2` は 2-ply で 90k 前後**になり、`expectimax_id`（75,202）を上回る。
  しかも 0.57 games/s で、`expectimax_id`（0.07 games/s）の 8 倍速い。
- 既定重みが悪いのは `par` ではなく**学習ゲーム数**。100k がこの recipe のピークで、
  400k まで回すと 73,007 に劣化する（lr 一定のまま延ばすと自己対戦が短くなり学習信号が減る）。
  さらに延ばすなら lr 減衰か早期終了が要る。
- 評価は 2-ply（`nn.deepMove`）で、leaves/move ≈ 1,040、0.41 ms/手。1-ply は 36k〜42k で伸びない。
- golden テストは 1 局だけなので、`nn`（1-ply）の golden は 70,364 → 32,380 と下がった。
  母集団の 1-ply は 41,611 → 39,711 でほぼ同等なので、**1 局の golden で優劣を判断しない**こと
  （`nn2` の golden は 73,964 → 77,164 に上がっている）。

### 再現

```bash
zig build nn -Doptimize=ReleaseFast -- 100000 10 9000 0.003 10 base 30 ai/solver/nn_weights.bin 2
```

## nn の学習を 100k より延ばせるか (#135, 2026-10-31)

`#133` で 100k ゲーム / `par` なしが既定になったが、`ai/solver/nn_main.zig:220` を見ると
**`par` なしの経路には既に線形減衰（`lr * (1 - 0.9 * frac)`、100% → 10%）が入っている**。
したがって「減衰なしで延ばすと劣化する」ではなく「減衰させても延ばすと劣化する」が正しく、
残るノブは減衰の形と fine tuning だった。

train seed 3 で統一し、評価は 100 ゲーム（10 × 10 seed レンジ, seed 9000–9009, 2-ply）。

| 学習 | 1-ply | 2-ply | 学習時間 |
| --- | ---: | ---: | ---: |
| 100k / lr 0.003（減衰 0.003 → 0.0003, 対照） | 36,743 ± 1,489 | **81,296 ± 1,986** | 1,229 s |
| 200k / lr 0.003（減衰を 200k に引き伸ばす） | 39,633 ± 1,998 | **70,210 ± 1,213** | 2,677 s |
| 100k から +100k / lr 0.001（0.001 → 0.0001 で fine tuning） | 45,662 ± 2,285 | **74,841 ± 476** | 1,800 s |

### 解釈

- **100k がこの recipe の最適点**。減衰を引き伸ばして 200k 回すと −14%、低 lr で追化学習すると −8%。
- fine tuning は **1-ply を大きく上げる**（36,743 → 45,662, +24%）が **2-ply は下げる**。
  低 lr で追い込むと価値が「今の方策の先読み」に寄り、探索の上乗せが効かなくなるため。
  1-ply だけを見て採用判断すると罠になる。
- 100k の train seed 間のばらつきは大きい: seed 1 = 86,864 / seed 2 = 93,852 / seed 3 = 81,296
  （平均 87,337 ± 6,300）。`#133` で出荷した seed 2 の 93,852 は**上振れの側**なので、
  「既定 nn は 90k」は 1 seed の値であって期待値ではない。母集団としては 85k 前後と見るのが妥当。
- seed を選んで出荷している以上、この種の数値には選択バイアスが乗る。新しい候補を比べる時は
  同一 train seed で並べること（本節はそれを守っている）。

## ranking 表の再生成と速度列（#136）

`src/web/ranking.json`（`localhost:8080` のUIが読む比較表）は 2026-10-26 の生成物で、
`latency_ms_per_move` / `games_per_sec` を出す前の形式だった。おかげでプレビューの表には
速度が無く、スコアも #133 出荷前のまま古かった（nn2 が 72,397 と出ていた）。

`zig build ranking -Doptimize=ReleaseFast -- 5 20 1000`（5 run × 20 ゲーム、seed 1000）で
作り直し、UI の表に games/s と ms/move の列を足した。生成物を `src/web/ranking.json` に
置いたら**ビルドし直す**必要がある（`src/server.zig` が `@embedFile` している）。

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

計測中は nn3 のベンチマークが 2 コアを占有していたので、絶対速度は低めに出ている。表の中では
同じ条件なので順序（nn2 ≪ nn ≪ expectimax_tt ≪ expectimax）は信用できる。nn2 は 1 手 3.2 ms
（expectimax の 17 倍遅い）でスコアは 2.4 倍。速度で選ぶなら nn、強さで選ぶなら nn2。

## WASM 版はモデル選択をやめて nn2 固定（#143）

（方針は #152 で `nn3` に、速度は「常に最大」から select に変えた。以下の実測値はその時点のもの）

静的版（`zig build wasm`）は 13 方針のドロップダウンと速度のドロップダウンを出していたが、
ブラウザで選ばせる意味が薄いので外した。実測（同一マシン、V8 上の wasm、1 手あたり、
終盤盤面、depth 2、単一スレッド）:

| policy | ms/move | native の avg score |
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

- **`nn2` を固定**にした。1 手 1.3 ms で、最大速度の自動再生でも 1 フレーム 8 ms の予算に収まる。
  速度優先なら `nn`（0.13 ms / 69,717）、強さ優先なら `nn3`（22.9 ms / 116,205）だが、
  `nn3` はメインスレッドを掴むので既定にはしない。（#152 で `nn3` 既定に戻した。上の 22.9 ms は
  終盤の盤面だけの値で、序盤は 0.3〜0.4 s かかる）
- 方針は **index ではなく名前で解決**する（`algo_name` を走査して `"nn2"` を探し、無ければ
  `expectimax`）。`Policy` の並びが変わったときに黙って別の方針になるのを防ぐ。
- 自動再生は `requestAnimationFrame` ごとに 8 ms ぶんだけ解いて返す。以前の "max (no delay)" は
  `while` で 1 ゲームを同期実行していたので、タブがその間固まっていた。
- **`expectimax_id` はブラウザで壊れている**: freestanding の `clockNow` は常に 0 なので時間予算が
  効かず、`0 < budget` が真のままで `id_max_depth = 8` まで深化する。`docs/solver-history.md` の
  以前の記述（「深さ 2 で止まる」）は実装と食い違っていた。選択肢から外したので実害は無いが、
  WASM で `expectimax_id` を出すなら時計かノード上限が要る。
- WASM の制約（実測）: モジュール 763,545 B、初期メモリ 2.44 MiB（`memory.grow` 無し）、
  スタック 1 MiB、TT 640 KiB + `cached_net` 148 KB が bss。スレッドは使わない。

## 静的版を nn3 に戻し、再生速度を選べるようにした（#152）

（#154 で方針 select が戻り、同梱する方針は `nn2` と `nn3` の 2 つになった。以下の
「方針は固定」「smoke test は `SOLVER_NAME` を見る」は #152 時点の記録）

#143 で `nn2` 固定・常に最大速度にしたが、最大速度では 1 秒に何十手も進んで何が起きているかを
目で追えない。既定の方針を `nn3`（native avg score 116,205、`nn2` は 82,772）にし、自動再生の
速度を select で選べるようにした（当初は `max` / 250 ms / 500 ms / 1 s / 2 s、既定は `max`。
1 秒以上の待ちは使わないので #154 で 1 s / 2 s を外した）。

wasm（ReleaseFast、V8、このマシン）での 1 手あたりの実測。`nn3` のコストは盤面に強く依存する。
#143 の表の 22.9 ms は終盤の詰まった盤面だけを測った値で、序盤は 10 倍以上かかる:

| 局面 | 手数 | `nn2` | `nn3` |
| --- | ---: | ---: | ---: |
| 開始直後 | 0 | 14 ms | 270 ms |
| 256 タイル | 250 | 9 ms | 390 ms |
| 2048 タイル | 1,000 | 9 ms | 400 ms |
| 4096 タイル | 2,500 | — | 4 ms |
| 4096 タイル（約 3,500 手） | 3,500 | — | 80 ms |

序盤は空きマスが多く分岐が最大になるので 0.3〜0.4 s、盤面が詰まると 10 ms 前後まで落ちる。
1 ゲーム 4,000 手前後なので、`nn3` の自動再生は数分〜十数分かかる。

- 速度は「1 手あたりの待ち時間」。`max` は従来どおり 1 フレーム 8 ms の予算で回すが、`nn3` は
  1 手でその予算を超えるので実質 1 フレーム 1 手になる。
- 待ち時間は *solve が返ってから* 測り直す。solve 自体が 0.3 s かかる手に 250 ms を上乗せして
  2 倍遅くしてしまうのを避けるため。
- `nn3` の深さは `nn.bestMove(net, state, .deep3)` で固定なので、`solve` に渡す depth は
  フォールバック（`expectimax`）のためのもの。
- 自動再生のループは 1 本だけにする。`stopAuto` が保留中の `requestAnimationFrame` を
  `cancelAnimationFrame` するので、停止→開始が 1 タスク内で起きても 2 本目が生まれない
  （nn3 の solve が 0.3 s 固まる間に押されると起きていた）。
- 速度の `<select>` にフォーカスがあるときは矢印キーを盤面に使わない。以前は select の
  上下キーが `preventDefault` されて手になってしまっていた（サーバー版の `src/web/index.html` も同じ）。
- `scripts/wasm-smoke.mjs` は「静的版の方針が wasm に載っている」ことを検査する。当時は
  ページ側の定数名 `SOLVER_NAME` を見ていたが、#154 でページは方針名をモジュールから取るように
  なり、この検査は同梱一覧と既定（`DEFAULT_POLICY`）の突き合わせに置き換わった。

## 同梱する方針を設定ファイルで決め、nn2 / nn3 の 2 択にした（#154）

#143 で 13 方針のドロップダウンを外したが、`#152` で `nn3` を既定にすると「速すぎて何が
起きているか分からない」局面が出たので、方針の select を戻した。ただし 13 方針を全部載せる
理由は無いので、**同梱する方針はコードではなく設定ファイル** `wasm.policies` に持たせた
（1 行 1 名、`#` 以降はコメント）。既定は `nn2` と `nn3`。

- `build.zig` が `@embedFile("wasm.policies")` でそのまま `options` モジュールに渡し、
  `src/wasm.zig` が comptime で解析する。未知の名前・重複・n-tuple 表が要る `ntuple` は
  ビルドエラー。ファイルを編集すると wasm が再ビルドされる。
- 同梱した方針だけをリンクするために、`solver.choose` を `chooseOne(comptime policy, ...)` に
  分け、`solve` は `inline 0...policies.len - 1` でインデックスごとに定数方針を渡す。
  実行時の `switch (policy)`（= 従来の `choose`）は全アームをリンクしてしまう。
- ABI の `algo` は「モジュールに入れた順」の index になった。`Policy` の並びは ABI ではなく
  なり、`wasm_policy_count`（"表が要る方針は最後" を強制する仕掛け）は消した。
- ページの方針 select は `algo_count()` / `algo_name()` から組み立て、選んだ方針の**測った
  平均スコアと 1 手の時間**を `algo_score()` / `algo_speed()`（`policy_info` の `score` と
  `speed`）で select の下に出す。実装の説明（`description`）はページには長すぎるので出さない
  （#155 の指摘）。`speed` はブラウザ実測（V8 on wasm）で、native の ms/move とは別物なので
  測っていない方針は空にし、ページはスコアだけを出す。ページが自分で持つ方針名は
  既定を指す `DEFAULT_POLICY = "nn3"` の 1 つだけで、同梱していない方針は選べないし、UI と
  モジュールの一覧はずれない（既定が同梱に無いときだけ最後のエントリに落ちる）。
- `scripts/wasm-smoke.mjs` は `wasm.policies` を読み、モジュールの一覧が名前も順序も一致する
  こと、各エントリに説明があること、ビルド済み HTML が `algo_count()` で一覧を作っていること、
  その `DEFAULT_POLICY` が同梱一覧に含まれることを検査する（後者が無いと、既定の方針を
  `wasm.policies` から外したときにページが黙って別の方針になる）。`scripts/deploy-pages.sh` は
  デプロイログに同梱方針を出す（インラインコメントは落とす。`--skip-smoke` のときは
  「モジュールと設定ファイルの突き合わせをしていない」と警告する）。

サイズの実測（ReleaseFast、`zig build wasm`、`wc -c`）:

| 同梱した方針 | バイト |
| --- | ---: |
| 13 方針すべて（#152 まで） | 764,897 |
| `nn2` + `nn3`（既定） | 701,728（説明文をページに出さなくして -298 B、`speed` で +142 B） |
| `nn3` のみ | 701,167 |
| `nn2` のみ | 701,107 |
| NN 以外の 8 方針 | 591,768 |
| `random` のみ（床） | 529,463 |

- 13 → 2 で 63 KB（8%）しか減らない。`nn2` と `nn3` は同じ net と探索コードを共有していて、
  2 つ目は 700 B しか増えないため。方針を削る効果はここで頭打ちになる。
### レビューで見つかって直したこと（#154）

- `solve(algo, depth)` の `depth` を捨てていた。同梱が `nn2`/`nn3` の間は害が無い（両方とも
  深さ固定）が、この機能の売り文句どおり `wasm.policies` に `expectimax` か `ga` を足すと
  `depth = 0` が `chanceValueW` で `0 - 1` の u8 ラップで 255 手になり、タブも smoke テストも
  戻ってこない。`depth` を通し、`0` は `policy_info` の既定深さ（`expectimax`/`ga` は 2）を
  使う、という `expectimax_id` と同じ約束にした。`expectimax` を足したビルドで
  `solve(0, 0)` が 1 手で返ることを確認済み。
- パーサが `nn2#コメント` の名前を捨てていた（`#` の分岐が区切りの処理より先にあった）。
  区切りとコメントの前で必ず名前を確定させ、先頭 BOM も読み飛ばし、`found[32]` の上限も
  `@compileError` にした。
- smoke テストの 2 s 上限は呼び出しが返ってからしか見ないので、ハングを検出できなかった
  （#143/#145 の保護は実は「depth 引数」だった）。方針ごとに `--solve-one` の子プロセスで
  回し、5 s で `spawnSync` が kill するようにした。
- `src/wasm.zig` の `depth` ABI 説明と、`expectimax_id` の説明（「既定の40ms」→ 実際は
  `id_budget_ms`）が実装とずれていたので直した。

- 内訳（既定ビルド）: code 11,787 B / data 689,801 B。data の大半は `src/core.zig` の
  `row_table`（`[65536]RowEntry` = 512 KiB、1 行のマージを引く表）で、NN 重み 148 KB よりも
  大きい。次にサイズを削るならここ（`RowEntry` は 8 B。score を半分にして u32 に詰めれば
  256 KiB 減る）で、方針を削っても効かない。

## 2-ply の目標と自己対戦方策を評価側に揃える（#158）

評価は 2-ply の探索値で手を選ぶのに、学習は 1-ply greedy の自己対戦と `reward + V(next afterstate)`
で回っていた。価値関数が「今の方策の先読み」に寄るほど探索の上乗せが消える（#135 で fine tuning
すると 1-ply は +24%、2-ply は −8%）ので、学習の両側を評価側に寄せる 4 セルを用意した。

`zig build nn -Doptimize=ReleaseFast -- 100000 10 9000 0.003 10 {t,tp,p,tp2} 30 <tmp> 3`

| セル | 目標 | 自己対戦 |
| --- | --- | --- |
| `base`（対照） | `td1`: `reward + V(next afterstate)` | `greedy1`: 1-ply argmax |
| `t` 目標のみ | `search2`: 2-ply 探索値 | `greedy1` |
| `p` 方策のみ | `td1` | `deep2`: 評価の `.deep` で手を選ぶ |
| `tp` 両方 | `search2` | `deep2` |

いずれのセルも spawn を列挙するので乱数の消費は `base` と同じで、ゲームの spawn 列が揃う。

### 目標の二重計上（レビューで見つけて直したこと）

最初の `search2` は `Search.value(spawned, 2, 1)`、つまり「報酬 2 つ + 葉の `V`」だった。`value` の
葉は**最後の報酬を足した後の afterstate** を評価するので、その `V` には次の報酬が既に入っている。
決定的な鎖（各手の報酬が 1、割引なし）で確かめると固定点は 1, 2, 4, 5, 7, 10 と真値の約 1.5 倍に
なっていた（`test "search2 target does not double count the last reward (#158)"` が差を 8/reward_scale
として固定している）。

`Search.bootValue(state, rewards, cprob)`（同 `bootChance`）を足し、葉を「最後の報酬より 1 手先の
afterstate」にずらした。`bootValue(S, 0)` は 1 手先 afterstate の最大 `V`、`bootValue(S, r)` は
`max_d (score_delta / reward_scale + bootChance(S', r - 1))` なので、`search2Value` は
`r_t + r_{t+1} + V(A_{t+2})` を返す。`Search.ply` を `search2_ply + 1` にして置換表のキーも葉の
深さに合わせた（値自体は深さ非依存）。

## 深さを空き数で切り替える（#157）の実測

`deep`（1 通り 2-ply）、`deep3`（3-ply, cutoff 1e-2）、`phase_h1/h2/h3`（空き数で 2/3-ply を混ぜる）
を同じ net・同じ 3 局（seed 9000-9002）で測った。native ReleaseFast、重みは同梱の
`ai/solver/nn_weights.bin`（md5 `f584d99667f499184d7da2d5d0e9e0a1`）。この net は md5 を控えてから
`/tmp` にコピーして使う（リポジトリのファイルを上書きしない・取り違えないため）。

`zig build nn -Doptimize=ReleaseFast -- 0 3 9000 0.003 1 base 30 /tmp/w.bin 1`

| 行 | avg score | moves | best | ms/move | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nn 1-ply | 60,980 | 2,849 | 4096 | 0.00 | 2/3 | 0/3 |
| nn + 1 spawn lookahead | 64,479 | 3,051 | 4096 | 0.08 | 2/3 | 0/3 |
| nn 2-ply (deep) | 69,272 | 3,181 | 4096 | 0.72 | 3/3 | 0/3 |
| expectimax depth 2 | 23,335 | 1,275 | 2048 | 0.05 | 0/3 | 0/3 |
| nn 3-ply (deep3) | 100,728 | 4,354 | 8192 | 16.58 | 2/3 | **2/3** |
| phase h1 | 93,912 | 4,143 | 8192 | 21.44 | 3/3 | 1/3 |
| phase h2 | 110,843 | 4,862 | 8192 | 26.37 | 3/3 | 1/3 |
| phase h3 | 108,901 | 4,663 | 8192 | 29.58 | 3/3 | 2/3 |

**結論: 空き数で深さを切り替えてもコストは下がらない。** h1/h2/h3 は `deep3` より高い
（21-30 ms/move 対 16.6 ms/move）。理由は 1 局の手数の大半が空き 4-7 で消費されることで、
そこを 3-ply にすると 3-ply 一律と同じだけ読む。切り替えの副作用で序盤だけ 2-ply にした分は
安く済むが、全体では深い局面の 3-ply が支配する。

到達率では `deep3` と h3 が 8192 に 2/3 で届き、`deep` は 0/3、1-ply は 4096 に 2/3 止まり。
**n=3 なので avg score の差（h2 110,843 対 deep3 100,728）は判断材料にしない**。差が出ているのは
深さ（2-ply 対 3-ply）の方で、これは 8192 到達率としてはっきり出ている。

### 計測まわりの事故（同じ轍を踏まないための記録）

最初にこの表を取ったとき、作業ツリーの `ai/solver/nn_weights.bin` が #133 以前の 200k+40k net
（md5 `0a8e6bf01e256adf2fc481317c5072c9`）に入れ替わっていて、全行が 4-14 倍低い値になった
（`deep` 17,408、`deep3` 25,662、h3 22,773）。コードは壊れておらず、golden テスト
（`ai/solver/solver.zig:1865-1879`、`nn` seed 20240601 = 32,380 / `nn2` = 77,164）は同梱 net で緑、
古い net では赤になる。**評価を回す前に `md5` を控え、`/tmp` のコピーを渡す**こと。

## 終盤だけ 4-ply（#171）

探索の深さを空きマス数で選ぶ表（`ai/solver/nn.zig` の `phase_schedules`）に `phase_h4` を足した。
中盤は `deep3`（3-ply、cutoff 1e-2）のままで、**空きマスが 2 以下になってからだけ 4-ply** にする。
#157 の `phase_h1/2/3` は深さを切り替えて**総コストを節約する**方向だったが、こちらは逆向きで、
**終盤の数手にコストを積む**実験。

理由は 2 つ。終盤は盤面が詰まって分岐が潰れるので、1 ply 増やす絶対コストが小さい（終盤の 1 手は
4 ms 前後まで落ち、遅い局面でも ~80 ms。#152 の表）。そして 4096 → 8192 の判断はまさにその数手で
決まる。**#157 が否定したのは「深さの切り替えで総コストを下げる」ことであって、「終盤だけ深く
する」ことではない**ので、その結果とこの実験は矛盾しない。

| `min_empty` | ply | cutoff | 意味 |
| ---: | ---: | ---: | --- |
| 3 以上 | 3 | 1e-2 | `deep3` と同じ（控えめな基準線） |
| 0〜2 | 4 | 1e-2 | 空きが 2 以下なら 1 ply 追加 |

4-ply の行も cutoff は `deep3` と同じ 1e-2。空き 2 の盤面では spawn の分岐が数個しかなく、木は
もともと小さいので、`phase_h2`/`phase_h3` の 1e-4 まで絞る必要がない。

計測（3 局、seed 9000–9002）:

    zig build nn -Doptimize=ReleaseFast -- 0 3 9000 0.003 1 base 30 <tmp_weights> 1

| 方針 | avg score | moves | ms/move | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nn 3-ply (deep3, #108)`（基準線） | 100,728 | 4,354 | 21.42 | 2/3 | 2/3 |
| `nn phase h4 (#171)` | **131,201** | 5,569 | 31.69 | 3/3 | 2/3 |

`deep3` は同じ net・同じ seed で 16.58-21.42 ms/move（マシンの空き具合で振れる）なので、
**4-ply にした分の上乗せは約 +48%**。avg score は +30%、手数は +28%、4096 到達は 2/3 → 3/3。
8192 到達率は 2/3 のまま（n=3 では差が出ない）。**n=3 なので avg score の差は参考値**だが、
同じ 3 局で手数と 4096 到達が揃って上を向いているので、10 局に増やして判断する価値はある。

## 深さ別の比較（10 局, seed 9000-9009, 同梱 net）

`deep`（2-ply）/ `deep3`（3-ply）/ `phase_h4`（終盤 0-2 空きだけ 4-ply, `#171`）を 10 局で測った。
native ReleaseFast、重みは同梱 `ai/solver/nn_weights.bin`（md5 `f584d99667f499184d7da2d5d0e9e0a1`）を
`/tmp` にコピーして使う。

`zig build nn -Doptimize=ReleaseFast -- 0 10 9000 0.003 1 base 30 /tmp/w.bin 1`

| 方針 | avg score | moves | ms/move | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| nn 1-ply | 50,496 | 2,384 | 0.00 | 6/10 | 0/10 |
| nn 2-ply (deep) | 86,565 | 3,828 | 0.58 | **10/10** | 3/10 |
| nn 3-ply (deep3) | 106,936 | 4,567 | 12.04 | 8/10 | **6/10** |
| nn phase h4 (#171) | **117,566** | 5,016 | 22.88 | **10/10** | **6/10** |

読み方:

- **4096 は 2-ply で 10/10**。1 手 0.58 ms（native）で、ユーザの前提「nn2 で 4096 まではほぼ確実」は
  10 局では確定。3-ply の 8/10 は 10 局ではノイズの範囲（2 局が 4096 直後で死んでいる）。
- **8192 は 2-ply で 3/10、3-ply で 6/10**。`#171` の終盤 4-ply も 6/10 で、**8192 到達率は上がらない**。
  一方で avg score は `deep3` より +10%（106,936 → 117,566）、手数 +10%、コスト 1.9 倍。
  「終盤を深く読むと、8192 に届くかどうかは変わらないが、届いたときの伸びが良くなる」。
- 8192 到達率を動かすには、**探索の深さではなく評価関数そのもの**を良くする必要がある（この表の
  3 段の差はすべて V の質の差であって、探索の差ではない）。`#167`/`#175` の蒸留が本命。
