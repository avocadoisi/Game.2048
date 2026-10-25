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
- Training: 20,000 self-play games, learning rate 0.003 (default in `nn_main.zig`). Weights go to `ai/solver/nn_weights.bin` (git-ignored).
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

## Training focus: near-loss states (proposal, not implemented)

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
