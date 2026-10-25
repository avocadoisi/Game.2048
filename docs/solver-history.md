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
