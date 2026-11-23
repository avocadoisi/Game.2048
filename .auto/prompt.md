# Autoresearch: raise the 2048 solver's average score (no local-optimum overfit)

## Objective
Maximize the 2-ply (`deep`) average score of the shipped value net on held-out seeds, then
adopt only nets that also hold up on a second, untouched seed set. The project's shipped
baseline is nn2 ≈ 113.7k (seed 9000) and 129.5k (seed 9100) at ~2 ms/move native.

## Metrics
- **Primary**: avg2ply (points, higher better) — 20 games at seed 9200, `resume` 128/32
- **Secondary**: avg1ply, reach8192 out of 20, ms/move, moves/game
- Confirmation (only for adoption candidates): SEED=9300 and SEED=9400.

## How to Run
`./.auto/measure.sh <net.bin>` — outputs `METRIC name=number` lines.
Training (expspawn, 128/32, lr 0.001, par, batch_scale 1 because the variant guard requires it):
`zig build nn -Doptimize=ReleaseFast -- <games> 0 9200 0.001 1 expspawn 30 <out.bin> <train_seed> par - -`

## Files in Scope
- ai/solver/nn.zig — value net, TD variants, expspawn implementation
- ai/solver/nn_main.zig — CLI, variants, validation, training loop
- ai/solver/solver.zig — search/eval paths (only if a measured win requires it)
- ai/solver/nn_weights.bin — shipped weights (never overwrite; candidates in /tmp)
- .auto/* — session files

## Off Limits
- Nothing may break `zig build test` / `scripts/gate.sh`.
- Do not train or tune on eval seeds 9200, 9300, 9400 or on seed 9000/9100.
- Do not overwrite ai/solver/nn_weights.bin until a candidate passes the adoption test.

## Constraints
- 128/32 architecture and the existing file format unless an experiment proves otherwise.
- Use `par` (8 threads) for training; batch_scale 1 with expspawn (guard), sqrt for base.
- Project budget: one training attempt ≤ 4 h (14,400 s); 60 s/game soft limit.

## What's Been Tried
- Shipped net: base TD(0), 300k games, lr 0.003, par sqrt. nn2 113.7k/129.5k on seeds 9000/9100.
- `expspawn` (Rao-Blackwellized spawn expectation) at 20k games: 61.6k vs base 30.7k — 2x better
  at equal games, but 13.2x slower per move; at 20k still only 54% of shipped. Retraining the
  shipped net with expspawn was deferred (estimated 12 h). This session tests whether the
  current machine can close that gap in a reasonable budget.
- Everything else in docs/measurement.md was rejected with numbers (resume, distillation, PopArt,
  sym/rsym/rtgt, wider nets, features, Adam, explore variants). See that file before retrying.

## Session Notes
- Machine: 8 physical cores; expspawn measured at ~20.6k moves/s wall (10k games, 145 s).
- Eval cost: 10 games 2-ply ≈ 64 s; shipped net seed 9200 10 games = 131,709.6, 8192 7/10.
