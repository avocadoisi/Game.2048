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

## Update 2026-11-24 (session 2)
- Baseline avg2ply (seed 9200, 20 games): 134,867.2 for the shipped net.
- lr 0.002 / 250k, seed-2 divergence, fresh 300k rerun (116.9k), SWA 4/8/16/32: all discarded.
- Endgame 3-ply rows h5/h6/h7/h9: h6 looked +12.6k on 9200 but did not replicate
  (9300 +10.8k ± 10.1k, 9400 +0.7k) at 4x ms/move. Discarded (log #6). 20-game seed
  noise is ~±10k; a single-seed +10k is not evidence.
- expspawn is affordable after all: `expspawn` + `par` + `batch_scale 1.0` is accepted
  (the guard rejects only non-1 batch_scale). 300k games ≈ 1.5-2 h on 8 cores.
  Run in flight: `/tmp/expspawn300k.bin`, lr 0.001, seed 1.

## Update 2026-11-24 (session 2, final state)
- REPRODUCIBILITY: the shipped recipe is deterministic. `base`, 128/32, 300k games, lr 0.003,
  seed 1, `par`, `batch_scale 2.8284271247461903` rebuilds ai/solver/nn_weights.bin bit-for-bit
  (sha256 c98ea27c...). Eval on seed 9200 is exactly 134,867.2 / 1-ply 60,864.6 / 15-20 8192.
- TRAP: `-` in the batch_scale slot means 1.0, not sqrt(batch) (`parseBatchScale` returns 1.0 for
  `-`). Every earlier "variance" and SWA conclusion came from 8x undertrained bs1 runs.
- SEARCH: endgame 3-ply (`phase_h6`) is a wash at 4x cost:
  9200 +12,575.6 (unpaired), 9300 +10,801.8 +/- 10,050.1, 9400 +747.6, 9000 +2,708.0 +/- 8,531.5
  (paired A=141,548.0, B=144,256.0, 10W/10L). Reverted. Do not retry without a cheaper tree.
- EXPSPAWN: closed. par+bs1 is 8x undertrained; par+sqrt(8) reaches 92,377 at 190k (behind base
  at the same budget); serial is ~1.7 h per 40k games.
- NO measured lever above 134,867.2 exists in this repo today.
- EFFICIENCY: native node-TT 4096 -> 16384 entries (ai/solver/nn.zig:759): 3-ply score identical
  (146,470.7) but leaves/move 10,251 -> 9,581 (-6.5%) and 17.42 -> 16.57 ms/move (-4.9%),
  hit rate 17.8%. 2-ply untouched (134,867.2, 1.00 ms/move). Hit rate is now collision-bound;
  4-way associativity is the next step if 3-ply ever ships.
- TILE-GATED DEPTH (h5): 3-ply only on 4096+ boards at <=3 empties scores 119,494.6 (12/20 8192,
  3.22 ms/move) on 9200 - worse than 2-ply and than ungated h6. Do not gate depth on progress.
- 4-WAY TT: 4-way set association at 16k entries is 0.6% slower than direct-mapped with a lower
  hit rate (16.6% vs 17.8%); direct-mapped 16384 is the best TT. 3-ply cost is tree-bound.
- SCHEDULE STRETCH (400k, lr 0.003 -> 0.0003 linear): 85,445.2 on 9200 (3/20 8192,
  1-ply 49,102.8) vs 134,867.2 for the 300k recipe. The anneal must be the last ~10% in absolute
  games; stretching it proportionally leaves the net hot. Same failure as the 1M run and #204 C.
- LR SWEEP: 0.002 (250k) 101,526.4; 0.0028 (300k, clean) 113,191.8; 0.003 (300k, shipped)
  134,867.2. 0.003 is the measured peak of the recipe; downward is strictly worse and upward
  diverges on seed 2. No free parameter remains.
- SEED SPREAD AT 300k (lr 0.003, par sqrt, all deterministic): seed 1 = 134,867.2 (shipped),
  seed 2 = divergence at 50,440 games, seed 3 = collapse to 1,160 moves/game and 23,119.2
  (0/20 4096). The recipe is a lottery (~1/3 success); the shipped net is a high draw.
  Any retrain must budget 3-4 seeds and gate on moves/game + 4096 reach, never on benchmark score.
- RECIPE KNOBS ARE EXHAUSTED: variant `base` ignores the CLI replay_k slot (`buildBase` returns
  `{}`); replay_k=0 vs 30 yields byte-identical weights. Active inputs are only (games, lr, train
  seed, par batch_scale), all pinned: lr 0.003 peak (0.0028=113k, 0.002=101k), 300k the end of the
  anneal (400k=85k), sqrt(8) the par rule, seed 1 the only non-collapsing seed at 300k.
## Current state (2026-11-24)
- Baseline is now the 30k-polished net: 9200 144,437.4 (18/20), 9000 136,147.2, 9300 129,883.8,
  40 games 9000 135,032.0; 1.07-1.10 ms/move. Shipped in 017da7f (goldens + f32 regenerated).
- Its recipe: `resume` 30k games from the previous weights at constant lr 0.0003, seed 1, par sqrt.
  Seed-4 control of the same polish was neutral (-809), so the gain belongs to this trajectory.
- Next candidates: polish-length sweep (15k/60k), a second sequential polish, and confirming the
  polish on another base net. Judge on seeds 9000/9300 (9200 is the benchmark and must not be the
  only selector).
- NOISE LESSON (#21): a 20-game per-seed comparison flips sign between seeds. The 15k polish won
  by +12k on the two selection seeds and lost by -9k and -13k on the two held-out ones. Rank nets on
  >=4 seeds (or 40 games/seed, ~9 min each) and never on 9200 alone.
- POLISH SERIES (current shipped = 60k, commit 996140a): 0k 113.7/134.9/125.3/133.7 (9000/9200/9300
  /9400), 30k 136.1/144.4/129.9/134.8 (agg 545,257), 60k 135.4/150.7/144.8/153.7 (agg 584,697).
  Each pass = `resume` 30k at constant lr 0.0003, seed 1, par sqrt (~570 s). Rule for the next pass:
  keep only if the four-seed aggregate beats 584,697.0 with >=3 of 4 seeds non-worse.
- POLISH SERIES CLOSED at 60k (#23): 0k ~500,790 / 30k 545,257 / 60k 584,697 / 90k 578,303
  (four-seed sums). 90k trades seeds (9000 +5,135, 9300 +1,525, 9200 -3,347, 9400 -9,707).
  Do not extend the constant-lr polish further; a lower-lr anneal tail (1e-4) is a different,
  untested mechanism.
- ANNEAL TAIL REJECTED (#24): 10k games at lr 1e-4 after the 60k polish gives aggregate +6,504 but
  9200 -8,287 (-5.5%) and 9300 -3,598 (-2.5%); 2 of 4 seeds non-worse. Seed-trading, outside the 3%
  band. Lesson: a positive four-seed SUM can hide a benchmark regression - check the per-seed band.
- OUT-OF-SAMPLE CONFIRMATION (#25): on seeds 9500/9600 (never used in any decision) the 60k net
  scores 155,944.2 / 151,678.0 vs 118,945.0 / 122,604.4 for the pre-polish 300k net (+31%/+24%,
  8192 19/20 both). Six-seed mean 124.9k -> 148.7k. The polish gain generalizes.

## Update 2026-11-25 (session 2, converged at 40+ experiments)

Shipped net: 128/32, base 300k (lr 3e-3, seed 1, par sqrt(8)) + polish 60k (lr 3e-4, seed 1).
Benchmark 150,747.2 (9200, 20 games); four-seed sum 584,697.0 (135,419/150,747/144,828/153,702);
nine-seed mean 148,528.9; 9500/9600 out-of-sample 155,944.2/151,678.0. Commits: 996140a (60k),
017da7f (30k), 964c533 (9 seeds), 3721966, c404e0f (selection-effect rule), 3f5d5cd, fff9902,
9449206 (leaky knob).
- DETERMINISM: base 300k reproduces bit-identically (sha256 c98ea27c…, 553,202,634 moves) three
  times; training variance is exactly zero, all spread is train-seed choice.
- BATCH_SCALE TRAP: `-` means 1.0, not sqrt(8). Every early `par - -` run was 8x undertrained.
- SELECTION EFFECT (#32): the seed-2 polish (four-seed 609,407, benchmark 155,704.2) lost on both
  fresh seeds (152,174.6/147,647.6 vs 155,944.2/151,678.0). Rule now mandatory: four-seed screen,
  then two fresh seeds both >= shipped. Three polish samples: 584,697/609,407/550,843, sd 29,300.
- CLOSED WITH NUMBERS: 3-ply at <=3 empties (4 seeds, mean +3,220, every estimate <1.1 sigma at
  3.2–3.6x); endgame 4-ply (9x, +1.2%); tile-gated depth (119,494.6, horizon switch destroys the
  policy); node-TT 4-way (16.67 vs 16.57 ms); 400k straight decay (85,445.2); lr 2.8e-3 (113,191.8)
  and 4e-4 polish (143,142.4); third train seed (23,119.2) and seed-2 divergence at 50,440; weight
  merge (569,744.8); freeze/head-only (102,012.0); h1=160 (361,618, -38%); replay_k inert;
  leaky-ReLU 0.01 (376,386, -36%); p2 2-ply self-play fine-tune cost-closed (~3.6 s/game).
- ADOPTION EVIDENCE: 40-game pairw on seed 9000 gives mean diff -22,510.5 +- 9,470.6 (2.38 sigma),
  27:13, at 91448cd. Product f32 path: 154,007.4, 19/20 8192; f32 artifact byte-identical to a
  regeneration; wasm advertises 141,885 (nn2) / 152,463 (nn3).
- WHAT IS LEFT: only a different representation (CNN/residual; weight-format change, full retrain)
  or a product per-move time budget. Do not re-run recipe variants: every cheap axis is measured.
