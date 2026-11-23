# Ideas backlog

- **K-sampled expspawn** (`expected_spawn_branches`): average bestAfterValue over K uniformly sampled
  empty cells × both tile values instead of all 2E branches. Cost K/15 of full expspawn; unbiased,
  removes tile variance exactly and cuts cell variance by K. Needs an RNG or hash-based cell pick.
  Test K=2,4 at 20k serial lr 0.001 and compare per wall-clock second against base scaling.
- **Parallel expspawn with per-game nets + periodic weight averaging** (model averaging): 8 private
  nets, each trained serially at lr 0.001 (the stable regime), averaged every N games. Avoids `par`'s
  per-batch sum step that diverges at lr 0.008. Risk: averaging drift; test 8×2k games.
- **Batched forward for the 30-120 expspawn branch boards** (one 120×256×128 matmul instead of 120
  forward calls) to amortize the 32 KB weight scan. Estimated 1.5-2x, not 8x (already compute-bound).
- **LR annealing after the 1M base run**: resume the 1M net at lr 5e-4 / 1e-4 for 100k, eval. Cheap.
- **Base scaling beyond 1M** (2M / 3M games) if the 1M curve is still rising and time allows.
- **128/32 with h2=16**: 8% fewer params, slightly faster forward; quality unknown, no score lever.

## Search ideas (after endgame 3-ply tests, 2026-11-23)
- Node-TT hit rate is only ~16% in the 3-ply endgame; a 4-way set-associative / 4x larger
  table (768 KiB) could cut leaves ~20-30% for free. Speed only, no score change.
- Risk-conditioned depth: `Phase` has min_empty/max_empty only; add `min_max_tile` so the
  3-ply tree runs only when a 4096+ tile exists (the tail games). Cost drops on early boards,
  score should track the tail. Needs a code change + 20-game eval.
- Parallel root moves (<=4 threads) on the endgame: ~3x lower wall time, same CPU. Only helps
  the app's perceived latency, not the benchmark's ms/move.
- Dedicated low-empty value net (smaller/faster, trained on deep3 targets at <=4 empties):
  distillation from deep3 failed at #350, so only worth it with a much larger sample.
- 20-game evals on one seed have a noise floor of ~+/-10k; confirm any search change on 3 seeds
  or with pairrow before believing a +10k win.
- After expspawn 300k: fine-tune it with the *plain* TD(0) recipe for 50k at lr 0.0005
  (target mix: expected-spawn bootstrap -> sampled) to recover any 2-ply sharpness lost to
  the Rao-Blackwellized target; only if expspawn alone is close to 134.9k.
- Compare expspawn's 1-ply (greedy) score too: if it is much higher than 60.9k but 2-ply is
  not, the target is smoothing V in a way 2-ply punishes (same signature as low-lr fine-tune).
