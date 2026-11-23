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
