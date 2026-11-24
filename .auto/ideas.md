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
- expspawn 300k at lr 0.001 is ~5 h on this machine (60k games in 22 min, games ~500 moves),
  and the mid-run 50k checkpoint (lr 0.00085) scores only 44,141.8 (1-ply 15,133.6): the
  slowly-annealed target is smoothed. Killed at 60k. Retry with a *short* schedule instead
  (40k games, 0.001->0.0001) — matches the documented 20k point (61.6k) and anneals fully.

## Open after session 2
- Make the 3-ply tree cheaper before retesting h6: 4-way set-associative node TT and a
  leaf-only smaller net (h1 96) could cut leaves 30-50%; h6 needs ~4x -> ~2x to be arguable.
- Risk/tile-conditioned phase rows (`min_max_tile`) so only 4096+ boards pay for 3-ply.
- `pairw` the shipped net against a 300k retrain under a *changed* target is now cheap and
  deterministic; any target change can be A/B'd exactly.
- 2-net value ensemble (#180, backlog says "reserved for compute"): train a second 300k net with a
  different train seed and average V at the leaf. Mechanism: halves the random part of V error, so
  near-tie moves get sharper; docs' V error is 0.07 early / 0.30 late. Cost: 2x leaves (ms/move
  ~2.4), needs a `Value.pair` variant plus a two-path CLI. Only worth it if the score gain clears
  the noise at 2x latency.
- Polish-length sweep: TRIED (#21). 15k beat 30k on 9000/9300 (+10,989/+1,140) and lost on
  9200/9400 (-9,482/-13,404); four-seed aggregate favors 30k. 20 games per seed cannot rank nets -
  any future length point needs 4 seeds x 40 games (~1.5 h each). Do not repeat at 20 games.
