# Ideas backlog (pruned 2026-11-25)

Everything cheap is measured and closed; see `.auto/prompt.md` "Update 2026-11-25".
The list below is only what still has a mechanism behind it.

## Needs a user decision (design work, not an experiment loop step)
- **Representation change**: a small CNN (2x2 conv over the 4x4 one-hot, 32-64 channels) or a
  residual MLP. Params ~40-80k, needs a new weight magic, a wasm build, a f32 converter path and a
  full retrain (300k base + 60k polish, ~1.5 h at h1=128 scale). Prior: docs' `downmix` best 90,792
  (-14%), `sym` collapsed — but neither was a learned conv. Expected +0-8%, unknown.
- **Product time budget**: endgame 4-ply measured 10.03 ms/move with 12/12 8192 (12 games, 9200)
  and endgame 3-ply 3.2-3.6x; only worth shipping if the app accepts ~3-10 ms/move in the browser.

## Cheap leftovers, low prior
- `clamp` (target bound) on the base 300k recipe: only a divergence guard; seed 2/3 failures are
  instability, not score. No score mechanism.
- Rotation-canonical eval (rotate the board so the max tile sits in a fixed corner, rotate the move
  back): no format change, ~free; the net is near-equivariant, so expected ~0. Untested.
- 60-game pairw (needs 2 sigma at the docs' bar; 40 games already gives 2.38 sigma, so only for a
  formal record).
- `-Dfeat` at 300k: closed at 100k (feat helps at 20k, gone by 100k; `par`+feat refused).
