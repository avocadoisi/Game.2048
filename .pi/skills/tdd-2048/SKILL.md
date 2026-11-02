---
name: tdd-2048
description: TDD workflow for Game.2048 (Zig). Use when adding or changing 2048.core logic (move/merge/spawn/game over), CLI, or solver code in this repo.
---

# TDD for Game.2048

Pure logic in `src/core.zig` (module `core`). `src/main.zig` is a thin CLI. Logic in core, app layers thin.
Cycle and test discipline: `dev-philosophy`. Order of a change: `dev-workflow`.

## Commands

- `zig build test` — unit tests of 15 targets: core, exe, solver, bench, ranking, parallel, evalab, probe, nn, nn_main, ntuple, ntuple_main, ga-rl, tune, cmaes
- `zig build run` — play · `zig build run -- serve 8080` — HTTP server
- Zig **0.17** (`build.zig` uses `addPassthruArgs`; `std.Io`). `zig version` before building
- `nix develop` when zig is not on PATH

## Files

`src/core.zig` logic · `src/render.zig` · `src/cli.zig` · `src/server.zig` (+ `src/web/index.html`) · `src/main.zig` dispatch

## Must be covered

- `mergeLineLeft` / `mergeLine`: slide; each pair merges once per move; a new tile is not re-merged
- `moveUp/Left/Down/Right`: rotate/extract lines, reuse the left-merge — each direction tested
- spawn: only when the board changed; injected rng; empty cell
- unchanged board: no spawn, no state change
- determinism: fixed seed → same result (`SequenceRng`, `NoCallRng` in `core.zig`)
- `isGameOver`: true only with no empty cell and no equal neighbours

## Local rules

- inject RNG; core never calls a global random source
- core does not depend on CLI, IO, or allocator-heavy code
- randomness in tests: `SequenceRng` (scripted) or a fixed-seed `DefaultPrng`
- `zig build test -Doptimize=ReleaseFast` does not finish: the `nn` test binary alone compiles for 20+ min. Debug is the full-test setting
- measure a policy with `zig build solve`; `zig build ranking` runs every policy (nnmcts etc.) and takes 10+ min per game

## Parallel implementation, serial measurement

Writing code is parallelisable. Measuring is not.

**Training throughput (measured, #208).** Training itself is single-threaded, so a plain
`zig build nn` uses ~1/8 of the machine. The `par` flag (argument 10) batches min(cpus, 8) games
through `nn.trainBatchOpts`: 20,000 games from `base` (lr 0.003, seed 1) take 112 s serial and 32 s
with `par` (3.6x): `zig build nn -Doptimize=ReleaseFast -- 5000 20 9000 0.0003 1 resume 30 /tmp/w.bin 1 par`.
A batch averages the 8 per-game deltas into one step, so `batch_scale` (arg 12) decides the step size:
the default is now the batch size (linear scaling rule, #208); `-` keeps the old mean (= 1/8 the
updates). Same-seed nets are NOT equal — compare nets on shared spawns with `pairw`, never by hash.
`par` is refused for `mstd`, the #158 2-ply variants, `dist`/`dson`, `dump`/`ft2`/`restart` and the
priority buffer; `replay_k` does work under `par` (each game owns its buffer).
Always train with `replay_k=30` and `par`. Fine-tuning without the replay buffer is what destroys the
shipped net (5,000 games at lr 3e-5: 73,100 without replay vs 93,318 with `replay_k=30`, baseline
79,657 on 20 games) — never fine-tune without it. `eval_games = 0` means "train only" and skips the
15-row eval table (which can cost 20+ min on its own).

**Training is serial, always — one `zig build nn` at a time, even when the user asks for parallel work.**
Only implementation, review and exploration may run in parallel. Measured: five `dson` lanes started
together ran at 1.0-1.35 ms/move each, the same run alone is 0.5 ms/move; the wall clock and the
`ms/move` columns both become meaningless. If the CPU looks idle, add *code* work, not another training.

- parallel OK: several issues at once, each in its own worktree (`worktree: true`), one writer per worktree
- a subagent may run its own **filtered** Debug tests (`zig build test -Dtest-filter <name>`) and 1-2 game smoke runs
- serial, one at a time, parent-owned: full `zig build test`, `-Doptimize=ReleaseFast` builds of `nn`/`ranking`, any training run (`zig build nn -- <games> > 0`), `zig build ranking`, `zig build solve`, `zig build wasm`
- never let two trainings/benches run concurrently, not even across subagents: they stall each other and the wall-clock numbers in `docs/solver-history.md` stop meaning anything (load average is not throughput)
- so: brief every worker with "no long runs", and run the measurements yourself, serially, after the branches come back
- `std.testing`: `expectEqual` / `expectEqualStrings` / `expectError`
- core API names match README "Core API"
