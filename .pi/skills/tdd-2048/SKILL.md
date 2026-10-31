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
- one heavy run (`zig build test`, bench, training) at a time, even across subagents — parallel training jobs stall each other (load average is not throughput)
- `zig build test -Doptimize=ReleaseFast` does not finish: the `nn` test binary alone compiles for 20+ min. Debug is the full-test setting
- measure a policy with `zig build solve`; `zig build ranking` runs every policy (nnmcts etc.) and takes 10+ min per game
- `std.testing`: `expectEqual` / `expectEqualStrings` / `expectError`
- core API names match README "Core API"
