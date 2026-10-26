---
name: tdd-2048
description: TDD workflow for Game.2048 (Zig). Use when adding or changing 2048.core logic (move/merge/spawn/game over), CLI, or solver code in this repo.
---

# TDD for Game.2048

Project: Zig 2048. Pure logic lives in `src/core.zig` (module `core`). `src/main.zig` is a thin CLI. Keep logic in core; keep app layers thin.

## Commands

- Test everything: `zig build test` (runs the unit tests of 11 targets: core, exe, solver, bench, parallel, nn, nn_main, ntuple, ntuple_main, ga-rl, ranking)
- Run CLI: `zig build run` (play); `zig build run -- serve 8080` (HTTP server)
- Files: `src/core.zig` (logic), `src/render.zig`, `src/cli.zig`, `src/server.zig` (+ `src/web/index.html`), `src/main.zig` (dispatch)
- Zig version: **0.17** (`build.zig` uses `addPassthruArgs`; `std.Io` APIs). Check with `zig version` before building.
- Enter toolchain: `nix develop` (if zig is not on PATH)

Workflow for any change (worktree, branch, docs, PR): see `.pi/skills/feature-pr-workflow/SKILL.md`.

## Cycle (strict)

1. **Red**: write one failing `test "..."` next to the code in `src/core.zig` describing one behavior. Run `zig build test` and confirm it fails for the expected reason (not a compile typo).
2. **Green**: write the smallest code that passes. No extra options, abstractions, or features.
3. **Refactor**: clean names and structure with all tests green. Re-run `zig build test` after each step.

One behavior per cycle. Commit or report after each green step when useful.

## What must be covered by tests

- Line merge (`mergeLineLeft` / `mergeLine`): slide, merge each pair once per move, no re-merge of a new tile.
- Direction moves (`moveUp/Left/Down/Right`): implemented by rotating/extracting lines and reusing the left-merge; test each direction.
- Spawn: only when the board changed; uses the injected rng; placed on an empty cell.
- Unchanged board: no spawn, no state change.
- Determinism: fixed seed gives the same result; use the test RNG helpers (`SequenceRng`, `NoCallRng`) in `core.zig`.
- Game over: `isGameOver` true only when no empty cell and no equal neighbors.

## Rules

- Never write production code without a failing test first (exception: pure refactors with existing green tests).
- Inject RNG; never call a global random source from core.
- Keep test names describing the behavior, e.g. `test "moveLeft does not spawn on unchanged board"`.
- Do not change a test to make it pass; change the code, unless the spec itself was wrong, and say so.
- Core must not depend on CLI, IO, or allocator-heavy code.
- Test names describe behavior. A bug fix starts with a test that reproduces it.
- Each non-trivial helper (merge, spawn, route, parse) gets its own test; prefer `std.testing` `expectEqual` / `expectEqualStrings` / `expectError`.
- Tests that need randomness use `SequenceRng` (scripted) or a fixed-seed `DefaultPrng`.

## Done checklist

- [ ] New test failed before the change
- [ ] `zig build test` passes
- [ ] No unused code or speculative options
- [ ] Docs updated (see feature-pr-workflow step 5)
- [ ] Core API names match README "Core API" section
