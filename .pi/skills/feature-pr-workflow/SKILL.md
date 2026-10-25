---
name: feature-pr-workflow
description: Standard flow for every change in Game.2048. Use when starting any feature, fix, or refactor. Order is prompt -> git worktree -> new branch -> TDD -> fix docs -> PR.
---

# Feature PR workflow (Game.2048)

Every change follows these six steps in order. Do not skip a step. Do not work on `main` directly.

## 1. Prompt (capture the request)

- Restate the goal in one or two lines and list acceptance criteria.
- If the request is ambiguous, ask before creating anything.
- Write the list of steps with the todo tool.

## 2. Git worktree (isolated checkout)

Keep `main` clean and run each change in its own worktree next to the repo.

```bash
cd <repo>
git fetch origin
git worktree add -b <branch> ../Game.2048-<short-name> origin/main
cd ../Game.2048-<short-name>
```

- Worktree directory: `../Game.2048-<short-name>`.
- If the work already happened in `main`, move it with `git stash push -u` and `git stash pop` inside the new worktree. Do not leave changes in `main`.
- One writer per worktree. Parallel work gets its own worktree.

## 3. New branch

- Name: `feat/<topic>`, `fix/<topic>`, or `refactor/<topic>`, lowercase with hyphens.
- Created by `git worktree add -b` above. Branch from `origin/main` unless the user says otherwise.

## 4. TDD

Follow the `tdd-2048` skill (`.pi/skills/tdd-2048/SKILL.md`).

- Red: write a failing test first, and see it fail for the expected reason.
- Green: smallest change that passes.
- Refactor: only with green tests.
- Run `zig build test` with Zig **0.15.2** (see the Zig note in README). Zig 0.17 breaks `build.zig`.
- Bug fixes: first a test that reproduces the bug, then the fix.
- Commit in small steps: one commit per Red/Green/Refactor cycle or per logical unit.

## 5. Fix docs

After the code is green, update the docs the change touches:

- `README.md`: usage, commands, API names, milestone checkboxes, source layout table.
- Skill files under `.pi/skills/` if the workflow or commands changed.
- Remove stale statements (old names, old version, fixed bugs described as open).
- Check that every command in the README actually runs. Run it.

Commit docs separately: `docs: ...`.

## 6. PR

```bash
git push -u origin <branch>
gh pr create --base main --head <branch> --title "<imperative title>" --body-file <file>
```

PR body must include:
- Summary (what and why, 2-4 bullets)
- Tests: the command run and the result
- Manual checks (e.g. `curl` / browser) if any
- Known limits and what was skipped

Report the PR URL to the user. Remove the worktree only after the PR is merged or the user asks: `git worktree remove ../Game.2048-<short-name>`.

## Checklist

- [ ] Prompt and acceptance criteria captured
- [ ] Worktree created, branch named, nothing left on `main`
- [ ] Tests written first; `zig build test` passes with 0.15.2
- [ ] Docs match the code; README commands verified
- [ ] Pushed, PR opened with tests and checks in the body
