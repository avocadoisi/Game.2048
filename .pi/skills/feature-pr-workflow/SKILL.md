---
name: feature-pr-workflow
description: Standard flow for every change in Game.2048. Use when starting any feature, fix, or refactor, or when merging a PR. Order is prompt -> GitHub issue -> git worktree -> new branch -> TDD -> fix docs -> PR (linked to the issue) -> merge -> cleanup.
---

# Feature PR workflow (Game.2048)

Every change follows these steps in order (the issue step comes first). Do not skip a step. Do not work on `main` directly.

## 1. Prompt (capture the request)

- Restate the goal in one or two lines and list acceptance criteria.
- If the request is ambiguous, ask before creating anything.
- Write the list of steps with the todo tool.

## 1b. GitHub issue (before any work starts)

- Create an issue before touching code: `gh issue create --title "<title>" --body-file <file>`.
- Body: Goal, Background, Candidates or scope, and Acceptance criteria as a checklist.
- Report the issue URL to the user. Work starts only after the issue exists.
- The PR body must contain `Refs #<n>`, or `Closes #<n>` when the PR completes the issue.
- Draft PRs are allowed while work is partial. Mark the PR ready (`gh pr ready <n>`) when done.

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
- Run `zig build test` with Zig **0.17** (`zig version` must print 0.17.x).
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

Report the PR URL to the user.

## 7. Merge and cleanup

Merge only when the user says so (for example "mergeして良い"). Merge is not automatic after PR creation.

Before merging:
- Re-run `zig build test` in the worktree. It must pass with Zig 0.17.
- Confirm `gh pr view <n> --json mergeable` reports `MERGEABLE`.

Stacked PRs (a PR whose base is another feature branch):
- Merge the base PR first.
- Retarget the child PR to `main` with `gh pr edit <n> --base main`. Then check it is still `MERGEABLE`.
- Merge the child PR after the retarget.

Merge and cleanup:

```bash
gh pr merge <n> --merge            # keep commit history (TDD steps stay visible)
cd <repo> && git pull --ff-only origin main
git worktree remove ../Game.2048-<short-name>
git branch -d <branch>             # local branch; remote branch is deleted by gh if asked
```

- Use `--merge` by default. Use `--squash` only when the user asks or the commits are noise.
- After merge, run `zig build test` on the updated `main` once.
- Remove the worktree after merge, not before.

## Parallel work (subagents, applies throughout)

- When the user says to do things in parallel (「並行して」), split the items across subagents and run them concurrently; each writing subagent gets its own worktree (see §2).
- Even without the word "parallel", use subagents proactively whenever there are multiple independent items.
- Only one heavy run at a time (`zig build test`, benchmarks, training); everything else may run in parallel with it.

## Checklist

- [ ] Prompt and acceptance criteria captured
- [ ] GitHub issue created and its URL reported before work starts
- [ ] Worktree created, branch named, nothing left on `main`
- [ ] Tests written first; `zig build test` passes with Zig 0.17
- [ ] Docs match the code; README commands verified
- [ ] Pushed, PR opened with `Refs #<n>` / `Closes #<n>` and tests in the body
- [ ] Merged only after the user approved; stacked PRs retargeted to `main` first
- [ ] `main` pulled, tests re-run, worktree and local branch removed
