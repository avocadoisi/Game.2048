#!/usr/bin/env bash
# Adoption check for a candidate value net (issue #207, L1 of the goal): the same three numbers every
# time, printed in one block, so two candidates are comparable without re-deriving the command lines.
#
#   scripts/eval-net.sh <candidate.bin>             20 games, seed 9000, vs ai/solver/nn_weights.bin
#   scripts/eval-net.sh <candidate.bin> --games 40 --seed 9000
#   scripts/eval-net.sh <candidate.bin> --shipped /tmp/other.bin
#
# What it prints
#   1. pairw: A = shipped, B = candidate, identical spawn tape per game. `mean diff` is B - A, and
#      `stderr` is the standard error over the games, so the acceptance rule is mean diff - 2*stderr > 0.
#   2. the non-paired table row (2-ply / `deep`) for both nets on `--seed` and `--seed + 100`:
#      avg_score, 8192 reach and ms/move. A candidate must not lose ground here either.
#   3. a VERDICT line with the numbers that decided it. ADOPT needs all three, not just the pair:
#      mean - 2*stderr > 0, a non-regressing non-paired row on both seeds (score within 3%, 8192 reach
#      within one game, ms/move within 20%), and more paired games won than lost.
#
# The 2-ply row is the whole cost (a 3-ply row is 20 min); `rows 0,2` keeps the 1-ply row out.
set -euo pipefail

cd "$(dirname "$0")/.."

cand=""
games=20
seed=9000
shipped="ai/solver/nn_weights.bin"
while [ $# -gt 0 ]; do
  case "$1" in
    --games) games="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --shipped) shipped="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) cand="$1"; shift ;;
  esac
done

if [ -z "$cand" ] || [ ! -f "$cand" ]; then
  echo "usage: scripts/eval-net.sh <candidate.bin> [--games N] [--seed S] [--shipped PATH]" >&2
  exit 2
fi
if [ ! -f "$shipped" ]; then
  echo "error: shipped net $shipped not found" >&2
  exit 2
fi

seed_b=$((seed + 100))
run() { zig build nn -Doptimize=ReleaseFast -- "$@" 2>&1; }

table=$(mktemp)
trap 'rm -f "$table"' EXIT

echo "== pairw (A = $shipped, B = $cand, $games games, seed $seed) =="
pair=$(run 0 "$games" "$seed" 0.003 1 pairw 30 "$shipped" 1 - "$cand")
echo "$pair" | grep -E "^pair " | tail -1
mean=$(echo "$pair" | sed -nE 's/.*mean diff=(-?[0-9.]+).*/\1/p')
err=$(echo "$pair" | sed -nE 's/.*± ([0-9.]+).*/\1/p')
better=$(echo "$pair" | sed -nE 's/.*better=([0-9]+) worse=([0-9]+).*/\1/p')
worse=$(echo "$pair" | sed -nE 's/.*better=([0-9]+) worse=([0-9]+).*/\2/p')

echo
echo "== non-paired 2-ply table =="
printf '%-8s %6s %12s %10s %8s\n' net seed avg_score reach8192 ms/move
for spec in "shipped:$shipped" "cand:$cand"; do
  name=${spec%%:*}
  path=${spec#*:}
  for s in "$seed" "$seed_b"; do
    row=$(run 0 "$games" "$s" 0.003 1 resume 30 "$path" 1 - - - 0,2 | grep -E "^nn 2-ply")
    score=$(echo "$row" | sed -nE 's/.*avg score=([0-9.]+).*/\1/p')
    reach=$(echo "$row" | sed -nE 's/.*reach8192=([0-9]+)\/[0-9]+.*/\1/p')
    mspm=$(echo "$row" | sed -nE 's/.*ms\/move=([0-9.]+).*/\1/p')
    printf '%-8s %6s %12s %10s %8s\n' "$name" "$s" "$score" "$reach/$games" "$mspm"
    printf '%s %s %s %s %s\n' "$name" "$s" "$score" "$reach" "$mspm" >> "$table"
  done
done

echo
# The adoption rule of `docs/measurement.md` has three parts, and all three are checked here: the paired
# +2*stderr margin (the candidate is better than the shipped net, not noise), the non-paired 2-ply rows
# on two seeds (no regression in avg_score, in 8192 reach or in ms/move), and more games won than lost.
python3 - "$table" "$mean" "$err" "$better" "$worse" <<'PY_VERDICT'
import sys

table, mean_s, err_s, better_s, worse_s = sys.argv[1:6]
rows = {}
for line in open(table):
    parts = line.split()
    if len(parts) == 5:
        rows[(parts[0], parts[1])] = (float(parts[2]), int(parts[3]), float(parts[4]))

# tolerances: 20 games of 2-ply carry a few percent of spread, so a candidate that ties the row within
# these bands is not a regression (the paired test above is the sharp instrument)
SCORE_BAND = 0.97      # candidate may sit 3% below the shipped row
REACH_SLACK = 1        # and lose at most one 8192 game in 20
SLOW_BAND = 1.20       # and be at most 20% slower per move

reasons = []
try:
    mean, err = float(mean_s), float(err_s)
    margin = mean - 2 * err
except ValueError:
    margin = None
if margin is None:
    reasons.append("pairw row missing")
elif margin <= 0:
    reasons.append(f"paired margin {margin:+.1f} <= 0 (B is not better than A beyond noise)")
if int(better_s or 0) <= int(worse_s or 0):
    reasons.append(f"B won {better_s} and lost {worse_s} of the paired games")

for seed in sorted({s for (_, s) in rows}):
    a = rows.get(("shipped", seed))
    b = rows.get(("cand", seed))
    if a is None or b is None:
        reasons.append(f"seed {seed}: non-paired row missing")
        continue
    if b[0] < a[0] * SCORE_BAND:
        reasons.append(f"seed {seed}: avg_score {b[0]:.1f} < {a[0]:.1f} (band {SCORE_BAND:.0%})")
    if b[1] < a[1] - REACH_SLACK:
        reasons.append(f"seed {seed}: 8192 reach {b[1]} < {a[1]} (slack {REACH_SLACK})")
    if b[2] > a[2] * SLOW_BAND:
        reasons.append(f"seed {seed}: {b[2]:.2f} ms/move > {a[2]:.2f} (band {SLOW_BAND:.0%})")

verdict = "ADOPT" if not reasons else "REJECT"
print(f"VERDICT: {verdict} (mean diff {mean_s} ± {err_s}, B better/worse {better_s}/{worse_s})")
if margin is not None:
    print(f"  paired margin mean - 2*stderr = {margin:+.1f}  (rule: > 0, the goal's +2 sigma)")
print("  non-paired 2-ply: score band -3%, 8192 reach -1 game, ms/move +20%")
for r in reasons:
    print(f"  FAIL: {r}")
PY_VERDICT
