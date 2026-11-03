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
#   3. a VERDICT line with the numbers that decided it.
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

echo "== pairw (A = $shipped, B = $cand, $games games, seed $seed) =="
pair=$(run 0 "$games" "$seed" 0.003 1 pairw 30 "$shipped" 1 - "$cand")
echo "$pair" | grep -E "^pair " | tail -1
mean=$(echo "$pair" | sed -nE 's/.*mean diff=(-?[0-9.]+).*/\1/p')
err=$(echo "$pair" | sed -nE 's/.*± ([0-9.]+).*/\1/p')
better=$(echo "$pair" | sed -nE 's/.*better=([0-9]+) worse=([0-9]+).*/\1\/\2/p')

echo
echo "== non-paired 2-ply table =="
printf '%-8s %6s %12s %10s %8s\n' net seed avg_score reach8192 ms/move
for spec in "shipped:$shipped" "cand:$cand"; do
  name=${spec%%:*}
  path=${spec#*:}
  for s in "$seed" "$seed_b"; do
    row=$(run 0 "$games" "$s" 0.003 1 resume 30 "$path" 1 - - - 0,2 | grep -E "^nn 2-ply")
    score=$(echo "$row" | sed -nE 's/.*avg score=([0-9.]+).*/\1/p')
    reach=$(echo "$row" | sed -nE 's/.*reach8192=([0-9]+\/[0-9]+).*/\1/p')
    mspm=$(echo "$row" | sed -nE 's/.*ms\/move=([0-9.]+).*/\1/p')
    printf '%-8s %6s %12s %10s %8s\n' "$name" "$s" "$score" "$reach" "$mspm"
  done
done

echo
verdict="REJECT"
if [ -n "$mean" ] && [ -n "$err" ]; then
  verdict=$(awk -v m="$mean" -v e="$err" 'BEGIN { print (m - 2*e > 0) ? "ADOPT" : "REJECT" }')
fi
echo "VERDICT: $verdict (mean diff $mean ± $err, B better/worse $better; rule: mean diff - 2*stderr > 0)"
