#!/usr/bin/env bash
# Evaluate a value net at 1-ply and 2-ply on held-out eval seeds.
# usage: ./.auto/measure.sh [net_path]   (env: GAMES, SEED)
set -euo pipefail
NET="${1:-ai/solver/nn_weights.bin}"
GAMES="${GAMES:-20}"
SEED="${SEED:-9200}"

OUT="$(zig build nn -Doptimize=ReleaseFast -- 0 "$GAMES" "$SEED" 0.003 1 resume 30 "$NET" 1 - - - 0,2 2>&1)"

line1="$(printf '%s\n' "$OUT" | grep 'nn 1-ply' || true)"
line2="$(printf '%s\n' "$OUT" | grep 'nn 2-ply' || true)"
if [ -z "$line2" ]; then
  printf '%s\n' "$OUT" | tail -20
  echo "METRIC eval_error=1"
  exit 1
fi

getf() { printf '%s\n' "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p"; }
echo "$line1"
echo "$line2"
echo "METRIC avg2ply=$(getf "$line2" 'score')"
echo "METRIC avg1ply=$(getf "$line1" 'score')"
echo "METRIC reach8192=$(printf '%s\n' "$line2" | sed -n 's#.*reach8192=\([0-9]*\)/.*#\1#p')"
echo "METRIC ms_per_move=$(printf '%s\n' "$line2" | awk '{for(i=1;i<=NF;i++) if($i ~ /^ms\/move=/) {sub(/^ms\/move=/,"",$i); print $i}}')"
echo "METRIC moves=$(getf "$line2" 'moves')"
