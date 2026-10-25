#!/usr/bin/env bash
# Run a bench command and append its output to docs/solver-history.md as a dated block.
# Usage: scripts/record-bench.sh [--doc FILE] -- <command...>
#        scripts/record-bench.sh --selftest
set -u

if [ "${1:-}" = "--selftest" ]; then
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  printf '# history\n' > "$tmp/h.md"
  "$0" --doc "$tmp/h.md" -- echo selftest-output-line >/dev/null
  block=$(cat "$tmp/h.md")
  fail=0
  for want in '## Bench run' '- command: `echo selftest-output-line`' '- commit: ' '- date: ' '- exit: 0' 'selftest-output-line'; do
    case "$block" in *"$want"*) ;; *) echo "selftest FAIL: missing '$want'" >&2; fail=1 ;; esac
  done
  [ "$fail" = 0 ] && echo "selftest PASS" || { echo "selftest FAILED" >&2; exit 1; }
  exit "$fail"
fi

doc=docs/solver-history.md
if [ "${1:-}" = "--doc" ]; then doc=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: $0 [--doc FILE] -- <command...>" >&2; exit 2; }

# Refuse when the trained weights have uncommitted changes.
if [ -n "$(git status --porcelain -- ai/solver/nn_weights.bin)" ]; then
  echo "refusing: ai/solver/nn_weights.bin has uncommitted changes" >&2
  exit 1
fi

cmd="$*"
commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
date_str=$(date '+%Y-%m-%d %H:%M:%S %z')
out=$("$@" 2>&1); code=$?
printf '\n## Bench run %s\n\n- command: `%s`\n- commit: %s\n- date: %s\n- exit: %s\n\n```text\n%s\n```\n' \
  "$date_str" "$cmd" "$commit" "$date_str" "$code" "$out" >> "$doc"
printf '%s\n' "$out"
exit "$code"
