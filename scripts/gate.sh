#!/usr/bin/env bash
# One entry command for every gate. Hooks, CI and hands all call this.
#
#   scripts/gate.sh format [--write]  # zig fmt, shfmt, ruff format
#   scripts/gate.sh lint              # shellcheck, ruff check, node --check
#   scripts/gate.sh test              # zig build test
#   scripts/gate.sh fast              # format + lint (pre-commit)
#   scripts/gate.sh all               # format + lint + test (pre-push, CI)
#   scripts/gate.sh install           # enable .githooks (core.hooksPath)
#
# The gate is deliberately portable: it needs only the tools below, and every
# one of them comes from `nix develop`.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

ZIG_WANT=0.17.1

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

need() {
  if ! command -v "$1" > /dev/null 2>&1; then
    echo "gate: $1 not found. Run 'nix develop' (the flake provides it) or install $1." >&2
    exit 1
  fi
}

check_zig() {
  need zig
  local have
  have=$(zig version)
  if [[ "$have" != "$ZIG_WANT" ]]; then
    echo "gate: zig $have, but this repo pins $ZIG_WANT. Run inside 'nix develop'." >&2
    exit 1
  fi
}

format() {
  local mode=${1:-check}
  case "$mode" in
    check | --write) ;;
    *)
      echo "gate: format takes --write, not '$mode'" >&2
      exit 2
      ;;
  esac

  check_zig
  need shfmt
  need ruff
  if [[ "$mode" == "--write" ]]; then
    zig fmt build.zig src ai
    shfmt -w -i 2 -ci -sr scripts/*.sh .githooks/pre-commit .githooks/pre-push
    ruff format scripts
  else
    zig fmt --check build.zig src ai
    shfmt -d -i 2 -ci -sr scripts/*.sh .githooks/pre-commit .githooks/pre-push
    ruff format --check scripts
  fi
}

lint() {
  need shellcheck
  need ruff
  need node
  shellcheck scripts/*.sh .githooks/pre-commit .githooks/pre-push
  ruff check scripts
  find scripts -name '*.mjs' -exec node --check {} \;
}

test_gate() {
  check_zig
  zig build test
}

install_hooks() {
  need git
  git config core.hooksPath .githooks
  chmod +x scripts/gate.sh .githooks/pre-commit .githooks/pre-push
  echo "gate: core.hooksPath=.githooks (pre-commit: fast, pre-push: all)"
}

case "${1:-}" in
  format)
    format "${2:-check}"
    ;;
  lint)
    lint
    ;;
  test)
    test_gate
    ;;
  fast)
    format check
    lint
    ;;
  all)
    format check
    lint
    test_gate
    ;;
  install)
    install_hooks
    ;;
  -h | --help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
