#!/usr/bin/env bash
# Publish the WASM build to GitHub Pages (manual deploy, no CI).
#
#   scripts/deploy-pages.sh              build, smoke test, push gh-pages
#   scripts/deploy-pages.sh --dry-run    build and smoke test only
#   scripts/deploy-pages.sh --skip-smoke skip `node scripts/wasm-smoke.mjs`
#   scripts/deploy-pages.sh --enable     also turn the Pages site on (gh api)
#
# The site is the three files in `zig-out/web`; the NN weights are embedded in the
# module, and the 64 MiB n-tuple table is not part of the browser build at all.
set -euo pipefail

cd "$(dirname "$0")/.."

dry_run=0
run_smoke=1
enable=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --skip-smoke) run_smoke=0 ;;
    --enable) enable=1 ;;
    -h | --help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

remote_url=$(git remote get-url origin)
repo=$(git remote get-url origin | sed -e 's#^git@[^:]*:##' -e 's#^https://[^/]*/##' -e 's#\.git$##')
commit=$(git rev-parse --short HEAD)
branch=$(git rev-parse --abbrev-ref HEAD)

echo "==> zig build wasm -Doptimize=ReleaseFast"
zig build wasm -Doptimize=ReleaseFast

for file in index.html game2048.wasm; do
  test -s "zig-out/web/$file" || {
    echo "missing zig-out/web/$file" >&2
    exit 1
  }
done

# The repo has no LICENSE text on the page itself; keep the notice next to the build.
cp LICENSE NOTICE zig-out/web/
printf '' > zig-out/web/.nojekyll

if ((run_smoke)); then
  echo "==> node scripts/wasm-smoke.mjs"
  node scripts/wasm-smoke.mjs
fi

size=$(wc -c <zig-out/web/game2048.wasm | tr -d ' ')
echo "==> build ok: game2048.wasm ${size} bytes, from ${branch} @ ${commit}"

if ((dry_run)); then
  echo "==> --dry-run: not pushing"
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
cp -R zig-out/web/. "$tmp/"
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git config user.name || echo deploy)" \
  -c user.email="$(git config user.email || echo deploy@localhost)" \
  commit -q -m "deploy: wasm ${commit} (${branch})"

echo "==> git push -f origin gh-pages"
git -C "$tmp" push -q -f "$remote_url" HEAD:gh-pages

if ((enable)); then
  echo "==> enabling Pages (source: gh-pages /)"
  gh api -X POST "repos/${repo}/pages" \
    -f "source[branch]=gh-pages" -f "source[path]=/" >/dev/null || {
    echo "enabling Pages failed — is the repository public (or the plan Pro+)?" >&2
    exit 1
  }
  gh api -X POST "repos/${repo}/pages/builds" >/dev/null || true
  echo "==> https://$(echo "$repo" | cut -d/ -f1).github.io/$(echo "$repo" | cut -d/ -f2)/"
else
  echo "==> pushed. Enable Pages once with:"
  echo "    gh api -X POST repos/${repo}/pages -f source[branch]=gh-pages -f source[path]=/"
  echo "    then open https://$(echo "$repo" | cut -d/ -f1).github.io/$(echo "$repo" | cut -d/ -f2)/"
fi
