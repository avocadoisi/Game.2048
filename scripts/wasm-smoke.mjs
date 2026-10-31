// Smoke test for the static WASM build. Run: node scripts/wasm-smoke.mjs
// Exits non-zero on any failure.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasmPath = join(root, "zig-out/web/game2048.wasm");

function fail(message) {
  console.error("FAIL: " + message);
  process.exit(1);
}

const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const w = instance.exports;

for (const name of ["init", "move", "solve", "board_ptr", "score", "best_tile", "memory"]) {
  if (!(name in w)) fail(`missing export ${name}`);
}

// The ABI is documented in src/wasm.zig: 0 up, 1 left, 2 down, 3 right.
const DIRS = [0, 1, 2, 3];
const board = () => new Uint32Array(w.memory.buffer, w.board_ptr(), 16);
const snapshot = () => Array.from(board());

w.init(12345, 0);
const start = snapshot();
if (start.filter((v) => v !== 0).length !== 2) fail("init must spawn exactly two tiles, got " + JSON.stringify(start));
if (!start.every((v) => v === 0 || v === 2 || v === 4)) fail("init produced an invalid tile: " + JSON.stringify(start));

// Play a few moves: every applied move must change the board and keep tiles valid.
let applied = 0;
for (let step = 0; step < 40 && applied < 10; step++) {
  const dir = DIRS[step % DIRS.length];
  const before = snapshot();
  const result = w.move(dir);
  if (![0, 1, 2].includes(result)) fail(`move(${dir}) returned ${result}, expected 0/1/2`);
  const after = snapshot();
  if (result === 1) {
    applied++;
    if (before.join() === after.join()) fail(`move(${dir}) returned 1 but the board did not change`);
  }
  if (!after.every((v) => v === 0 || (v >= 2 && Number.isInteger(Math.log2(v))))) {
    fail("invalid tile after move: " + JSON.stringify(after));
  }
}
if (applied < 5) fail("expected at least 5 applied moves, got " + applied);

if (w.score() < 0) fail("score must not be negative");
const best = w.best_tile();
if (best !== Math.max(...snapshot())) fail(`best_tile() = ${best}, board max = ${Math.max(...snapshot())}`);

// Solver: every policy must return a legal direction (or -1), and moving it must work.
const algoCount = w.algo_count();
if (algoCount < 1) fail("algo_count() must be >= 1");

const names = [];
for (let i = 0; i < algoCount; i++) {
  const len = w.algo_name_len(i);
  names.push(new TextDecoder().decode(new Uint8Array(w.memory.buffer, w.algo_name(i), len)));
}
if (!names.includes("expectimax")) fail("algo names missing expectimax: " + JSON.stringify(names));
if (!names.includes("nn3")) fail("algo names missing nn3 (the static page's fixed policy): " + JSON.stringify(names));

// The page falls back to `expectimax` without a word when its policy name is missing, so bind the
// two: the shipped HTML and this test must agree on the name.
const html = readFileSync(join(root, "zig-out/web/index.html"), "utf8");
if (!html.includes('const SOLVER_NAME = "nn3"')) fail("zig-out/web/index.html is not built around nn3");

// A timed search cannot expire on a frozen clock, so `expectimax_id` used to run to id_max_depth and
// hang this script for minutes (#143, #145). Budget each call so the regression cannot come back
// silently. The slowest policy that may legitimately appear here is `nn3` (up to ~0.4 s per move in
// V8, #152), so the limit is a hang detector, not a speed budget.
const SOLVE_LIMIT_MS = 2000;
let slowest = 0;

for (let algo = 0; algo < algoCount; algo++) {
  w.init(777 + algo, 0);
  const t0 = performance.now();
  const dir = w.solve(algo, 1);
  const solveMs = performance.now() - t0;
  if (solveMs > SOLVE_LIMIT_MS) fail(`solve(${algo}) took ${Math.round(solveMs)} ms (limit ${SOLVE_LIMIT_MS})`);
  slowest = Math.max(slowest, solveMs);
  if (dir < -1 || dir > 3) fail(`solve(${algo}) returned out-of-range direction ${dir}`);
  if (dir === -1) continue; // no legal move
  const before = snapshot();
  const result = w.move(dir);
  if (result !== 1) fail(`solve(${algo}) returned direction ${dir} but moving it returned ${result}`);
  if (before.join() === snapshot().join()) fail(`solve(${algo}) direction ${dir} did not change the board`);
}

// A fresh game always has a legal move for the policies the page uses (depth 2).
w.init(777, 0);
const expectimax = names.indexOf("expectimax");
if (w.solve(expectimax, 1) < 0) fail("solve(expectimax) returned -1 on a fresh game");
w.init(777, 0);
if (w.solve(names.indexOf("nn3"), 2) < 0) fail("solve(nn3, 2) returned -1 on a fresh game");

console.log(`OK: ${bytes.length} bytes wasm, ${algoCount} policies (${names.join(", ")}), ${applied} applied moves, slowest solve ${Math.round(slowest)} ms`);
