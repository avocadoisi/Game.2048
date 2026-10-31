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

// Solver: every policy must return a legal direction, and moving it must work.
const algoCount = w.algo_count();
if (algoCount < 1) fail("algo_count() must be >= 1");

const text = (ptr, len) => new TextDecoder().decode(new Uint8Array(w.memory.buffer, ptr, len));
const names = [];
for (let i = 0; i < algoCount; i++) names.push(text(w.algo_name(i), w.algo_name_len(i)));

// Which policies ship is deploy config, so bind the module to that file (#154): same names, same
// order, and a description for each — the page renders both and has no list of its own.
const spec = readFileSync(join(root, "wasm.policies"), "utf8");
const want = spec
  .split("\n")
  .map((line) => line.split("#")[0])
  .join(" ")
  .split(/[\s,]+/)
  .filter(Boolean);
if (want.length === 0) fail("wasm.policies names no policy");
if (algoCount !== want.length) {
  fail(`wasm.policies lists ${want.length} policies (${want.join(", ")}) but the module offers ${algoCount} (${names.join(", ")})`);
}
for (const [i, name] of want.entries()) {
  if (names[i] !== name) fail(`algo ${i} is ${names[i]}, wasm.policies says ${name}`);
}
for (let i = 0; i < algoCount; i++) {
  const desc = text(w.algo_desc(i), w.algo_desc_len(i));
  if (desc.length < 10) fail(`algo ${names[i]} has no description (${JSON.stringify(desc)})`);
}

// The page builds its dropdown from the module, so it cannot name a policy that is not compiled in.
const html = readFileSync(join(root, "zig-out/web/index.html"), "utf8");
if (!html.includes("algo_count()")) fail("zig-out/web/index.html does not build its policy list from the module");
if (html.includes("SOLVER_NAME")) fail("zig-out/web/index.html still hardcodes a policy name");

// A timed search cannot expire on a frozen clock, so `expectimax_id` used to run to id_max_depth and
// hang this script for minutes (#143, #145). Budget each call so the regression cannot come back
// silently. The slowest policy that may legitimately appear here is `nn3` (up to ~0.4 s per move in
// V8, #152), so the limit is a hang detector, not a speed budget.
const SOLVE_LIMIT_MS = 2000;
let slowest = 0;

for (let algo = 0; algo < algoCount; algo++) {
  w.init(777 + algo, 0);
  const t0 = performance.now();
  const dir = w.solve(algo, 0);
  const solveMs = performance.now() - t0;
  if (solveMs > SOLVE_LIMIT_MS) fail(`solve(${algo}) took ${Math.round(solveMs)} ms (limit ${SOLVE_LIMIT_MS})`);
  slowest = Math.max(slowest, solveMs);
  if (dir < -1 || dir > 3) fail(`solve(${algo}) returned out-of-range direction ${dir}`);
  // A fresh game always has a legal move, so -1 here is a bug, not a dead end.
  if (dir === -1) fail(`solve(${names[algo]}) returned -1 on a fresh game`);
  const before = snapshot();
  const result = w.move(dir);
  if (result !== 1) fail(`solve(${algo}) returned direction ${dir} but moving it returned ${result}`);
  if (before.join() === snapshot().join()) fail(`solve(${algo}) direction ${dir} did not change the board`);
}

console.log(`OK: ${bytes.length} bytes wasm, ${algoCount} policies (${names.join(", ")}), ${applied} applied moves, slowest solve ${Math.round(slowest)} ms`);
