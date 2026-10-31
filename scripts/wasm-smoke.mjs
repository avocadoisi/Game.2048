// Smoke test for the static WASM build. Run: node scripts/wasm-smoke.mjs
// Exits non-zero on any failure.
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
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

const text = (ptr, len) => new TextDecoder().decode(new Uint8Array(w.memory.buffer, ptr, len));

// `--solve-one <algo>`: solve a single policy, so the parent can hold a hard deadline over a search
// that never returns. A JS timer cannot interrupt a blocked wasm call, so the deadline has to come
// from outside the process (#154 review).
const childAlgo = process.argv[2] === "--solve-one" ? Number(process.argv[3]) : -1;
if (childAlgo >= 0) {
  w.init(777 + childAlgo, 0);
  const t0 = performance.now();
  const dir = w.solve(childAlgo, 0);
  const ms = performance.now() - t0;
  // A fresh game always has a legal move, so -1 here is a bug, not a dead end.
  if (dir === -1) fail(`solve(${childAlgo}) returned -1 on a fresh game`);
  if (dir > 3) fail(`solve(${childAlgo}) returned out-of-range direction ${dir}`);
  const before = snapshot();
  const result = w.move(dir);
  if (result !== 1) fail(`solve(${childAlgo}) returned direction ${dir} but moving it returned ${result}`);
  if (before.join() === snapshot().join()) fail(`solve(${childAlgo}) direction ${dir} did not change the board`);
  console.log(`ms=${ms.toFixed(1)}`);
  process.exit(0);
}

// Solver: every policy must return a legal direction, and moving it must work.
const algoCount = w.algo_count();
if (algoCount < 1) fail("algo_count() must be >= 1");

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
// The page shows the score and the per-move cost, not the implementation notes, so a policy without
// a score renders blank. "Speed" is optional (only browser-measured policies have one).
for (let i = 0; i < algoCount; i++) {
  const score = text(w.algo_score(i), w.algo_score_len(i));
  if (!score.trim()) fail(`algo ${names[i]} has no score (${JSON.stringify(score)})`);
  const speed = text(w.algo_speed(i), w.algo_speed_len(i));
  if (speed && !/ms/.test(speed)) fail(`algo ${names[i]} has no unit in its speed (${JSON.stringify(speed)})`);
}

// The page builds its dropdown and its descriptions from the module, so it cannot list a policy that
// is not compiled in — but it does pick a default by name, and that name must be shipped here or the
// page silently falls back to the last entry.
const html = readFileSync(join(root, "zig-out/web/index.html"), "utf8");
if (!html.includes("algo_count()")) fail("zig-out/web/index.html does not build its policy list from the module");
if (!html.includes("algo_score(")) fail("zig-out/web/index.html does not show the module's policy scores");
if (!html.includes("algo_speed(")) fail("zig-out/web/index.html does not show the module's per-move cost");
const preferred = /const DEFAULT_POLICY = "([a-z0-9_]+)"/.exec(html);
if (!preferred) fail("zig-out/web/index.html has no DEFAULT_POLICY, so its default policy is unknown");
if (!want.includes(preferred[1])) {
  fail(`the page defaults to ${preferred[1]}, which wasm.policies does not ship (${want.join(", ")})`);
}

// A timed search cannot expire on a frozen clock, so `expectimax_id` used to run to id_max_depth and
// hang this script for minutes (#143, #145). The old guard was the `depth` argument; now that `solve`
// runs each policy at its own horizon (#154) nothing inside the process can cut a runaway search
// short, so every solve runs in a child process that is killed at the deadline. The slowest policy
// that may legitimately appear is `nn3` (~0.4 s per move in V8, #152), so this is a hang detector,
// not a speed budget.
const HANG_LIMIT_MS = 5000;
let slowest = 0;

for (let algo = 0; algo < algoCount; algo++) {
  const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--solve-one", String(algo)], {
    encoding: "utf8",
    timeout: HANG_LIMIT_MS,
  });
  if (child.error?.code === "ETIMEDOUT") {
    fail(`solve(${names[algo]}) did not return within ${HANG_LIMIT_MS} ms and was killed (a policy that never returns)`);
  }
  if (child.error) fail(`solve(${names[algo]}) could not run: ${child.error.message}`);
  if (child.status !== 0) fail(`solve(${names[algo]}) failed: ${(child.stderr || child.stdout).trim()}`);
  const solveMs = Number(/ms=([\d.]+)/.exec(child.stdout)?.[1]);
  if (!Number.isFinite(solveMs)) fail(`solve(${algo}) printed no timing: ${JSON.stringify(child.stdout)}`);
  slowest = Math.max(slowest, solveMs);
}

console.log(`OK: ${bytes.length} bytes wasm, ${algoCount} policies (${names.join(", ")}), ${applied} applied moves, slowest solve ${Math.round(slowest)} ms`);
