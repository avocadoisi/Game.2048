// Run after `zig build wasm -Doptimize=ReleaseFast`.
// Example: node scripts/wasm-sequence.mjs nn3 9000 100
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { bestDirection } from "../src/web/search-workers.mjs";

const [policy = "nn3", seed = "9000", cap = "100"] = process.argv.slice(2);
const module = await WebAssembly.compile(await readFile(new URL("../zig-out/web/game2048.wasm", import.meta.url)));
const instances = await Promise.all(Array.from({ length: 6 }, async () => (await WebAssembly.instantiate(module, {})).exports));
const [serial, replica, ...branches] = instances;
const text = (ptr, len) => new TextDecoder().decode(new Uint8Array(serial.memory.buffer, ptr, len));
const algo = Array.from({ length: serial.algo_count() }, (_, i) => text(serial.algo_name(i), serial.algo_name_len(i))).indexOf(policy);
assert.ok(algo >= 0 && (policy === "nn2" || policy === "nn3"), "unknown NN policy");
for (const wasm of [serial, replica]) wasm.init(Number(seed), 0);
const board = (wasm) => Array.from(new Uint32Array(wasm.memory.buffer, wasm.board_ptr(), 16));
let moves = 0;
let finished = false;
while (moves < Number(cap)) {
  const cells = board(replica);
  assert.deepEqual(board(serial), cells);
  const expected = serial.solve(algo, 0);
  const actual = bestDirection(branches.map((wasm, dir) => {
    new Uint32Array(wasm.memory.buffer, wasm.board_ptr(), 16).set(cells);
    return wasm.solve_root(algo, dir);
  }));
  assert.equal(actual, expected, `move ${moves}`);
  if (actual < 0) { finished = true; break; }
  const result = serial.move(actual);
  assert.equal(result, replica.move(actual));
  moves++;
  assert.equal(serial.score(), replica.score());
  if (moves % 100 === 0) console.log(`${policy} seed=${seed} moves=${moves} score=${serial.score()}`);
  if (result === 2) { finished = true; break; }
}
assert.deepEqual(board(serial), board(replica));
console.log(`${finished ? "finished" : "cap reached"}: ${policy} seed=${seed} moves=${moves} score=${serial.score()} best=${serial.best_tile()}`);
