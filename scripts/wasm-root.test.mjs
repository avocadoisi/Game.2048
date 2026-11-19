// Run after `zig build wasm -Doptimize=ReleaseFast`.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { bestDirection } from "../src/web/search-workers.mjs";

const bytes = await readFile(new URL("../zig-out/web/game2048.wasm", import.meta.url));
const module = await WebAssembly.compile(bytes);
const serial = (await WebAssembly.instantiate(module, {})).exports;
const branches = await Promise.all(Array.from({ length: 4 }, async () => (await WebAssembly.instantiate(module, {})).exports));
const names = [];
const decode = new TextDecoder();
for (let i = 0; i < serial.algo_count(); i++) {
  names.push(decode.decode(new Uint8Array(serial.memory.buffer, serial.algo_name(i), serial.algo_name_len(i))));
}
const boards = [
  [2, 4, 8, 16, 4, 8, 16, 32, 0, 2, 4, 8, 0, 0, 2, 4],
  [1024, 512, 256, 128, 64, 32, 16, 8, 4, 8, 16, 0, 2, 0, 0, 0],
  [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2, 2],
  [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2],
];
function setBoard(wasm, board) {
  new Uint32Array(wasm.memory.buffer, wasm.board_ptr(), 16).set(board);
}

test("four independent WASM root branches select the serial move", () => {
  for (const name of ["nn2", "nn3"]) {
    const algo = names.indexOf(name);
    if (algo < 0) continue;
    for (const board of boards) {
      setBoard(serial, board);
      const expected = serial.solve(algo, 0);
      const values = branches.map((wasm, dir) => {
        setBoard(wasm, board);
        return wasm.solve_root(algo, dir);
      });
      assert.equal(bestDirection(values), expected, `${name}: ${board.join(",")}`);
    }
  }
  assert.ok(names.includes("nn2") || names.includes("nn3"), "no NN search policy in wasm.policies");
});
