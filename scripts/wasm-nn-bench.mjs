// Time the NN policies inside the built wasm module, and report the module size and startup cost.
// Run after `zig build wasm -Doptimize=ReleaseFast` (and again with `-Dnn-f32=true` for the f32 net).
// Example: node scripts/wasm-nn-bench.mjs nn2 nn3 --games 3 --seed 9000
import { readFile, stat } from "node:fs/promises";

const args = process.argv.slice(2);
// `name value` pairs are flags; everything else is a policy name.
const flags = new Map();
const policies = [];
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith("--")) flags.set(args[i].slice(2), args[++i]);
  else policies.push(args[i]);
}
const flag = (name, fallback) => flags.get(name) ?? fallback;
const games = Number(flag("games", 3));
const seed = Number(flag("seed", 9000));
const cap = Number(flag("cap", 400));

const path = new URL("../zig-out/web/game2048.wasm", import.meta.url);
const bytes = await readFile(path);
const size = (await stat(path)).size;

const started = performance.now();
const module = await WebAssembly.compile(bytes);
const compile_ms = performance.now() - started;
const { exports: wasm } = await WebAssembly.instantiate(module, {});
const boot_ms = performance.now() - started;

const text = (ptr, len) => new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, ptr, len));
const names = Array.from({ length: wasm.algo_count() }, (_, i) => text(wasm.algo_name(i), wasm.algo_name_len(i)));
console.log(`wasm ${size} B, compile ${compile_ms.toFixed(1)} ms, compile+instantiate ${boot_ms.toFixed(1)} ms`);

for (const policy of policies) {
  const algo = names.indexOf(policy);
  if (algo < 0) throw new Error(`no ${policy} policy in the module`);
  let moves = 0;
  let score = 0;
  let best = 0;
  let search_ns = 0;
  for (let game = 0; game < games; game++) {
    wasm.init(seed + game, 0);
    for (;;) {
      const t0 = performance.now();
      const dir = wasm.solve(algo, 0);
      search_ns += (performance.now() - t0) * 1e6;
      if (dir < 0) break;
      moves++;
      const result = wasm.move(dir);
      if (result === 2) break;
      if (moves >= cap * games) break;
    }
    score += wasm.score();
    best = Math.max(best, wasm.best_tile());
  }
  console.log(`${policy}: ${games} games, ${moves} moves, mean score ${(score / games).toFixed(0)}, best tile ${best}, ` +
    `${(search_ns / moves / 1e6).toFixed(2)} ms/move (search only)`);
}
