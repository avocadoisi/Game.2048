import test from "node:test";
import assert from "node:assert/strict";
import { RootWorkerPool, bestDirection } from "../src/web/search-workers.mjs";

test("root aggregation skips illegal directions and breaks ties in serial order", () => {
  assert.equal(bestDirection([NaN, 5, 5, 2]), 1);
  assert.equal(bestDirection([NaN, NaN, NaN, NaN]), -1);
  assert.throws(() => bestDirection([0, Infinity, 1, 2]));
});

test("pool reuses four workers and resolves independent root scores", async () => {
  const workers = [];
  class FakeWorker {
    postMessage(message) {
      queueMicrotask(() => this.onmessage({ data: {
        id: message.id, value: message.type === "eval" ? [NaN, 2, 2, 1][message.dir] : null,
      } }));
    }
    terminate() { this.terminated = true; }
  }
  const pool = new RootWorkerPool({}, () => {
    const worker = new FakeWorker();
    workers.push(worker);
    return worker;
  });
  assert.equal(await pool.solve(new Uint32Array(16), 0), 1);
  assert.equal(await pool.solve(new Uint32Array(16), 1), 1);
  assert.equal(workers.length, 4);
  pool.close();
  assert.ok(workers.every((worker) => worker.terminated));
});

test("initialization failure closes every worker even before solving", async () => {
  const workers = [];
  class FailingInitWorker {
    postMessage(message) {
      queueMicrotask(() => this.onmessage({ data: { id: message.id, error: "init failed" } }));
    }
    terminate() { this.terminated = true; }
  }
  const pool = new RootWorkerPool({}, () => {
    const worker = new FailingInitWorker();
    workers.push(worker);
    return worker;
  });
  await assert.rejects(pool.ready, /init failed/);
  assert.ok(workers.every((worker) => worker.terminated));
});

test("worker failure rejects pending requests and closes the pool", async () => {
  const workers = [];
  class BrokenWorker {
    postMessage(message) {
      queueMicrotask(() => message.type === "init"
        ? this.onmessage({ data: { id: message.id, value: null } })
        : this.onerror({ message: "worker failed" }));
    }
    terminate() { this.terminated = true; }
  }
  const pool = new RootWorkerPool({}, () => {
    const worker = new BrokenWorker();
    workers.push(worker);
    return worker;
  });
  await assert.rejects(pool.solve(new Uint32Array(16), 0), /worker failed/);
  assert.ok(workers.every((worker) => worker.terminated));
});
