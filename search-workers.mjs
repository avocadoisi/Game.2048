export function bestDirection(values) {
  let direction = -1;
  let best = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (Number.isNaN(value)) continue;
    if (!Number.isFinite(value)) throw new Error("invalid root value");
    if (direction < 0 || value > best) { direction = i; best = value; }
  }
  return direction;
}

export class RootWorkerPool {
  constructor(module, makeWorker = () => new Worker(new URL("solver-worker.mjs", import.meta.url), { type: "module" }), timeoutMs = 30000) {
    this.workers = [];
    this.closed = false;
    this.nextId = 0;
    this.timeoutMs = timeoutMs;
    try {
      for (let i = 0; i < 4; i++) {
        const worker = makeWorker();
        const slot = { worker, pending: new Map() };
        this.workers.push(slot);
        worker.onmessage = ({ data }) => {
          const pending = slot.pending.get(data.id);
          if (!pending) return;
          slot.pending.delete(data.id);
          clearTimeout(pending.timer);
          if (data.error) pending.reject(new Error(data.error));
          else pending.resolve(data.value);
        };
        worker.onerror = (error) => this.close(new Error(error.message || "worker failed"));
        worker.onmessageerror = () => this.close(new Error("worker message failed"));
      }
      this.ready = Promise.all(this.workers.map((slot) => this.request(slot, { type: "init", module })))
        .catch((error) => { this.close(error); throw error; });
      // A worker can fail before the user first asks the pool to solve.
      void this.ready.catch(() => {});
    } catch (error) {
      this.close(error);
      throw error;
    }
  }

  request(slot, message) {
    if (this.closed) return Promise.reject(new Error("worker pool closed"));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.close(new Error("worker timed out")), this.timeoutMs);
      slot.pending.set(id, { resolve, reject, timer });
      try { slot.worker.postMessage({ ...message, id }); }
      catch (error) { this.close(error); }
    });
  }

  async solve(board, algo) {
    try {
      await this.ready;
      const values = await Promise.all(this.workers.map((slot, dir) =>
        this.request(slot, { type: "eval", board: Array.from(board), algo, dir })));
      return bestDirection(values);
    } catch (error) {
      this.close(error);
      throw error;
    }
  }

  close(error = new Error("worker pool closed")) {
    if (this.closed) return;
    this.closed = true;
    for (const { worker, pending } of this.workers) {
      worker.terminate();
      for (const request of pending.values()) {
        clearTimeout(request.timer);
        request.reject(error);
      }
      pending.clear();
    }
  }
}
