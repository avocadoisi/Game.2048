let wasm;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      const instance = await WebAssembly.instantiate(data.module, {});
      wasm = instance.exports;
      self.postMessage({ id: data.id, value: null });
    } else if (data.type === "eval") {
      if (!wasm || !Array.isArray(data.board) || data.board.length !== 16) throw new Error("invalid worker request");
      new Uint32Array(wasm.memory.buffer, wasm.board_ptr(), 16).set(data.board);
      const value = wasm.solve_root(data.algo, data.dir);
      self.postMessage({ id: data.id, value });
    } else {
      throw new Error("unknown worker request");
    }
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error) });
  }
};
