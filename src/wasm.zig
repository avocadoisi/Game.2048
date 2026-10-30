//! C ABI for the browser build (`zig build wasm`).
//!
//! All functions are plain `export fn`s with no parameters that need a
//! heap or a host import. The JS side reads the board straight from the
//! linear memory returned by `board_ptr`.
//!
//! - `init(seed_lo, seed_hi)`  start a new game, seed = (hi << 32) | lo
//! - `move(dir) -> u32`        0 = no change, 1 = moved, 2 = game over
//!                             dir: 0 up, 1 left, 2 down, 3 right
//! - `solve(algo, depth) -> i32` direction as above, -1 when no legal move
//!                             algo index follows `solver.Policy` (see `algo_name`)
//! - `board_ptr() -> *const u32` 16 row-major tile values (0 = empty)
//! - `score() -> u32`, `best_tile() -> u32`
//! - `algo_count() -> u32`, `algo_name(i) -> *const u8` (NUL-terminated),
//!   `algo_name_len(i) -> u32`

const std = @import("std");
const core = @import("core");
const solver = @import("solver");

var state: core.GameState = .{};
var prng: std.Random.DefaultPrng = undefined;

export fn init(seed_lo: u32, seed_hi: u32) void {
    const seed: u64 = @as(u64, seed_hi) << 32 | seed_lo;
    prng = std.Random.DefaultPrng.init(seed);
    state = core.GameState.gameStart(prng.random());
}

fn indexToDirection(dir: u32) ?core.Direction {
    return switch (dir) {
        0 => .up,
        1 => .left,
        2 => .down,
        3 => .right,
        else => null,
    };
}

export fn move(dir: u32) u32 {
    const direction = indexToDirection(dir) orelse return 0;
    const result = state.move(direction, prng.random());
    if (state.isGameOver()) return 2;
    return if (result.changed) 1 else 0;
}

export fn solve(algo: u32, depth: u32) i32 {
    const policy = std.enums.fromInt(solver.Policy, algo) orelse return -1;
    const dir = solver.choose(policy, state, prng.random(), @intCast(@min(depth, 255))) orelse return -1;
    return @intFromEnum(dir);
}

export fn board_ptr() [*]const u32 {
    return @ptrCast(&state.board);
}

export fn score() u32 {
    return state.score;
}

export fn best_tile() u32 {
    var best: u32 = 0;
    for (state.board) |row| {
        for (row) |value| best = @max(best, value);
    }
    return best;
}

/// The last policy (`ntuple`) needs a 64 MiB table file, which the browser build cannot read,
/// so the web UI only ever sees the ones before it (#129).
export fn algo_count() u32 {
    return @typeInfo(solver.Policy).@"enum".field_names.len - 1;
}

/// NUL-terminated, static lifetime; index is `0..algo_count()`.
export fn algo_name(algo: u32) [*]const u8 {
    const policy = std.enums.fromInt(solver.Policy, algo) orelse return "";
    return @tagName(policy).ptr;
}

export fn algo_name_len(algo: u32) u32 {
    const policy = std.enums.fromInt(solver.Policy, algo) orelse return 0;
    return @tagName(policy).len;
}
