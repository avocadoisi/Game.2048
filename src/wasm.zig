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
//!                             `algo` indexes the module's policy list, in the order
//!                             `wasm.policies` names them (see `algo_name`)
//! - `board_ptr() -> *const u32` 16 row-major tile values (0 = empty)
//! - `score() -> u32`, `best_tile() -> u32`
//! - `algo_count() -> u32`, `algo_name(i) -> *const u8` (NUL-terminated),
//!   `algo_name_len(i) -> u32`, and the same pair for the one-line description the page shows:
//!   `algo_desc(i)`, `algo_desc_len(i)`

const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const options = @import("options");

var state: core.GameState = .{};
var prng: std.Random.DefaultPrng = undefined;

/// Policies this module ships, in `wasm.policies` order (#154). Only these kernels are linked in —
/// `chooseAt` specialises per policy, so everything else is dead code — and the page builds its
/// dropdown from `algo_count`/`algo_name`, so the module and the UI cannot disagree.
const policies: []const solver.Policy = blk: {
    @setEvalBranchQuota(200_000); // stringToEnum builds a comptime string map per name
    var found: [32]solver.Policy = undefined;
    var n: usize = 0;
    var start: usize = 0;
    var i: usize = 0;
    while (i <= options.policies.len) : (i += 1) {
        const ch: u8 = if (i == options.policies.len) '\n' else options.policies[i];
        if (ch == '#') { // comment runs to the end of the line
            while (i < options.policies.len and options.policies[i] != '\n') : (i += 1) {}
            start = i + 1;
            continue;
        }
        if (ch != ' ' and ch != '\t' and ch != '\r' and ch != '\n' and ch != ',') continue;
        if (i > start) {
            const name = options.policies[start..i];
            const policy = std.meta.stringToEnum(solver.Policy, name) orelse
                @compileError("wasm.policies: unknown policy `" ++ name ++ "`");
            if (solver.policyInfo(policy).needs_table)
                @compileError("wasm.policies: `" ++ name ++ "` needs the n-tuple table file, which the browser cannot read");
            for (found[0..n]) |seen| {
                if (seen == policy) @compileError("wasm.policies: `" ++ name ++ "` is listed twice");
            }
            found[n] = policy;
            n += 1;
        }
        start = i + 1;
    }
    if (n == 0) @compileError("wasm.policies must name at least one policy");
    const list = found[0..n].*;
    break :blk &list;
};

/// Descriptions of the shipped policies only, resolved through `policyInfo` at compile time so the
/// strings of the policies that are *not* shipped stay out of the module.
const descriptions: []const []const u8 = blk: {
    var out: [policies.len][]const u8 = undefined;
    for (policies, 0..) |policy, i| out[i] = solver.policyInfo(policy).description;
    const list = out; // a const copy, so the block returns a static array rather than a comptime var
    break :blk &list;
};

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

/// Each arm calls `solver.chooseOne` with a constant policy, which is what keeps the unshipped
/// policies out of the module: a runtime `switch (policy)` inside `solver.choose` would link them
/// all. `inline` over the index range lets the linker drop the rest.
fn chooseAt(algo: u32, rng: std.Random) ?core.Direction {
    switch (algo) {
        inline 0...policies.len - 1 => |i| return solver.chooseOne(policies[i], state, rng, 0),
        else => return null,
    }
}

/// `depth` is part of the ABI but unused by the shipped policies: nn2 and nn3 are fixed-horizon by
/// construction (`nn.bestMove` picks 2-ply or 3-ply), the way `policy_info` records depth 0 for them.
export fn solve(algo: u32, depth: u32) i32 {
    _ = depth;
    const dir = chooseAt(algo, prng.random()) orelse return -1;
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

/// Number of policies in this module (`wasm.policies`). Everything below is empty / 0 out of range.
export fn algo_count() u32 {
    return @intCast(policies.len);
}

/// NUL-terminated, static lifetime; index is `0..algo_count()`.
export fn algo_name(algo: u32) [*]const u8 {
    if (algo >= policies.len) return "";
    return @tagName(policies[algo]).ptr;
}

export fn algo_name_len(algo: u32) u32 {
    if (algo >= policies.len) return 0;
    return @tagName(policies[algo]).len;
}

/// The one-line description from `solver.policy_info`, so the page explains the models it ships.
export fn algo_desc(algo: u32) [*]const u8 {
    if (algo >= policies.len) return "";
    return descriptions[algo].ptr;
}

export fn algo_desc_len(algo: u32) u32 {
    if (algo >= policies.len) return 0;
    return descriptions[algo].len;
}
