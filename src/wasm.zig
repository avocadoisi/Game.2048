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
//!                             `wasm.policies` names them (see `algo_name`); `depth` 0 means
//!                             "the policy's own horizon" (`policy_info`)
//! - `board_ptr() -> *const u32` 16 row-major tile values (0 = empty)
//! - `score() -> u32`, `best_tile() -> u32`
//! - `algo_count() -> u32`, `algo_name(i) -> *const u8` (NUL-terminated),
//!   `algo_name_len(i) -> u32`, and the same pair for the one-line description the page shows:
//!   `algo_score(i)`, `algo_score_len(i)`: the policy's measured average score, for the page

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
    // Some editors write a BOM; it is not part of the first name.
    const spec = if (std.mem.startsWith(u8, options.policies, "\xEF\xBB\xBF")) options.policies[3..] else options.policies;
    var found: [32]solver.Policy = undefined;
    var n: usize = 0;
    var start: usize = 0;
    var i: usize = 0;
    while (i <= spec.len) : (i += 1) {
        const ch: u8 = if (i == spec.len) '\n' else spec[i];
        const comment = ch == '#';
        if (!comment and ch != ' ' and ch != '\t' and ch != '\r' and ch != '\n' and ch != ',') continue;
        // A name ends at a separator *or* at the start of a comment (`nn2#note` still ships nn2).
        if (i > start) {
            const name = spec[start..i];
            const policy = std.meta.stringToEnum(solver.Policy, name) orelse
                @compileError("wasm.policies: unknown policy `" ++ name ++ "`");
            if (solver.policyInfo(policy).needs_table)
                @compileError("wasm.policies: `" ++ name ++ "` needs the n-tuple table file, which the browser cannot read");
            for (found[0..n]) |seen| {
                if (seen == policy) @compileError("wasm.policies: `" ++ name ++ "` is listed twice");
            }
            if (n == found.len) @compileError("wasm.policies: too many policies (max 32)");
            found[n] = policy;
            n += 1;
        }
        start = i + 1;
        if (comment) { // runs to the end of the line
            while (i < spec.len and spec[i] != '\n') : (i += 1) {}
            start = i + 1;
        }
    }
    if (n == 0) @compileError("wasm.policies must name at least one policy");
    const list = found[0..n].*;
    break :blk &list;
};

/// Scores of the shipped policies only, resolved through `policyInfo` at compile time so the
/// strings of the policies that are *not* shipped stay out of the module.
const scores: []const []const u8 = blk: {
    var out: [policies.len][]const u8 = undefined;
    for (policies, 0..) |policy, i| out[i] = solver.policyInfo(policy).score;
    const list = out; // a const copy, so the block returns a static array rather than a comptime var
    break :blk &list;
};

/// Default horizon per shipped policy: `solve`'s `depth == 0` means "the policy's own horizon",
/// the same convention `expectimax_id` uses for its time budget. Without it a depth-dependent
/// policy added to `wasm.policies` (`expectimax`, `ga`) would be searched at depth 0, which wraps
/// to 255 plies and freezes the tab (#154 review).
const depths: []const u8 = blk: {
    var out: [policies.len]u8 = undefined;
    for (policies, 0..) |policy, i| out[i] = solver.policyInfo(policy).depth;
    const list = out;
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
fn chooseAt(algo: u32, rng: std.Random, depth: u32) ?core.Direction {
    switch (algo) {
        inline 0...policies.len - 1 => |i| {
            // `depth` 0 = the policy's own horizon (nn2/nn3 ignore it: `nn.bestMove` is fixed-horizon)
            return solver.chooseOne(policies[i], state, rng, if (depth == 0) depths[i] else @intCast(depth));
        },
        else => return null,
    }
}

/// `depth` is the horizon in plies; 0 = the policy's own default (`policy_info`). The page passes 0
/// and the shipped policies (nn2, nn3) ignore the value entirely — it exists for the policies that
/// take a horizon (`expectimax`, `ga`), which a deploy can add through `wasm.policies`.
export fn solve(algo: u32, depth: u32) i32 {
    const dir = chooseAt(algo, prng.random(), @min(depth, 255)) orelse return -1;
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

/// The measured average score from `solver.policy_info`. The page shows this instead of the
/// implementation notes: a player comparing nn2 and nn3 wants to know how strong they are, not how
/// they search.
export fn algo_score(algo: u32) [*]const u8 {
    if (algo >= policies.len) return "";
    return scores[algo].ptr;
}

export fn algo_score_len(algo: u32) u32 {
    if (algo >= policies.len) return 0;
    return scores[algo].len;
}
