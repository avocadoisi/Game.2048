//! Search-value diagnosis (#356): does a higher search value at a board predict a higher score
//! actually realised from that board?
//!
//! The 2-ply/3-ply/4-ply rows only change the *tree*. If their values barely correlate with what the
//! same policy goes on to score, more search is fitting estimator noise rather than a better
//! decision, and the next step has to change something else (leaf value, trajectory, ensemble).
//!
//! For each sampled board this prints the afterstate value, the 2-ply and 3-ply root search values,
//! and the mean of `rollouts` independent greedy continuations with the real spawn RNG.
//!
//!   nn-diag <weights> <boards> <rollouts> <every> <game-seed> [rollout-seed]
//!
//! stdout (stderr keeps the log line) is a CSV:
//! `move,empties,max_tile,v_afterstate,v_deep,v_deep3,outcome_mean,outcome_sd`

const std = @import("std");
const core = @import("core");
const nn = @import("nn");
const cli = @import("args.zig");

const Direction = core.Direction;
const dir_order = [4]Direction{ .up, .left, .down, .right };

const usage =
    \\usage: zig build nn-diag -- <weights> <boards> <rollouts> <every> <game-seed> [rollout-seed]
    \\  weights       net file (f64 `.bin`)
    \\  boards        number of boards to sample (stops early at game over)
    \\  rollouts      independent greedy continuations per board
    \\  every         sample one board every N moves
    \\  game-seed     seed of the sampling game
    \\  rollout-seed  seed of the continuations (default: game-seed + 1)
    \\  [greedy|deep] continuation policy (default: greedy)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len <= 1) {
        std.debug.print("{s}", .{usage});
        return;
    }
    const weights_path = args[1];
    const boards: usize = try cli.argOr(usize, args, 2, 64);
    const rollouts: u32 = try cli.argOr(u32, args, 3, 8);
    const every: u32 = try cli.argOr(u32, args, 4, 25);
    const game_seed: u64 = try cli.argOr(u64, args, 5, 7000);
    const rollout_seed: u64 = try cli.argOr(u64, args, 6, game_seed +% 1);
    // The continuation policy matters: `greedy` is cheap and gives many samples, `deep` rolls the
    // same policy that produced the search values, which is the faithful "same policy" test.
    const rollout_mode: nn.Mode = if (args.len > 7 and std.mem.eql(u8, args[7], "deep")) .deep else .greedy;

    const net = try init.gpa.create(nn.Net);
    defer init.gpa.destroy(net);
    net.* = std.mem.zeroes(nn.Net);
    try nn.load(net, init.io, init.gpa, weights_path);
    const vfn: nn.Value = .{ .single = net };

    var cache: nn.ValueCache = .{};
    var fcache: nn.Cache = undefined;
    var found: usize = 0;
    var game_prng = std.Random.DefaultPrng.init(game_seed);
    var state = core.GameState.gameStart(game_prng.random());
    var moves: u32 = 0;
    std.debug.print("loaded {s}: up to {d} boards every {d} moves, {d} {s} rollouts each\n", .{ weights_path, boards, every, rollouts, @tagName(rollout_mode) });
    while (found < boards) {
        if (moves % every == 0) {
            const r = rollout(state, vfn, rollouts, rollout_seed, rollout_mode, &cache);
            std.debug.print("{d},{d},{d},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{
                moves,
                core.countEmpty(state.board),
                core.maxTile(state.board),
                nn.valueAt(vfn, state.board, &fcache),
                positionValue(vfn, state, .deep, &cache),
                positionValue(vfn, state, .deep3, &cache),
                r.mean,
                r.sd,
            });
            found += 1;
        }
        var stats: nn.Stats = .{};
        const dir = nn.bestMoveWithCacheValueOpts(vfn, state, .deep, &stats, &cache, .{}) orelse break;
        _ = state.move(dir, game_prng.random());
        moves += 1;
    }
}

/// Root search value of a position under `mode`: the best move's reward plus its subtree, exactly as
/// the search scores it (`nn.scoreRootMoveValue`), with the mode's own cutoff.
fn positionValue(vfn: nn.Value, state: core.GameState, mode: nn.Mode, cache: *nn.ValueCache) f64 {
    var best: f64 = -std.math.inf(f64);
    for (dir_order) |d| {
        const v = nn.scoreRootMoveValue(vfn, state, mode, d, cache) orelse continue;
        if (v > best) best = v;
    }
    return if (best == -std.math.inf(f64)) 0 else best;
}

const Rollout = struct { mean: f64, sd: f64 };

/// `rollouts` independent `.greedy` continuations from `start`, in points divided by
/// `nn.reward_scale` so they are comparable with the search values.
fn rollout(start: core.GameState, vfn: nn.Value, rollouts: u32, seed: u64, mode: nn.Mode, cache: *nn.ValueCache) Rollout {
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    const n = @max(rollouts, 1);
    for (0..n) |k| {
        var state = start;
        state.score = 0;
        var move_prng = std.Random.DefaultPrng.init(seed +% @as(u64, k) *% 0x9E3779B97F4A7C15);
        while (true) {
            var stats: nn.Stats = .{};
            const dir = nn.bestMoveWithCacheValueOpts(vfn, state, mode, &stats, cache, .{}) orelse break;
            _ = state.move(dir, move_prng.random());
        }
        const v = @as(f64, @floatFromInt(state.score)) / nn.reward_scale;
        sum += v;
        sum_sq += v * v;
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const variance = if (n > 1) @max(sum_sq / @as(f64, @floatFromInt(n)) - mean * mean, 0) else 0;
    return .{ .mean = mean, .sd = @sqrt(variance) };
}
