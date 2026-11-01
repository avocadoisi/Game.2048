const std = @import("std");
const solver = @import("solver");
const parallel = @import("parallel");

pub const Summary = struct {
    games: u32 = 0,
    avg_score: f64 = 0,
    avg_moves: f64 = 0,
    best_tile: u32 = 0,
    elapsed_ns: i96 = 0,
    /// Sum of the per-run serial times (each run plays its games on one thread).
    serial_elapsed_ns: i96 = 0,
    total_moves: u64 = 0,
    /// Deepest expectimax depth any move of the whole batch finished (`expectimax_id` only, else 0).
    max_depth: u8 = 0,
    /// Games whose final `max_tile` is at or above the tile (counts, not rates); the average score is a
    /// central measure, so how often a batch actually gets there has to be counted separately (#156).
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    /// Games that ended below 2048, i.e. died early.
    dead_below_2048: u32 = 0,

    /// Wall-clock seconds of the batch (`elapsed=… s`, `elapsed … s` in every runner's report).
    pub fn elapsedSec(self: Summary) f64 {
        return @as(f64, @floatFromInt(self.elapsed_ns)) / 1e9;
    }

    pub fn gamesPerSec(self: Summary) f64 {
        return @as(f64, @floatFromInt(self.games)) * 1e9 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    /// Serial latency: ms per move for one game played alone on its thread (not the parallel wall time).
    pub fn latencyMsPerMove(self: Summary) f64 {
        const moves: f64 = @floatFromInt(@max(self.total_moves, 1));
        return @as(f64, @floatFromInt(self.serial_elapsed_ns)) / 1e6 / moves;
    }

    pub fn movesPerSec(self: Summary) f64 {
        return self.avg_moves * @as(f64, @floatFromInt(self.games)) * 1e9 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    /// Wall-clock time per move; runs execute in parallel, so use one run for a serial per-move cost.
    pub fn msPerMove(self: Summary) f64 {
        return @as(f64, @floatFromInt(self.elapsed_ns)) / 1e6 / (@as(f64, @floatFromInt(self.games)) * self.avg_moves);
    }
};

const RunResult = struct {
    score: u64 = 0,
    moves: u64 = 0,
    best_tile: u32 = 0,
    elapsed_ns: i96 = 0,
    max_depth: u8 = 0,
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    dead_below_2048: u32 = 0,
};

/// Score per CPU second; 0 when no time elapsed (avoids division by zero).
pub fn scorePerSecond(total_score: f64, elapsed_ns: i96) f64 {
    if (elapsed_ns <= 0) return 0;
    return total_score / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
}

/// The row `zig build nn` and `zig build ntuple` print per evaluated variant: the average over seed
/// ranges (`avg_score`) with its standard error, then moves, best tile, speed and the leaf counter.
/// Both runners show these same columns, so the wording lives here; `zig build nn` appends its own
/// reach counts to the same line, and the moves divisor stays `games * avg_moves` as it always was.
/// No newline on purpose: the caller finishes the line.
pub fn printEvalRow(name: []const u8, stderr: f64, ranges: u32, s: Summary, total_score: u64, leaves: u64) void {
    const moves: f64 = @max(s.avg_moves * @as(f64, @floatFromInt(s.games)), 1);
    std.debug.print("{s}: avg score={d:.1} ± {d:.1} (stderr over {d} ranges) avg moves={d:.1} best tile={d} elapsed={d:.1} s score/s={d:.1} leaves={d} leaves/move={d:.1} ms/move={d:.2}", .{
        name,
        s.avg_score,
        stderr,
        ranges,
        s.avg_moves,
        s.best_tile,
        s.elapsedSec(),
        scorePerSecond(@floatFromInt(total_score), s.elapsed_ns),
        leaves,
        @as(f64, @floatFromInt(leaves)) / moves,
        @as(f64, @floatFromInt(s.elapsed_ns)) / 1e6 / moves,
    });
}

/// What a run plays with: a named policy, or custom evaluation weights (`zig build evalab`).
const Plan = union(enum) { policy: solver.Policy, weights: solver.Weights };

const RunCtx = struct { plan: Plan, depth: u8, games_per_run: u32, seed: u64, io: std.Io };

fn playOne(rng: std.Random, plan: Plan, depth: u8) solver.GameResult {
    return switch (plan) {
        .policy => |p| solver.playPolicy(rng, p, depth),
        .weights => |w| solver.playWeights(rng, w, depth),
    };
}

/// One run: one PRNG seeded `seed + r`, games played in order (same stream as the serial loop).
fn playRun(c: RunCtx, r: usize) RunResult {
    const start = std.Io.Clock.awake.now(c.io);
    var prng = std.Random.DefaultPrng.init(c.seed + r);
    var res: RunResult = .{};
    for (0..c.games_per_run) |_| {
        const result = playOne(prng.random(), c.plan, c.depth);
        res.score += result.score;
        res.moves += result.moves;
        res.best_tile = @max(res.best_tile, result.max_tile);
        res.max_depth = @max(res.max_depth, solver.id_depth_reached);
        if (result.max_tile >= 4096) res.reach_4096 += 1;
        if (result.max_tile >= 8192) res.reach_8192 += 1;
        if (result.max_tile >= 16384) res.reach_16384 += 1;
        if (result.max_tile < 2048) res.dead_below_2048 += 1;
    }
    res.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(c.io)).toNanoseconds();
    return res;
}

/// Play `runs * games_per_run` games, one PRNG seed per run (seed, seed+1, ...). Runs execute in parallel.
pub fn run(io: std.Io, gpa: std.mem.Allocator, policy: solver.Policy, runs: u32, games_per_run: u32, depth: u8, seed: u64) !Summary {
    return runPlan(io, gpa, .{ .policy = policy }, runs, games_per_run, depth, seed);
}

/// Same games and seeds as `run`, but with custom evaluation weights (`zig build evalab`).
pub fn runW(io: std.Io, gpa: std.mem.Allocator, w: solver.Weights, runs: u32, games_per_run: u32, depth: u8, seed: u64) !Summary {
    return runPlan(io, gpa, .{ .weights = w }, runs, games_per_run, depth, seed);
}

fn runPlan(io: std.Io, gpa: std.mem.Allocator, plan: Plan, runs: u32, games_per_run: u32, depth: u8, seed: u64) !Summary {
    const start = std.Io.Clock.awake.now(io);
    const per_run = try gpa.alloc(RunResult, runs);
    defer gpa.free(per_run);
    parallel.map(RunResult, per_run, RunCtx{ .plan = plan, .depth = depth, .games_per_run = games_per_run, .seed = seed, .io = io }, playRun);
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best_tile: u32 = 0;
    var serial_elapsed: i96 = 0;
    var max_depth: u8 = 0;
    var reach_4096: u32 = 0;
    var reach_8192: u32 = 0;
    var reach_16384: u32 = 0;
    var dead_below_2048: u32 = 0;
    for (per_run) |r| {
        total_score += r.score;
        total_moves += r.moves;
        best_tile = @max(best_tile, r.best_tile);
        max_depth = @max(max_depth, r.max_depth);
        serial_elapsed += r.elapsed_ns;
        reach_4096 += r.reach_4096;
        reach_8192 += r.reach_8192;
        reach_16384 += r.reach_16384;
        dead_below_2048 += r.dead_below_2048;
    }
    const games: u32 = runs * games_per_run;
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const n: f64 = @floatFromInt(games);
    return .{
        .games = games,
        .avg_score = @as(f64, @floatFromInt(total_score)) / n,
        .avg_moves = @as(f64, @floatFromInt(total_moves)) / n,
        .best_tile = best_tile,
        .elapsed_ns = elapsed,
        .serial_elapsed_ns = serial_elapsed,
        .total_moves = total_moves,
        .max_depth = max_depth,
        .reach_4096 = reach_4096,
        .reach_8192 = reach_8192,
        .reach_16384 = reach_16384,
        .dead_below_2048 = dead_below_2048,
    };
}

test "scorePerSecond divides total score by elapsed seconds" {
    // 3000 points in 1.5 s is 2000 points per second
    try std.testing.expectApproxEqAbs(@as(f64, 2000), scorePerSecond(3000, 1_500_000_000), 1e-9);
}

test "scorePerSecond of zero elapsed time is zero" {
    try std.testing.expectEqual(@as(f64, 0), scorePerSecond(3000, 0));
}

test "run with the baseline weights matches the expectimax policy on the same games" {
    const s = try runW(std.testing.io, std.testing.allocator, solver.baseline, 2, 2, 1, 5);
    const p = try run(std.testing.io, std.testing.allocator, .expectimax, 2, 2, 1, 5);
    try std.testing.expectEqual(p.avg_score, s.avg_score);
    try std.testing.expectEqual(p.avg_moves, s.avg_moves);
    try std.testing.expectEqual(p.best_tile, s.best_tile);
}

test "summary separates serial latency from parallel throughput" {
    const s = try run(std.testing.io, std.testing.allocator, .expectimax, 2, 2, 1, 1);
    // latency is per-move serial cost (ms), throughput is games/sec of the parallel batch
    try std.testing.expect(s.latencyMsPerMove() > 0);
    try std.testing.expect(s.gamesPerSec() > 0);
}

test "run plays the requested number of games and reports timing" {
    const s = try run(std.testing.io, std.testing.allocator, .expectimax, 2, 2, 1, 1);
    try std.testing.expectEqual(@as(u32, 4), s.games);
    try std.testing.expect(s.avg_moves > 0);
    try std.testing.expect(s.elapsed_ns > 0);
}

test "parallel run matches a serial loop over the same 8 games" {
    const s = try run(std.testing.io, std.testing.allocator, .expectimax, 4, 2, 1, 7);
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best: u32 = 0;
    var reach_4096: u32 = 0;
    var reach_8192: u32 = 0;
    var reach_16384: u32 = 0;
    var dead: u32 = 0;
    for (0..4) |r| {
        var prng = std.Random.DefaultPrng.init(7 + r);
        for (0..2) |_| {
            const res = solver.playPolicy(prng.random(), .expectimax, 1);
            total_score += res.score;
            total_moves += res.moves;
            best = @max(best, res.max_tile);
            if (res.max_tile >= 4096) reach_4096 += 1;
            if (res.max_tile >= 8192) reach_8192 += 1;
            if (res.max_tile >= 16384) reach_16384 += 1;
            if (res.max_tile < 2048) dead += 1;
        }
    }
    try std.testing.expectEqual(@as(f64, @floatFromInt(total_score)) / 8, s.avg_score);
    try std.testing.expectEqual(@as(f64, @floatFromInt(total_moves)) / 8, s.avg_moves);
    try std.testing.expectEqual(best, s.best_tile);
    try std.testing.expectEqual(reach_4096, s.reach_4096);
    try std.testing.expectEqual(reach_8192, s.reach_8192);
    try std.testing.expectEqual(reach_16384, s.reach_16384);
    try std.testing.expectEqual(dead, s.dead_below_2048);
}
