const std = @import("std");
const solver = @import("solver");
const parallel = @import("parallel");

pub const Summary = struct {
    games: u32,
    avg_score: f64,
    avg_moves: f64,
    best_tile: u32,
    elapsed_ns: i96,

    pub fn gamesPerSec(self: Summary) f64 {
        return @as(f64, @floatFromInt(self.games)) * 1e9 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn movesPerSec(self: Summary) f64 {
        return self.avg_moves * @as(f64, @floatFromInt(self.games)) * 1e9 / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    /// Wall-clock time per move; runs execute in parallel, so use one run for a serial per-move cost.
    pub fn msPerMove(self: Summary) f64 {
        return @as(f64, @floatFromInt(self.elapsed_ns)) / 1e6 / (@as(f64, @floatFromInt(self.games)) * self.avg_moves);
    }
};

const RunResult = struct { score: u64 = 0, moves: u64 = 0, best_tile: u32 = 0 };

const RunCtx = struct { policy: solver.Policy, depth: u8, games_per_run: u32, seed: u64 };

/// One run: one PRNG seeded `seed + r`, games played in order (same stream as the serial loop).
fn playRun(c: RunCtx, r: usize) RunResult {
    var prng = std.Random.DefaultPrng.init(c.seed + r);
    var res: RunResult = .{};
    for (0..c.games_per_run) |_| {
        const result = solver.playPolicy(prng.random(), c.policy, c.depth);
        res.score += result.score;
        res.moves += result.moves;
        res.best_tile = @max(res.best_tile, result.max_tile);
    }
    return res;
}

/// Play `runs * games_per_run` games, one PRNG seed per run (seed, seed+1, ...). Runs execute in parallel.
pub fn run(io: std.Io, gpa: std.mem.Allocator, policy: solver.Policy, runs: u32, games_per_run: u32, depth: u8, seed: u64) !Summary {
    const start = std.Io.Clock.awake.now(io);
    const per_run = try gpa.alloc(RunResult, runs);
    defer gpa.free(per_run);
    parallel.map(RunResult, per_run, RunCtx{ .policy = policy, .depth = depth, .games_per_run = games_per_run, .seed = seed }, playRun);
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best_tile: u32 = 0;
    for (per_run) |r| {
        total_score += r.score;
        total_moves += r.moves;
        best_tile = @max(best_tile, r.best_tile);
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
    };
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
    for (0..4) |r| {
        var prng = std.Random.DefaultPrng.init(7 + r);
        for (0..2) |_| {
            const res = solver.playPolicy(prng.random(), .expectimax, 1);
            total_score += res.score;
            total_moves += res.moves;
            best = @max(best, res.max_tile);
        }
    }
    try std.testing.expectEqual(@as(f64, @floatFromInt(total_score)) / 8, s.avg_score);
    try std.testing.expectEqual(@as(f64, @floatFromInt(total_moves)) / 8, s.avg_moves);
    try std.testing.expectEqual(best, s.best_tile);
}
