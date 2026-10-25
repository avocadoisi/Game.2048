const std = @import("std");
const solver = @import("solver");

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
};

/// Play `runs * games_per_run` games, one PRNG seed per run (seed, seed+1, ...).
pub fn run(io: std.Io, runs: u32, games_per_run: u32, depth: u8, seed: u64) Summary {
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best_tile: u32 = 0;
    var games: u32 = 0;
    const start = std.Io.Clock.awake.now(io);
    for (0..runs) |r| {
        var prng = std.Random.DefaultPrng.init(seed + r);
        for (0..games_per_run) |_| {
            const result = solver.playGame(prng.random(), depth);
            total_score += result.score;
            total_moves += result.moves;
            best_tile = @max(best_tile, result.max_tile);
            games += 1;
        }
    }
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
    const s = run(std.testing.io, 2, 2, 1, 1);
    try std.testing.expectEqual(@as(u32, 4), s.games);
    try std.testing.expect(s.avg_moves > 0);
    try std.testing.expect(s.elapsed_ns > 0);
}
