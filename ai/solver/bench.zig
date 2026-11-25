const std = @import("std");
const solver = @import("solver");
const parallel = @import("parallel");

/// Bins of the final max-tile histogram: `max_tiles[i]` counts the games whose highest tile was
/// `2 << i` (2, 4, 8 … 131072, the largest tile a 4x4 board can hold). Reach at any threshold is a
/// suffix sum, so one counter per game serves every tile the page can show instead of a fixed set of
/// `reach_*` fields (issue #369).
pub const max_tile_logs = 17;

/// Scores are bucketed into fixed-width bands: `bands[i]` counts the games that ended in
/// `[i * score_band_width, (i + 1) * score_band_width)`. 40 bands of 10,000 cover the 0-400,000 range a
/// 4x4 game can reach; anything above clamps into the last band (issue #371).
pub const score_band_width: u32 = 10_000;
pub const score_bands = 40;

/// Games whose final max tile was one bin, and the sum of their scores: the average score of the games
/// that reached a tile is a suffix sum over the bins, exactly like the reach counts (#371).
pub const TileTally = struct { games: u32 = 0, score: u64 = 0 };

/// How the scores of a batch are spread: the tails and the middle the average alone cannot show, plus
/// the band histogram the page draws (#371).
pub const ScoreStats = struct {
    min: u32 = 0,
    /// Lower middle score for an even number of games: the mean of the two middle scores.
    median: u32 = 0,
    max: u32 = 0,
    stddev: f64 = 0,
    bands: [score_bands]u32 = @splat(0),
};

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
    /// central measure, so how often a batch actually gets there has to be counted separately.
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    /// Games that ended below 2048, i.e. died early.
    dead_below_2048: u32 = 0,
    /// Final max tile per game, exactly one bin per game (see `max_tile_logs`). The `reach_*` fields
    /// above are suffix sums of it, so a tile the page wants that has no field (`1024`, say) is still
    /// available.
    max_tiles: [max_tile_logs]TileTally = @splat(.{}),
    /// Spread of the same games' scores (#371).
    scores: ScoreStats = .{},

    /// Games whose final max tile was at least `tile` (a power of two). 0 when `tile` is out of range.
    pub fn reached(self: Summary, tile: u32) u32 {
        return self.reachedTally(tile).games;
    }

    /// Games that reached `tile` and their summed scores, in one pass over the bins.
    pub fn reachedTally(self: Summary, tile: u32) TileTally {
        var tally: TileTally = .{};
        for (self.max_tiles, 0..) |bin_tally, i| {
            const bin: u32 = @as(u32, 1) << @intCast(i + 1);
            if (bin < tile) continue;
            tally.games += bin_tally.games;
            tally.score += bin_tally.score;
        }
        return tally;
    }

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
    max_tiles: [max_tile_logs]TileTally = @splat(.{}),
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

/// `scores` holds one slot per game of the whole batch (run `r`, game `i` at `r * games_per_run + i`),
/// so the score distribution is computed from the games themselves instead of from a lossy summary.
const RunCtx = struct { plan: Plan, depth: u8, games_per_run: u32, seed: u64, io: std.Io, scores: []u32 };

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
    for (0..c.games_per_run) |i| {
        const result = playOne(prng.random(), c.plan, c.depth);
        res.score += result.score;
        res.moves += result.moves;
        res.best_tile = @max(res.best_tile, result.max_tile);
        res.max_depth = @max(res.max_depth, solver.id_depth_reached);
        // One bin per game: 2 -> 0, 4 -> 1, … 131072 -> max_tile_logs - 1.
        const tile = @max(result.max_tile, 2);
        const bin = &res.max_tiles[std.math.log2_int(u32, tile) - 1];
        bin.games += 1;
        bin.score += result.score;
        c.scores[r * c.games_per_run + i] = @intCast(result.score);
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
    const games: u32 = runs * games_per_run;
    const per_run = try gpa.alloc(RunResult, runs);
    defer gpa.free(per_run);
    const scores = try gpa.alloc(u32, games);
    defer gpa.free(scores);
    parallel.map(RunResult, per_run, RunCtx{ .plan = plan, .depth = depth, .games_per_run = games_per_run, .seed = seed, .io = io, .scores = scores }, playRun);
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best_tile: u32 = 0;
    var serial_elapsed: i96 = 0;
    var max_depth: u8 = 0;
    var max_tiles: [max_tile_logs]TileTally = @splat(.{});
    for (per_run) |r| {
        total_score += r.score;
        total_moves += r.moves;
        best_tile = @max(best_tile, r.best_tile);
        max_depth = @max(max_depth, r.max_depth);
        serial_elapsed += r.elapsed_ns;
        for (r.max_tiles, 0..) |bin, i| {
            max_tiles[i].games += bin.games;
            max_tiles[i].score += bin.score;
        }
    }
    const elapsed = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const n: f64 = @floatFromInt(games);
    const avg_score: f64 = @as(f64, @floatFromInt(total_score)) / n;
    var s: Summary = .{
        .games = games,
        .avg_score = avg_score,
        .avg_moves = @as(f64, @floatFromInt(total_moves)) / n,
        .best_tile = best_tile,
        .elapsed_ns = elapsed,
        .serial_elapsed_ns = serial_elapsed,
        .total_moves = total_moves,
        .max_depth = max_depth,
        .max_tiles = max_tiles,
        .scores = scoreStats(scores, avg_score),
    };
    // The `reach_*` counters are suffix sums of the histogram, so they cannot drift from it.
    s.reach_4096 = s.reached(4096);
    s.reach_8192 = s.reached(8192);
    s.reach_16384 = s.reached(16384);
    s.dead_below_2048 = games - s.reached(2048);
    return s;
}

/// Distribution of the games' scores: tails, middle, spread and the band histogram. Sorts `scores` in
/// place, so pass it as a scratch copy the caller no longer needs.
fn scoreStats(scores: []u32, avg: f64) ScoreStats {
    var st: ScoreStats = .{};
    if (scores.len == 0) return st;
    var sum_sq: f64 = 0;
    for (scores) |score| {
        const v: f64 = @floatFromInt(score);
        sum_sq += v * v;
        st.bands[@min(score / score_band_width, score_bands - 1)] += 1;
    }
    std.mem.sort(u32, scores, {}, std.sort.asc(u32));
    st.min = scores[0];
    st.max = scores[scores.len - 1];
    const mid = scores.len / 2;
    st.median = if (scores.len % 2 == 1)
        scores[mid]
    else
        @intCast((@as(u64, scores[mid - 1]) + scores[mid]) / 2);
    const n: f64 = @floatFromInt(scores.len);
    st.stddev = @sqrt(@max(sum_sq / n - avg * avg, 0));
    return st;
}

test "scorePerSecond divides total score by elapsed seconds" {
    // 3000 points in 1.5 s is 2000 points per second
    try std.testing.expectApproxEqAbs(@as(f64, 2000), scorePerSecond(3000, 1_500_000_000), 1e-9);
}

test "scorePerSecond of zero elapsed time is zero" {
    try std.testing.expectEqual(@as(f64, 0), scorePerSecond(3000, 0));
}

test "final max-tile histogram holds every game once and the reach counters are its suffix sums" {
    const s = try run(std.testing.io, std.testing.allocator, .greedy, 2, 3, 1, 11);
    var total: u32 = 0;
    var total_score: u64 = 0;
    for (s.max_tiles) |tally| {
        total += tally.games;
        total_score += tally.score;
    }
    try std.testing.expectEqual(s.games, total);
    // one bin per game, each a power of two no larger than the best tile of the batch
    for (s.max_tiles, 0..) |tally, bin| {
        if (tally.games == 0) continue;
        try std.testing.expect(@as(u32, 1) << @intCast(bin + 1) <= s.best_tile);
    }
    // every game's score is in exactly one bin, so the tallies carry the batch's whole score
    try std.testing.expectEqual(@as(u64, @intFromFloat(s.avg_score * @as(f64, @floatFromInt(s.games)))), total_score);
    // the tally above every tile is the whole batch
    const all = s.reachedTally(1);
    try std.testing.expectEqual(s.games, all.games);
    try std.testing.expectEqual(total_score, all.score);
    // a reach average never exceeds the batch average, and no game means no average
    try std.testing.expectEqual(@as(u32, 0), s.reachedTally(1 << 20).games);
    try std.testing.expectEqual(s.reach_4096, s.reached(4096));
    try std.testing.expectEqual(s.reach_8192, s.reached(8192));
    try std.testing.expectEqual(s.games - s.dead_below_2048, s.reached(2048));
    // a threshold above the largest bin, and one below the smallest, are still well defined
    try std.testing.expectEqual(@as(u32, 0), s.reached(1 << 20));
    try std.testing.expectEqual(s.games, s.reached(1));
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
    var scores: [8]u32 = undefined;
    var played: usize = 0;
    for (0..4) |r| {
        var prng = std.Random.DefaultPrng.init(7 + r);
        for (0..2) |_| {
            const res = solver.playPolicy(prng.random(), .expectimax, 1);
            scores[played] = @intCast(res.score);
            played += 1;
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
    // the score distribution is the same batch seen from the other side
    std.mem.sort(u32, &scores, {}, std.sort.asc(u32));
    try std.testing.expectEqual(scores[0], s.scores.min);
    try std.testing.expectEqual(scores[7], s.scores.max);
    try std.testing.expectEqual((@as(u64, scores[3]) + scores[4]) / 2, s.scores.median);
    var bands: [score_bands]u32 = @splat(0);
    for (scores) |score| bands[score / score_band_width] += 1;
    try std.testing.expectEqualSlices(u32, &bands, &s.scores.bands);
    // 8 games of one policy: the spread is real, so a zero deviation would be a bug
    try std.testing.expect(s.scores.stddev > 0);
}
