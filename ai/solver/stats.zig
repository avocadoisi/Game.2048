const std = @import("std");

/// Mean and standard error (sample sd / sqrt(n)) of per-seed-range averages.
pub const MeanErr = struct { mean: f64, stderr: f64 };

pub fn meanStderr(xs: []const f64) MeanErr {
    const n: f64 = @floatFromInt(xs.len);
    var sum: f64 = 0;
    for (xs) |x| sum += x;
    const mean = sum / n;
    if (xs.len < 2) return .{ .mean = mean, .stderr = 0 };
    var ss: f64 = 0;
    for (xs) |x| ss += (x - mean) * (x - mean);
    return .{ .mean = mean, .stderr = @sqrt(ss / (n - 1) / n) };
}

/// Add one eval's per-game results to `acc`, one seed range at a time.
///
/// `Result` is the solver's per-game result: `score`, `moves`, `max_tile` and `leaves` are required;
/// `node_probes` / `node_hits` are read only when both it and `Acc` have them. `Acc` is the runner's
/// row, whose `range_scores` slice the caller allocates with one slot per seed range. The reach
/// counters are filled only when `Acc` has those fields, so the nn and n-tuple runners can share
/// this loop without either printing a column it does not have (#304).
///
/// `avg_score` stays the mean of the seed-range averages, never the pooled average; `avg_moves`
/// divides by every game (`eval_games * ranges`) exactly as the two hand-written loops did.
pub fn rangeTotals(
    comptime Result: type,
    comptime Acc: type,
    acc: *Acc,
    results: []const Result,
    eval_games: u32,
) void {
    const ranges = acc.range_scores.len;
    const games: usize = ranges * @as(usize, eval_games);
    std.debug.assert(results.len == games);

    acc.total_score = 0;
    acc.leaves = 0;
    acc.best_tile = 0;
    acc.avg_moves = 0;
    if (comptime @hasField(Acc, "node_probes")) acc.node_probes = 0;
    if (comptime @hasField(Acc, "node_hits")) acc.node_hits = 0;
    if (comptime @hasField(Acc, "reach_4096")) acc.reach_4096 = 0;
    if (comptime @hasField(Acc, "reach_8192")) acc.reach_8192 = 0;
    if (comptime @hasField(Acc, "reach_16384")) acc.reach_16384 = 0;
    if (comptime @hasField(Acc, "dead_below_2048")) acc.dead_below_2048 = 0;

    var total_moves: u64 = 0;
    for (acc.range_scores, 0..) |*range_score, range| {
        var range_total: u64 = 0;
        const start = range * @as(usize, eval_games);
        for (results[start..][0..@as(usize, eval_games)]) |res| {
            range_total += res.score;
            acc.total_score += res.score;
            total_moves += res.moves;
            acc.leaves += res.leaves;
            acc.best_tile = @max(acc.best_tile, res.max_tile);
            if (comptime @hasField(Result, "node_probes") and @hasField(Acc, "node_probes")) {
                acc.node_probes += res.node_probes;
            }
            if (comptime @hasField(Result, "node_hits") and @hasField(Acc, "node_hits")) {
                acc.node_hits += res.node_hits;
            }
            if (comptime @hasField(Acc, "reach_4096")) {
                if (res.max_tile >= 4096) acc.reach_4096 += 1;
            }
            if (comptime @hasField(Acc, "reach_8192")) {
                if (res.max_tile >= 8192) acc.reach_8192 += 1;
            }
            if (comptime @hasField(Acc, "reach_16384")) {
                if (res.max_tile >= 16384) acc.reach_16384 += 1;
            }
            if (comptime @hasField(Acc, "dead_below_2048")) {
                if (res.max_tile < 2048) acc.dead_below_2048 += 1;
            }
        }
        range_score.* = @as(f64, @floatFromInt(range_total)) / @as(f64, @floatFromInt(eval_games));
    }
    acc.avg_moves = @as(f64, @floatFromInt(total_moves)) / @as(f64, @floatFromInt(games));
}

test "meanStderr gives mean and sample standard error" {
    const xs = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const r = meanStderr(&xs);
    try std.testing.expectApproxEqAbs(@as(f64, 5), r.mean, 1e-12);
    // sample variance = 32 / 7, stderr = sqrt(32/7) / sqrt(8)
    try std.testing.expectApproxEqAbs(@sqrt(32.0 / 7.0) / @sqrt(8.0), r.stderr, 1e-12);
}

test "meanStderr of one value has zero stderr" {
    const r = meanStderr(&[_]f64{3});
    try std.testing.expectApproxEqAbs(@as(f64, 3), r.mean, 1e-12);
    try std.testing.expectEqual(@as(f64, 0), r.stderr);
}

const PlainResult = struct {
    score: u32,
    moves: u32,
    max_tile: u32,
    leaves: u64 = 0,
};

const NodeResult = struct {
    score: u32,
    moves: u32,
    max_tile: u32,
    leaves: u64 = 0,
    node_probes: u64 = 0,
    node_hits: u64 = 0,
};

const PlainAcc = struct {
    range_scores: []f64,
    avg_moves: f64 = 0,
    best_tile: u32 = 0,
    total_score: u64 = 0,
    leaves: u64 = 0,
};

const CounterAcc = struct {
    range_scores: []f64,
    avg_moves: f64 = 0,
    best_tile: u32 = 0,
    total_score: u64 = 0,
    leaves: u64 = 0,
    node_probes: u64 = 0,
    node_hits: u64 = 0,
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    dead_below_2048: u32 = 0,
};

test "rangeTotals aggregates per seed range and fills optional counters" {
    const plain = [_]PlainResult{
        .{ .score = 100, .moves = 10, .max_tile = 1024, .leaves = 1 },
        .{ .score = 200, .moves = 20, .max_tile = 4096, .leaves = 2 },
        .{ .score = 300, .moves = 30, .max_tile = 8192, .leaves = 3 },
        .{ .score = 400, .moves = 40, .max_tile = 16384, .leaves = 4 },
    };
    var plain_scores = [_]f64{ 0, 0 };
    var plain_acc = PlainAcc{ .range_scores = &plain_scores };
    rangeTotals(PlainResult, PlainAcc, &plain_acc, &plain, 2);

    const expected_scores = [_]f64{ 150, 350 };
    try std.testing.expectEqualSlices(f64, &expected_scores, plain_acc.range_scores);
    try std.testing.expectEqual(@as(u64, 1000), plain_acc.total_score);
    try std.testing.expectEqual(@as(u64, 10), plain_acc.leaves);
    try std.testing.expectEqual(@as(u32, 16384), plain_acc.best_tile);
    try std.testing.expectApproxEqAbs(@as(f64, 25), plain_acc.avg_moves, 1e-12);

    const node = [_]NodeResult{
        .{ .score = 100, .moves = 10, .max_tile = 1024, .leaves = 1, .node_probes = 5, .node_hits = 1 },
        .{ .score = 200, .moves = 20, .max_tile = 4096, .leaves = 2, .node_probes = 6, .node_hits = 2 },
        .{ .score = 300, .moves = 30, .max_tile = 8192, .leaves = 3, .node_probes = 7, .node_hits = 3 },
        .{ .score = 400, .moves = 40, .max_tile = 16384, .leaves = 4, .node_probes = 8, .node_hits = 4 },
    };
    var node_scores = [_]f64{ 0, 0 };
    var node_acc = CounterAcc{ .range_scores = &node_scores };
    rangeTotals(NodeResult, CounterAcc, &node_acc, &node, 2);

    try std.testing.expectEqualSlices(f64, &expected_scores, node_acc.range_scores);
    try std.testing.expectEqual(plain_acc.total_score, node_acc.total_score);
    try std.testing.expectEqual(plain_acc.leaves, node_acc.leaves);
    try std.testing.expectEqual(plain_acc.best_tile, node_acc.best_tile);
    try std.testing.expectApproxEqAbs(plain_acc.avg_moves, node_acc.avg_moves, 1e-12);
    try std.testing.expectEqual(@as(u64, 26), node_acc.node_probes);
    try std.testing.expectEqual(@as(u64, 10), node_acc.node_hits);
    try std.testing.expectEqual(@as(u32, 3), node_acc.reach_4096);
    try std.testing.expectEqual(@as(u32, 2), node_acc.reach_8192);
    try std.testing.expectEqual(@as(u32, 1), node_acc.reach_16384);
    try std.testing.expectEqual(@as(u32, 1), node_acc.dead_below_2048);
}

test "rangeTotals skips counters the result or accumulator does not have" {
    const node = [_]NodeResult{
        .{ .score = 10, .moves = 1, .max_tile = 2048, .leaves = 1, .node_probes = 9, .node_hits = 3 },
        .{ .score = 20, .moves = 2, .max_tile = 2048, .leaves = 2, .node_probes = 9, .node_hits = 3 },
    };
    var scores_without_counter_fields = [_]f64{0};
    var plain_acc = PlainAcc{ .range_scores = &scores_without_counter_fields };
    rangeTotals(NodeResult, PlainAcc, &plain_acc, &node, 2);
    try std.testing.expectEqual(@as(u64, 30), plain_acc.total_score);
    try std.testing.expectApproxEqAbs(@as(f64, 15), plain_acc.range_scores[0], 1e-12);

    const plain = [_]PlainResult{
        .{ .score = 10, .moves = 1, .max_tile = 4096, .leaves = 1 },
        .{ .score = 20, .moves = 2, .max_tile = 4096, .leaves = 2 },
    };
    var scores_with_counter_fields = [_]f64{0};
    var counter_acc = CounterAcc{ .range_scores = &scores_with_counter_fields };
    rangeTotals(PlainResult, CounterAcc, &counter_acc, &plain, 2);
    try std.testing.expectEqual(@as(u64, 30), counter_acc.total_score);
    try std.testing.expectEqual(@as(u64, 0), counter_acc.node_probes);
    try std.testing.expectEqual(@as(u64, 0), counter_acc.node_hits);
    try std.testing.expectEqual(@as(u32, 2), counter_acc.reach_4096);
}
