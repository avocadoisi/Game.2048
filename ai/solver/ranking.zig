const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");

const usage =
    \\usage: ranking [runs] [games_per_run] [seed] [policies]
    \\
    \\Plays each policy on the same seeds and prints JSON sorted by average score.
    \\  runs            number of runs, one PRNG seed each (default 5)
    \\  games_per_run   games per run (default 20)
    \\  seed            base PRNG seed (default 1000, not used for tuning)
    \\  policies        comma-separated policy names to run (default: all)
    \\                  e.g. "nn,nn2"; unknown names are rejected
    \\
;

const Entry = struct {
    policy: solver.Policy,
    depth: u8,
    summary: bench.Summary,
};

const PolicySpec = struct { policy: solver.Policy, depth: u8 };

const policies = [_]PolicySpec{
    .{ .policy = .random, .depth = 0 },
    .{ .policy = .greedy, .depth = 0 },
    .{ .policy = .expectimax, .depth = 2 },
    .{ .policy = .mcts, .depth = 0 },
    .{ .policy = .beam, .depth = 0 },
    .{ .policy = .expectimax_tt, .depth = 0 },
    .{ .policy = .ga, .depth = 2 },
    .{ .policy = .nn, .depth = 0 },
    .{ .policy = .nn2, .depth = 2 },
    .{ .policy = .nnmcts, .depth = 0 },
    .{ .policy = .nnbeam, .depth = 0 },
};

fn policyByName(name: []const u8) ?PolicySpec {
    for (policies) |p| {
        if (std.mem.eql(u8, @tagName(p.policy), name)) return p;
    }
    return null;
}

/// Policies named in `filter` (comma-separated, in the given order); every policy when `filter` is empty.
/// `out` must hold at least as many slots as `filter` has names.
fn selectPolicies(out: []PolicySpec, filter: []const u8) ![]PolicySpec {
    if (filter.len == 0) {
        @memcpy(out[0..policies.len], policies[0..]);
        return out[0..policies.len];
    }
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, filter, ',');
    while (it.next()) |name| {
        if (n >= out.len) return error.TooManyPolicies;
        out[n] = policyByName(name) orelse return error.UnknownPolicy;
        n += 1;
    }
    return out[0..n];
}

/// Run each selected policy; `out` must hold at least `selected.len` entries. Only the selected policies play games.
fn runEntries(io: std.Io, gpa: std.mem.Allocator, selected: []const PolicySpec, out: []Entry, runs: u32, games_per_run: u32, seed: u64) !void {
    for (selected, 0..) |p, i| {
        out[i] = .{ .policy = p.policy, .depth = p.depth, .summary = try bench.run(io, gpa, p.policy, runs, games_per_run, p.depth, seed) };
    }
}

/// One-line summary of each policy, shown next to its ranking row.
fn describe(policy: solver.Policy) []const u8 {
    return switch (policy) {
        .random => "合法手からランダムに1手選ぶ（比較の基準）",
        .greedy => "1手先だけ見る。滑らせた盤面の評価が最も高い手を選ぶ（出現や探索はしない）",
        .expectimax => "2手先まで探索し、ランダムな出現（2か4）を平均して評価する。評価は蛇行順と合体ペアの手作り関数",
        .mcts => "UCTで候補手を選び、ランダムな対局を繰り返して評価する",
        .beam => "幅32で3手先まで探索し、各空きマスに2が出現すると仮定して評価する",
        .expectimax_tt => "depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）",
        .ga => "GAで評価の重みを進化させた期待値探索（depth 2）。GA内のholdout(seed 500〜)では+4%だが、ranking seed 1000では素の expectimax より低い",
        .nn => "TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）",
        .nn2 => "TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）",
        .nnmcts => "UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）",
        .nnbeam => "NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）",
    };
}

fn byScoreDesc(_: void, a: Entry, b: Entry) bool {
    return a.summary.avg_score > b.summary.avg_score;
}

test "filter runs only the named policies" {
    var buf: [policies.len]PolicySpec = undefined;
    const selected = try selectPolicies(&buf, "nn,beam");
    try std.testing.expectEqual(@as(usize, 2), selected.len);
    try std.testing.expectEqual(solver.Policy.nn, selected[0].policy);
    try std.testing.expectEqual(solver.Policy.beam, selected[1].policy);
}

test "empty filter selects every policy" {
    var buf: [policies.len]PolicySpec = undefined;
    const selected = try selectPolicies(&buf, "");
    try std.testing.expectEqual(policies.len, selected.len);
}

test "unknown policy name is rejected" {
    var buf: [policies.len]PolicySpec = undefined;
    try std.testing.expectError(error.UnknownPolicy, selectPolicies(&buf, "nn,nope"));
}

test "runEntries executes only the filtered policies" {
    var buf: [policies.len]PolicySpec = undefined;
    const selected = try selectPolicies(&buf, "greedy");
    var entries: [policies.len]Entry = undefined;
    try runEntries(std.testing.io, std.testing.allocator, selected, &entries, 1, 1, 3);
    try std.testing.expectEqual(solver.Policy.greedy, entries[0].policy);
    try std.testing.expectEqual(@as(u32, 1), entries[0].summary.games);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const runs: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 5;
    const games_per_run: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 20;
    const seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    const filter: []const u8 = if (args.len > 4) args[4] else "";
    if (args.len > 5 or runs == 0 or games_per_run == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var table: [policies.len]PolicySpec = undefined;
    const selected = selectPolicies(&table, filter) catch |err| {
        std.debug.print("filter \"{s}\": {s}\n{s}", .{ filter, @errorName(err), usage });
        return;
    };

    var entries: [policies.len]Entry = undefined;
    try runEntries(init.io, init.arena.allocator(), selected, &entries, runs, games_per_run, seed);
    std.mem.sort(Entry, entries[0..selected.len], {}, byScoreDesc);

    var buf: [8192]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    const w = &out.interface;
    try w.print("{{\"seed\":{d},\"runs\":{d},\"games_per_run\":{d},\"entries\":[", .{ seed, runs, games_per_run });
    for (entries[0..selected.len], 0..) |e, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("{{\"policy\":\"{s}\",\"depth\":{d},\"games\":{d},\"avg_score\":{d:.1},\"avg_moves\":{d:.1},\"best_tile\":{d},\"latency_ms_per_move\":{d:.4},\"games_per_sec\":{d:.2},\"description\":\"{s}\"}}", .{
            @tagName(e.policy), e.depth, e.summary.games, e.summary.avg_score, e.summary.avg_moves, e.summary.best_tile, e.summary.latencyMsPerMove(), e.summary.gamesPerSec(), describe(e.policy),
        });
    }
    try w.writeAll("]}\n");
    try w.flush();
}
