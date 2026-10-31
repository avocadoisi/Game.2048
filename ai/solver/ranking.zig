const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");
const cli = @import("args.zig");

const default_runs: u32 = 5;
const default_games_per_run: u32 = 20;
const default_seed: u64 = 1000;

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

/// Policies `zig build ranking` runs without a filter, in `Policy` order: exactly the entries
/// `solver.policy_info` marks `ranked`, so the list and the depths come from one table (#139).
const policies = blk: {
    var list: []const PolicySpec = &.{};
    for (solver.policy_info) |info| {
        if (info.ranked) list = list ++ &[_]PolicySpec{.{ .policy = info.policy, .depth = info.depth }};
    }
    break :blk list;
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

/// One-line summary of each policy, shown next to its ranking row. The text lives in
/// `solver.policy_info`, so the ranking and the HTTP UI cannot disagree (#139).
fn describe(policy: solver.Policy) []const u8 {
    return solver.policyInfo(policy).description;
}

fn byScoreDesc(_: void, a: Entry, b: Entry) bool {
    return a.summary.avg_score > b.summary.avg_score;
}

test "every policy has a ranking description" {
    // The wording lives in `solver.policy_info`, which refuses to compile without a description for
    // every variant; this keeps the table from being emptied out or the text duplicated.
    for (std.enums.values(solver.Policy)) |p| {
        try std.testing.expect(describe(p).len > 0);
    }
}

test "the default ranking list is the ranked half of policy_info, in Policy order" {
    // Compatibility surface: the default `zig build ranking` run (and its output order before the
    // sort) has always been these 11 policies. Heavy / experimental ones stay opt-in.
    const expected = [_]solver.Policy{
        .random, .greedy, .expectimax, .mcts,   .beam,   .expectimax_tt,
        .ga,     .nn,     .nn2,        .nnmcts, .nnbeam,
    };
    try std.testing.expectEqual(expected.len, policies.len);
    for (expected, 0..) |p, i| try std.testing.expectEqual(p, policies[i].policy);
    try std.testing.expectEqual(@as(u8, 2), policies[2].depth); // expectimax keeps its depth 2
    try std.testing.expectEqual(@as(u8, 2), policies[6].depth); // ga and nn2 too
    for ([_]solver.Policy{ .nn3, .ntuple, .expectimax_id }) |p| {
        try std.testing.expect(!solver.policyInfo(p).ranked);
    }
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

test {
    // `zig build test` only runs tests of files it reaches from a target's root, so the shared CLI
    // helper's tests ride along with the ranking target instead of adding a 16th one (#139).
    std.testing.refAllDecls(cli);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const runs: u32 = try cli.argOr(u32, args, 1, default_runs);
    const games_per_run: u32 = try cli.argOr(u32, args, 2, default_games_per_run);
    const seed: u64 = try cli.argOr(u64, args, 3, default_seed);
    const filter: []const u8 = try cli.argOr([]const u8, args, 4, "");
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
