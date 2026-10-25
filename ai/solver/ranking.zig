const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");

const usage =
    \\usage: ranking [runs] [games_per_run] [seed]
    \\
    \\Plays each policy on the same seeds and prints JSON sorted by average score.
    \\  runs            number of runs, one PRNG seed each (default 5)
    \\  games_per_run   games per run (default 20)
    \\  seed            base PRNG seed (default 1000, not used for tuning)
    \\
;

const Entry = struct {
    policy: solver.Policy,
    depth: u8,
    summary: bench.Summary,
};

/// One-line summary of each policy, shown next to its ranking row.
fn describe(policy: solver.Policy) []const u8 {
    return switch (policy) {
        .random => "legal move chosen uniformly at random (baseline)",
        .greedy => "one move ahead: pick the move whose slid board scores best (no spawn, no search)",
        .expectimax => "search 2 player moves ahead, averaging over random spawns (2 or 4); scores with the snake-order and merge heuristic",
    };
}

fn byScoreDesc(_: void, a: Entry, b: Entry) bool {
    return a.summary.avg_score > b.summary.avg_score;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const runs: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 5;
    const games_per_run: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 20;
    const seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    if (args.len > 4 or runs == 0 or games_per_run == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    const policies = [_]struct { policy: solver.Policy, depth: u8 }{
        .{ .policy = .random, .depth = 0 },
        .{ .policy = .greedy, .depth = 0 },
        .{ .policy = .expectimax, .depth = 2 },
    };
    var entries: [policies.len]Entry = undefined;
    for (policies, 0..) |p, i| {
        entries[i] = .{ .policy = p.policy, .depth = p.depth, .summary = bench.run(init.io, p.policy, runs, games_per_run, p.depth, seed) };
    }
    std.mem.sort(Entry, &entries, {}, byScoreDesc);

    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    const w = &out.interface;
    try w.print("{{\"seed\":{d},\"runs\":{d},\"games_per_run\":{d},\"entries\":[", .{ seed, runs, games_per_run });
    for (entries, 0..) |e, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("{{\"policy\":\"{s}\",\"depth\":{d},\"games\":{d},\"avg_score\":{d:.1},\"avg_moves\":{d:.1},\"best_tile\":{d},\"description\":\"{s}\"}}", .{
            @tagName(e.policy), e.depth, e.summary.games, e.summary.avg_score, e.summary.avg_moves, e.summary.best_tile, describe(e.policy),
        });
    }
    try w.writeAll("]}\n");
    try w.flush();
}
