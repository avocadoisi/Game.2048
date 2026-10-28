const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");

const usage =
    \\usage: evalab [specs] [runs] [games_per_run] [depth] [seed]
    \\
    \\Compares evaluation variants on the same seeds and prints one row each (issues #111-#115).
    \\  specs           ';'-separated variants, e.g. "layout=snake;layout=spiral;monotonic=3"
    \\                  keys: layout=<snake|column_snake|spiral|block2>, anchor=<0..7>,
    \\                        empty, order, order_late, merge, forced_down, smoothness, monotonic,
    \\                        tile_sum
    \\                  "baseline" and "ga" are shorthands; missing keys keep the shipped weights.
    \\                  `order` sets both the mid and the late-phase order penalty.
    \\  runs            number of runs, one PRNG seed each (default 5)
    \\  games_per_run   games per run (default 20)
    \\  depth           expectimax depth in player moves (default 2)
    \\  seed            base PRNG seed (default 1000, not used for tuning)
    \\
;

/// Shipped weights plus the keys in `spec`; unknown keys or values are an error.
fn parseSpec(spec: []const u8) !solver.Weights {
    if (std.mem.eql(u8, spec, "baseline")) return solver.baseline;
    if (std.mem.eql(u8, spec, "ga")) return solver.ga_weights;
    var w = solver.baseline;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |pair| {
        const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.BadSpec;
        const key = pair[0..eq];
        const value = pair[eq + 1 ..];
        if (std.mem.eql(u8, key, "layout")) {
            w.layout = std.meta.stringToEnum(solver.Layout, value) orelse return error.BadSpec;
        } else if (std.mem.eql(u8, key, "anchor")) {
            w.anchor = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, key, "ordered")) {
            w.ordered = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, key, "empty")) {
            w.empty = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "order")) {
            w.order_penalty = try std.fmt.parseFloat(f64, value);
            w.order_penalty_late = w.order_penalty;
        } else if (std.mem.eql(u8, key, "order_late")) {
            w.order_penalty_late = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "merge")) {
            w.merge = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "forced_down")) {
            w.forced_down = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "smoothness")) {
            w.smoothness = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "monotonic")) {
            w.monotonic = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, key, "tile_sum")) {
            w.tile_sum = try std.fmt.parseFloat(f64, value);
        } else return error.BadSpec;
    }
    return w;
}

/// Play every variant on the same seeds and print avg score / moves / best tile / speed.
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const specs = if (args.len > 1) args[1] else "baseline";
    const runs: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 5;
    const games_per_run: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 20;
    const depth: u8 = if (args.len > 4) try std.fmt.parseInt(u8, args[4], 10) else 2;
    const seed: u64 = if (args.len > 5) try std.fmt.parseInt(u64, args[5], 10) else 1000;
    if (args.len > 6 or runs == 0 or games_per_run == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    std.debug.print("{s:<44} {s:>10} {s:>10} {s:>7} {s:>8}\n", .{ "variant", "avg score", "avg moves", "best", "games/s" });
    var it = std.mem.splitScalar(u8, specs, ';');
    while (it.next()) |spec| {
        const w = parseSpec(spec) catch {
            std.debug.print("unknown spec: {s}\n\n{s}", .{ spec, usage });
            return;
        };
        const s = try bench.runW(init.io, init.arena.allocator(), w, runs, games_per_run, depth, seed);
        std.debug.print("{s:<44} {d:>10.1} {d:>10.1} {d:>7} {d:>8.2}\n", .{ spec, s.avg_score, s.avg_moves, s.best_tile, s.gamesPerSec() });
    }
}

test "parseSpec starts from the shipped weights and applies every key" {
    const w = try parseSpec("layout=spiral,anchor=3,empty=5,order=20,merge=0,forced_down=0,smoothness=1.5,monotonic=2,tile_sum=0.5");
    try std.testing.expectEqual(solver.Layout.spiral, w.layout);
    try std.testing.expectEqual(@as(u8, 3), w.anchor);
    try std.testing.expectEqual(@as(f64, 5), w.empty);
    try std.testing.expectEqual(@as(f64, 20), w.order_penalty);
    try std.testing.expectEqual(@as(f64, 20), w.order_penalty_late);
    try std.testing.expectEqual(@as(f64, 0), w.merge);
    try std.testing.expectEqual(@as(f64, 0), w.forced_down);
    try std.testing.expectEqual(@as(f64, 1.5), w.smoothness);
    try std.testing.expectEqual(@as(f64, 2), w.monotonic);
    try std.testing.expectEqual(@as(f64, 0.5), w.tile_sum);
}

test "parseSpec sets the late-phase order penalty on its own" {
    const w = try parseSpec("order=40,order_late=25");
    try std.testing.expectEqual(@as(f64, 40), w.order_penalty);
    try std.testing.expectEqual(@as(f64, 25), w.order_penalty_late);
}

test "parseSpec keeps the shipped weights for the presets" {
    try std.testing.expectEqual(solver.baseline, try parseSpec("baseline"));
    try std.testing.expectEqual(solver.ga_weights, try parseSpec("ga"));
    try std.testing.expectEqual(solver.baseline, try parseSpec("layout=snake"));
}

test "parseSpec rejects unknown keys and values" {
    try std.testing.expectError(error.BadSpec, parseSpec("nope=1"));
    try std.testing.expectError(error.BadSpec, parseSpec("layout=triangle"));
    try std.testing.expectError(error.BadSpec, parseSpec("merge"));
}
