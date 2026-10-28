const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");
const cmaes = @import("cmaes.zig");

/// Dimensions CMA-ES may move, in `Weights` field order. `--dims` picks a subset; the dims left
/// out keep the baseline value, because their bounds collapse onto it (see `main`).
const dim_count = 4;
const names = [dim_count][]const u8{ "order_penalty", "order_penalty_late", "merge", "monotonic" };
const lo = [dim_count]f64{ 0, 0, 0, 0 };
const hi = [dim_count]f64{ 120, 120, 40, 96 };
/// Initial step size in weight units, about a tenth of the search range.
const sigma = 8;

const usage =
    \\usage: tune [--dims <list>] [--pop <n>] [--gens <n>] [--games <n>] [--depth <n>]
    \\            [--seed <n>] [--time-budget-sec <s>]
    \\
    \\Searches the snake evaluation weights with CMA-ES (issue #105) and prints the best weights
    \\as an `evalab` spec. Objective: average score of `bench.runW` on `--games` seeds from `--seed`.
    \\  --dims             comma-separated dims to optimise, from order_penalty, order_penalty_late,
    \\                     merge, monotonic (default: all four; the others stay at baseline)
    \\  --pop              CMA-ES population (default 6, even and >= 2)
    \\  --gens             generations (default 20)
    \\  --games            games per evaluation, one seed each (default 15)
    \\  --depth            expectimax depth (default 2)
    \\  --seed             base PRNG seed for the training games (default 8000)
    \\  --time-budget-sec  wall-clock budget; the run stops at the next generation (default 600)
    \\
;

/// What one evaluation needs: the training games plus the wall-clock budget.
const State = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    games: u32,
    depth: u8,
    seed: u64,
    start: std.Io.Timestamp,
    budget_ns: i96,
    expired: bool = false,
};

const Ctx = struct { state: *State };

fn weights(x: [dim_count]f64) solver.Weights {
    var w = solver.baseline;
    w.order_penalty = x[0];
    w.order_penalty_late = x[1];
    w.merge = x[2];
    w.monotonic = x[3];
    return w;
}

/// Negative average score over the training seeds: CMA-ES minimises, the score is maximised.
/// Returns inf, which ends the run at the next generation, once the time budget is spent.
fn objective(ctx: Ctx, x: [dim_count]f64) f64 {
    const s = ctx.state;
    if (s.start.durationTo(std.Io.Clock.awake.now(s.io)).toNanoseconds() > s.budget_ns) {
        if (!s.expired) {
            s.expired = true;
            std.debug.print("time budget spent, stopping\n", .{});
        }
        return std.math.inf(f64);
    }
    const summary = bench.runW(s.io, s.gpa, weights(x), s.games, 1, s.depth, s.seed) catch |err| {
        std.debug.panic("tune: bench failed: {t}", .{err});
    };
    return -summary.avg_score;
}

/// `evalab` spec of `w`; see `docs/evalab.md`.
fn spec(w: solver.Weights, buf: []u8) []const u8 {
    // `order` sets the mid and the late penalty, so `order_late` has to follow it
    return std.fmt.bufPrint(buf, "order={d:.3},order_late={d:.3},merge={d:.3},monotonic={d:.3}", .{ w.order_penalty, w.order_penalty_late, w.merge, w.monotonic }) catch unreachable;
}

/// Index of `name` in `names`, or null.
fn dimIndex(name: []const u8) ?usize {
    for (names, 0..) |candidate, k| {
        if (std.mem.eql(u8, candidate, name)) return k;
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var dims_spec: []const u8 = "order_penalty,order_penalty_late,merge,monotonic";
    var pop: usize = 6;
    var gens: usize = 20;
    var games: u32 = 15;
    var depth: u8 = 2;
    var seed: u64 = 8000;
    var budget_sec: f64 = 600;

    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) {
            std.debug.print("{s}", .{usage});
            return;
        }
        const flag = args[i];
        const value = args[i + 1];
        if (std.mem.eql(u8, flag, "--dims")) {
            dims_spec = value;
        } else if (std.mem.eql(u8, flag, "--pop")) {
            pop = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--gens")) {
            gens = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--games")) {
            games = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--depth")) {
            depth = try std.fmt.parseInt(u8, value, 10);
        } else if (std.mem.eql(u8, flag, "--seed")) {
            seed = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, flag, "--time-budget-sec")) {
            budget_sec = try std.fmt.parseFloat(f64, value);
        } else {
            std.debug.print("unknown option: {s}\n\n{s}", .{ flag, usage });
            return;
        }
    }
    if (pop < 2 or pop % 2 != 0 or pop > 64 or gens == 0 or games == 0 or depth == 0 or budget_sec <= 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var selected: [dim_count]bool = @splat(false);
    var it = std.mem.splitScalar(u8, dims_spec, ',');
    while (it.next()) |name| {
        const k = dimIndex(name) orelse {
            std.debug.print("unknown dim: {s}\n\n{s}", .{ name, usage });
            return;
        };
        selected[k] = true;
    }

    // The search bounds: the tuned range for a selected dim, a pinned `[baseline, baseline]` else.
    const base = [dim_count]f64{
        solver.baseline.order_penalty,
        solver.baseline.order_penalty_late,
        solver.baseline.merge,
        solver.baseline.monotonic,
    };
    var bounds: [dim_count][2]f64 = undefined;
    for (0..dim_count) |k| bounds[k] = if (selected[k]) .{ lo[k], hi[k] } else .{ base[k], base[k] };

    var state: State = .{
        .io = init.io,
        .gpa = arena,
        .games = games,
        .depth = depth,
        .seed = seed,
        .start = std.Io.Clock.awake.now(init.io),
        .budget_ns = @intFromFloat(budget_sec * 1e9),
    };
    const start = state.start;
    const best = cmaes.minimise(dim_count, Ctx{ .state = &state }, .{
        .x0 = base,
        .sigma = sigma,
        .pop = pop,
        .generations = gens,
        .seed = seed,
        .bounds = bounds,
        .verbose = true,
    }, objective);

    const spec_buf = try arena.alloc(u8, 128);
    const best_spec = spec(weights(best), spec_buf);
    const elapsed = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e9;
    std.debug.print("best weights: {s}\nelapsed {d:.1} s\nholdout: zig build evalab -Doptimize=ReleaseFast -- \"baseline;{s}\" 5 40 {d} 9000\n", .{ best_spec, elapsed, best_spec, depth });
}

test "dimIndex maps the optimised weight names" {
    try std.testing.expectEqual(@as(?usize, 0), dimIndex("order_penalty"));
    try std.testing.expectEqual(@as(?usize, 1), dimIndex("order_penalty_late"));
    try std.testing.expectEqual(@as(?usize, 2), dimIndex("merge"));
    try std.testing.expectEqual(@as(?usize, 3), dimIndex("monotonic"));
    try std.testing.expectEqual(@as(?usize, null), dimIndex("order"));
}

test "weights keeps the baseline values the search does not move" {
    const w = weights(.{ 50, 25, 10, 32 });
    try std.testing.expectEqual(@as(f64, 50), w.order_penalty);
    try std.testing.expectEqual(@as(f64, 25), w.order_penalty_late);
    try std.testing.expectEqual(@as(f64, 10), w.merge);
    try std.testing.expectEqual(@as(f64, 32), w.monotonic);
    try std.testing.expectEqual(solver.baseline.empty, w.empty);
    try std.testing.expectEqual(solver.baseline.forced_down, w.forced_down);
    try std.testing.expectEqual(solver.baseline.layout, w.layout);
}

test "spec prints every tuned weight as an evalab key" {
    var buf: [128]u8 = undefined;
    try std.testing.expectEqualStrings("order=43.174,order_late=25.000,merge=10.000,monotonic=32.000", spec(solver.baseline, &buf));
}
