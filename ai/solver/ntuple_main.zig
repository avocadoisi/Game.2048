const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const ntuple = @import("ntuple");
const parallel = @import("parallel");

const usage =
    \\usage: ntuple [train_games] [eval_games] [eval_seed] [learning_rate] [variant] [table_path] [train_seed]
    \\
    \\  train_games   TD(0) self-play games; 0 loads table_path (default 2000)
    \\  eval_games    games per policy on the eval seed range (default 100)
    \\  eval_seed     first eval seed; game i uses eval_seed + i (default 1000)
    \\  learning_rate TD step for n-tuple tables (default 0.01)
    \\  variant       base (greedy 1-ply) or chance (1 spawn lookahead); deep (2-ply) adds a
    \\                second own move. light / lightdeep use only the 5 base 4-tuples.
    \\                wide / widedeep add the second 2x3 6-tuple (both 2x3 orbits, #106).
    \\                maxb / maxbchance / maxbdeep are wide with the max-backup TD target (#131).
    \\                resume loads table_path first, then trains train_games more games.
    \\  table_path    file to load (train_games 0) or write (default ai/solver/ntuple_weights.bin);
    \\                a path ending in .q16.bin reads/writes the int16 quantized format (#129)
    \\  train_seed    seed for self-play games (default 1)
    \\
    \\The table is written to `table_path` only when train_games > 0.
    \\
;

const default_table_path = "ai/solver/ntuple_weights.bin";

/// Positional arg 7, after `table_path`.
fn parseTrainSeed(args: []const []const u8) !u64 {
    return if (args.len > 7) try std.fmt.parseInt(u64, args[7], 10) else 1;
}

const Eval = struct {
    name: []const u8,
    range_scores: []f64 = &.{},
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    total_score: u64 = 0,
    leaves: u64 = 0,
    elapsed_ns: i96 = 0,
};

const MeanErr = struct { mean: f64, stderr: f64 };

/// Same shape as `nn.meanStderr`: mean and sample standard error of the seed-range averages.
fn meanStderr(xs: []const f64) MeanErr {
    const n: f64 = @floatFromInt(xs.len);
    var sum: f64 = 0;
    for (xs) |x| sum += x;
    const mean = sum / n;
    if (xs.len < 2) return .{ .mean = mean, .stderr = 0 };
    var ss: f64 = 0;
    for (xs) |x| ss += (x - mean) * (x - mean);
    return .{ .mean = mean, .stderr = @sqrt(ss / (n - 1) / n) };
}

fn scorePerSecond(total_score: f64, elapsed_ns: i96) f64 {
    if (elapsed_ns <= 0) return 0;
    return total_score / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 2000;
    const eval_games: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 100;
    const eval_seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    const learning_rate: f64 = if (args.len > 4) try std.fmt.parseFloat(f64, args[4]) else 0.01;
    const variant_name = if (args.len > 5) args[5] else "base";
    const variant = ntuple.parseVariant(variant_name) orelse {
        std.debug.print("{s}", .{usage});
        return;
    };
    const mode = variant.mode;
    const table_path = if (args.len > 6) args[6] else default_table_path;
    const train_seed = try parseTrainSeed(args);
    if (args.len > 8 or eval_games == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var net = try ntuple.TupleNet.initWith(init.gpa, variant.set);
    defer net.deinit(init.gpa);
    std.debug.print("table: variant={s} patterns={d} entries={d} raw={d:.1} MiB\n", .{
        variant_name,
        net.count,
        net.storage.len,
        @as(f64, @floatFromInt(net.storage.len * @sizeOf(f32))) / (1024 * 1024),
    });

    // A `.q16.bin` path writes (or, for an eval-only run, reads) the int16 quantized file (#129).
    const quantized_path = std.mem.endsWith(u8, table_path, ".q16.bin");
    if (train_games == 0 or variant.resume_from_file) {
        if (quantized_path) {
            var q = try ntuple.loadQuantized(init.io, init.gpa, table_path);
            defer q.deinit(init.gpa);
            ntuple.dequantizeInto(&q, &net);
        } else try ntuple.load(&net, init.io, init.gpa, table_path);
        std.debug.print("loaded {s}\n", .{table_path});
    } else {
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        var total_moves: u64 = 0;
        var last_score: u32 = 0;
        for (0..train_games) |g| {
            // linear decay to 10% of the start rate, same schedule as nn_main
            const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
            const game = ntuple.trainGameWith(&net, rng.random(), learning_rate * (1 - 0.9 * frac), mode, variant.target);
            last_score = game.score;
            total_moves += game.moves;
            if ((g + 1) % 5000 == 0) {
                std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("trained {d} games, {d} moves in {d:.1} s (lr={d}, seed={d}, variant={s})\n", .{
            train_games,
            total_moves,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            learning_rate,
            train_seed,
            variant_name,
        });
        if (quantized_path) {
            var q = try ntuple.quantize(init.gpa, &net);
            defer q.deinit(init.gpa);
            try ntuple.saveQuantized(&q, init.io, init.gpa, table_path);
            std.debug.print("saved {s} (quantized int16)\n", .{table_path});
        } else try ntuple.save(&net, init.io, init.gpa, table_path);
    }

    var results = [_]Eval{
        .{ .name = "ntuple 1-ply" },
        .{ .name = "ntuple + 1 spawn lookahead" },
        .{ .name = "ntuple depth 2" },
        .{ .name = "expectimax depth 2" },
    };
    const seed_ranges: u32 = 1;
    const games = eval_games * seed_ranges;
    const EvalCtx = struct { net: *const ntuple.TupleNet, policy: usize, eval_seed: u64 };
    const playGame = struct {
        fn f(c: EvalCtx, i: usize) ntuple.PlayResult {
            var prng = std.Random.DefaultPrng.init(c.eval_seed + i);
            return switch (c.policy) {
                0 => ntuple.play(c.net, prng.random(), .greedy),
                1 => ntuple.play(c.net, prng.random(), .chance),
                2 => ntuple.play(c.net, prng.random(), .deep),
                else => blk: {
                    const b = solver.playPolicy(prng.random(), .expectimax, 2);
                    break :blk ntuple.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves };
                },
            };
        }
    }.f;
    for (&results, 0..) |*r, idx| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        const res_all = try init.arena.allocator().alloc(ntuple.PlayResult, games);
        const start = std.Io.Clock.awake.now(init.io);
        parallel.map(ntuple.PlayResult, res_all, EvalCtx{ .net = &net, .policy = idx, .eval_seed = eval_seed }, playGame);
        var total_moves: u64 = 0;
        for (0..seed_ranges) |range| {
            var total_score: u64 = 0;
            for (res_all[range * eval_games ..][0..eval_games]) |res| {
                total_score += res.score;
                r.total_score += res.score;
                r.leaves += res.leaves;
                total_moves += res.moves;
                r.best_tile = @max(r.best_tile, res.max_tile);
            }
            r.range_scores[range] = @as(f64, @floatFromInt(total_score)) / @as(f64, @floatFromInt(eval_games));
        }
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        r.avg_moves = @as(f64, @floatFromInt(total_moves)) / @as(f64, @floatFromInt(games));
    }
    const last_seed = eval_seed + seed_ranges * eval_games - 1;
    std.debug.print("eval: games={d}/range ranges={d} seeds={d}..{d}\n", .{ eval_games, seed_ranges, eval_seed, last_seed });
    for (results) |r| {
        const st = meanStderr(r.range_scores);
        const total_moves = r.avg_moves * @as(f64, @floatFromInt(games));
        const ms_per_move = @as(f64, @floatFromInt(r.elapsed_ns)) / 1e6 / @max(total_moves, 1);
        const leaves_per_move = @as(f64, @floatFromInt(r.leaves)) / @max(total_moves, 1);
        std.debug.print("{s}: avg score={d:.1} ± {d:.1} (stderr over {d} ranges) avg moves={d:.1} best tile={d} elapsed={d:.1} s score/s={d:.1} leaves={d} leaves/move={d:.1} ms/move={d:.2}\n", .{
            r.name,
            st.mean,
            st.stderr,
            seed_ranges,
            r.avg_moves,
            r.best_tile,
            @as(f64, @floatFromInt(r.elapsed_ns)) / 1e9,
            scorePerSecond(@floatFromInt(r.total_score), r.elapsed_ns),
            r.leaves,
            leaves_per_move,
            ms_per_move,
        });
    }
}

test "train seed argument is optional and defaults to 1" {
    const no_args = [_][]const u8{"ntuple"};
    try std.testing.expectEqual(@as(u64, 1), try parseTrainSeed(&no_args));
    const path_only = [_][]const u8{ "ntuple", "0", "20", "2000", "0.01", "base", "x.bin" };
    try std.testing.expectEqual(@as(u64, 1), try parseTrainSeed(&path_only));
    const with_seed = [_][]const u8{ "ntuple", "0", "20", "2000", "0.01", "base", "x.bin", "3" };
    try std.testing.expectEqual(@as(u64, 3), try parseTrainSeed(&with_seed));
}

test "usage text lists both variants" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "base (greedy") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "chance (1 spawn") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "lightdeep") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "widedeep") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "resume") != null);
}
