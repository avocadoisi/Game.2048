const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate] [seed_ranges|diag]
    \\
    \\  train_games   TD(0) self-play games; 0 loads ai/solver/nn_weights.bin (default 2000)
    \\  eval_games    games per seed range per policy, same seeds for every policy (default 100)
    \\  eval_seed     first eval seed; range r, game g uses eval_seed + r*eval_games + g (default 1000)
    \\  learning_rate SGD step for TD training (default 0.003)
    \\  seed_ranges   number of disjoint seed ranges; report mean ± stderr over them (default 1)
    \\  diag          literal "diag": print the TD error / value drop table by empty cells and moves to game over
    \\
    \\Training seed is fixed (1). Weights are written to ai/solver/nn_weights.bin only when
    \\train_games > 0; train_games = 0 (eval only) never writes.
    \\
;

const weights_path = "ai/solver/nn_weights.bin";
const train_seed: u64 = 1;

const Eval = struct {
    name: []const u8,
    range_scores: []f64 = &.{},
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    elapsed_ns: i96 = 0,
};

fn printDiag(diag: *const nn.Diag) void {
    const empty_names = [_][]const u8{ "0..3", "4..7", "8+" };
    const dist_names = [_][]const u8{ "last 10", "11..30", "earlier" };
    std.debug.print("diag: |TD error| and value drop V(s')-V(s) (mean per step)\n", .{});
    std.debug.print("{s:<10} {s:<9} {s:>10} {s:>12} {s:>12} {s:>12}\n", .{ "empty", "moves", "steps", "mean |err|", "mean drop", "share err" });
    var total_err: f64 = 0;
    for (diag.stats) |row| for (row) |s| {
        total_err += s.abs_err;
    };
    for (diag.stats, 0..) |row, e| for (row, 0..) |s, d| {
        const n: f64 = @floatFromInt(@max(s.count, 1));
        std.debug.print("{s:<10} {s:<9} {d:>10} {d:>12.4} {d:>12.4} {d:>11.1}%\n", .{
            empty_names[e],
            dist_names[d],
            s.count,
            s.abs_err / n,
            s.drop / n,
            100 * s.abs_err / @max(total_err, 1e-300),
        });
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 2000;
    const eval_games: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 100;
    const eval_seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    const learning_rate: f64 = if (args.len > 4) try std.fmt.parseFloat(f64, args[4]) else 0.003;
    // arg 5 is either the seed-range count or the literal "diag"
    const arg5: ?[]const u8 = if (args.len > 5) args[5] else null;
    const want_diag = arg5 != null and std.mem.eql(u8, arg5.?, "diag");
    const seed_ranges: u32 = if (arg5) |a| (if (want_diag) 1 else try std.fmt.parseInt(u32, a, 10)) else 1;
    if (args.len > 6 or eval_games == 0 or seed_ranges == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var net: nn.Net = undefined;
    if (train_games == 0) {
        try nn.load(&net, init.io, init.gpa, weights_path);
        std.debug.print("loaded {s}\n", .{weights_path});
    } else {
        var init_rng = std.Random.DefaultPrng.init(train_seed);
        net = nn.Net.init(init_rng.random());
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        var last_score: u64 = 0;
        var diag: nn.Diag = .{};
        for (0..train_games) |g| {
            // linear decay to 10% of the start rate: big steps early, fine tuning late
            const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
            last_score = nn.trainGame(&net, rng.random(), learning_rate * (1 - 0.9 * frac), if (want_diag) &diag else null);
            if ((g + 1) % 5000 == 0) {
                std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
                try nn.save(&net, init.io, weights_path); // checkpoint
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("trained {d} games in {d:.1} s (lr={d}, seed={d})\n", .{
            train_games,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            learning_rate,
            train_seed,
        });
        if (want_diag) printDiag(&diag);
        try nn.save(&net, init.io, weights_path);
    }

    var results = [_]Eval{
        .{ .name = "nn 1-ply" },
        .{ .name = "nn + 1 spawn lookahead" },
        .{ .name = "expectimax depth 2" },
    };
    for (&results, 0..) |*r, idx| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        var total_moves: u64 = 0;
        const start = std.Io.Clock.awake.now(init.io);
        for (0..seed_ranges) |range| {
            var total_score: u64 = 0;
            for (0..eval_games) |g| {
                var prng = std.Random.DefaultPrng.init(eval_seed + range * eval_games + g);
                const res = switch (idx) {
                    0 => nn.play(&net, prng.random(), .greedy),
                    1 => nn.play(&net, prng.random(), .chance),
                    else => blk: {
                        const b = solver.playPolicy(prng.random(), .expectimax, 2);
                        break :blk nn.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves };
                    },
                };
                total_score += res.score;
                total_moves += res.moves;
                r.best_tile = @max(r.best_tile, res.max_tile);
            }
            r.range_scores[range] = @as(f64, @floatFromInt(total_score)) / @as(f64, @floatFromInt(eval_games));
        }
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const n: f64 = @floatFromInt(eval_games * seed_ranges);
        r.avg_moves = @as(f64, @floatFromInt(total_moves)) / n;
    }
    const last_seed = eval_seed + seed_ranges * eval_games - 1;
    std.debug.print("eval: games={d}/range ranges={d} seeds={d}..{d}\n", .{ eval_games, seed_ranges, eval_seed, last_seed });
    for (results) |r| {
        const st = nn.meanStderr(r.range_scores);
        std.debug.print("{s}: avg score={d:.1} ± {d:.1} (stderr over {d} ranges) avg moves={d:.1} best tile={d} elapsed={d:.1} s\n", .{
            r.name,
            st.mean,
            st.stderr,
            seed_ranges,
            r.avg_moves,
            r.best_tile,
            @as(f64, @floatFromInt(r.elapsed_ns)) / 1e9,
        });
    }
}
