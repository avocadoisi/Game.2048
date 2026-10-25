const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate] [diag] [variant] [replay_k] [weights_path]
    \\
    \\  train_games   TD(0) self-play games; 0 loads ai/solver/nn_weights.bin (default 2000)
    \\  eval_games    games per policy, same seeds for every policy (default 100)
    \\  eval_seed     first eval seed; game g uses eval_seed + g (default 1000)
    \\  learning_rate SGD step for TD training (default 0.003)
    \\  diag          literal "diag": print the TD error / value drop table by empty cells and moves to game over; "-" for none
    \\  variant       base (default), backward (sweep stored afterstates last to first), replay (re-train the last replay_k)
    \\  replay_k      afterstates re-trained per game for variant replay (default 30)
    \\  weights_path  file to load (train_games 0) or write (default ai/solver/nn_weights.bin); use a temp path for variant runs
    \\
    \\Training seed is fixed (1).
    \\
;

const default_weights_path = "ai/solver/nn_weights.bin";
const train_seed: u64 = 1;

const Eval = struct {
    name: []const u8,
    avg_score: f64 = 0,
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
    const want_diag = args.len > 5 and std.mem.eql(u8, args[5], "diag");
    const variant = if (args.len > 6) args[6] else "base";
    const replay_k: usize = if (args.len > 7) try std.fmt.parseInt(usize, args[7], 10) else 30;
    const weights_path = if (args.len > 8) args[8] else default_weights_path;
    const opts: nn.Options = if (std.mem.eql(u8, variant, "base"))
        .{}
    else if (std.mem.eql(u8, variant, "backward"))
        .{ .backward = true }
    else if (std.mem.eql(u8, variant, "replay"))
        .{ .replay_k = replay_k }
    else {
        std.debug.print("{s}", .{usage});
        return;
    };
    if (args.len > 9 or (args.len > 5 and !want_diag and !std.mem.eql(u8, args[5], "-")) or eval_games == 0) {
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
        var total_moves: u64 = 0;
        var diag: nn.Diag = .{};
        for (0..train_games) |g| {
            const game = nn.trainGameOpts(&net, rng.random(), learning_rate, if (want_diag) &diag else null, opts);
            last_score = game.score;
            total_moves += game.moves;
            if ((g + 1) % 500 == 0) std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("trained {d} games, {d} moves in {d:.1} s (lr={d}, seed={d}, variant={s})\n", .{
            train_games,
            total_moves,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            learning_rate,
            train_seed,
            variant,
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
        var total_score: u64 = 0;
        var total_moves: u64 = 0;
        const start = std.Io.Clock.awake.now(init.io);
        for (0..eval_games) |g| {
            var prng = std.Random.DefaultPrng.init(eval_seed + g);
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
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        r.avg_score = @as(f64, @floatFromInt(total_score)) / @as(f64, @floatFromInt(eval_games));
        r.avg_moves = @as(f64, @floatFromInt(total_moves)) / @as(f64, @floatFromInt(eval_games));
    }
    std.debug.print("eval: games={d} seeds={d}..{d}\n", .{ eval_games, eval_seed, eval_seed + eval_games - 1 });
    for (results) |r| {
        std.debug.print("{s}: avg score={d:.1} avg moves={d:.1} best tile={d} elapsed={d:.1} s\n", .{
            r.name,
            r.avg_score,
            r.avg_moves,
            r.best_tile,
            @as(f64, @floatFromInt(r.elapsed_ns)) / 1e9,
        });
    }
}
