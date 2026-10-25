const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate]
    \\
    \\  train_games   TD(0) self-play games; 0 loads ai/solver/nn_weights.bin (default 2000)
    \\  eval_games    games per policy, same seeds for every policy (default 100)
    \\  eval_seed     first eval seed; game g uses eval_seed + g (default 1000)
    \\  learning_rate SGD step for TD training (default 0.003)
    \\
    \\Training seed is fixed (1). Trained weights are written to ai/solver/nn_weights.bin.
    \\
;

const weights_path = "ai/solver/nn_weights.bin";
const train_seed: u64 = 1;

const Eval = struct {
    name: []const u8,
    avg_score: f64 = 0,
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    elapsed_ns: i96 = 0,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 2000;
    const eval_games: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 100;
    const eval_seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    const learning_rate: f64 = if (args.len > 4) try std.fmt.parseFloat(f64, args[4]) else 0.003;
    if (args.len > 5 or eval_games == 0) {
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
        for (0..train_games) |g| {
            // linear decay to 10% of the start rate: big steps early, fine tuning late
            const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
            last_score = nn.trainGame(&net, rng.random(), learning_rate * (1 - 0.9 * frac));
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
