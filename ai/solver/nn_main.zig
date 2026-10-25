const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");
const parallel = @import("parallel");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate] [seed_ranges]
    \\
    \\  train_games   TD(0) self-play games; 0 loads ai/solver/nn_weights.bin (default 2000)
    \\  eval_games    games per seed range per policy, same seeds for every policy (default 100)
    \\  eval_seed     first eval seed; range r, game g uses eval_seed + r*eval_games + g (default 1000)
    \\  learning_rate SGD step for TD training (default 0.003)
    \\  seed_ranges   number of disjoint seed ranges; report mean ± stderr over them (default 1)
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
    total_score: u64 = 0,
    elapsed_ns: i96 = 0,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 2000;
    const eval_games: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 100;
    const eval_seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1000;
    const learning_rate: f64 = if (args.len > 4) try std.fmt.parseFloat(f64, args[4]) else 0.003;
    const seed_ranges: u32 = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 1;
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
        for (0..train_games) |g| {
            last_score = nn.trainGame(&net, rng.random(), learning_rate);
            if ((g + 1) % 500 == 0) std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
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
    const games = eval_games * seed_ranges;
    const EvalCtx = struct { net: *const nn.Net, policy: usize, eval_seed: u64 };
    // Game i uses seed eval_seed + i (= eval_seed + range*eval_games + g); games run in parallel.
    const playGame = struct {
        fn f(c: EvalCtx, i: usize) nn.PlayResult {
            var prng = std.Random.DefaultPrng.init(c.eval_seed + i);
            return switch (c.policy) {
                0 => nn.play(c.net, prng.random(), .greedy),
                1 => nn.play(c.net, prng.random(), .chance),
                else => blk: {
                    const b = solver.playPolicy(prng.random(), .expectimax, 2);
                    break :blk nn.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves };
                },
            };
        }
    }.f;
    for (&results, 0..) |*r, idx| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        const res_all = try init.arena.allocator().alloc(nn.PlayResult, games);
        const start = std.Io.Clock.awake.now(init.io);
        parallel.map(nn.PlayResult, res_all, EvalCtx{ .net = &net, .policy = idx, .eval_seed = eval_seed }, playGame);
        var total_moves: u64 = 0;
        for (0..seed_ranges) |range| {
            var total_score: u64 = 0;
            for (res_all[range * eval_games ..][0..eval_games]) |res| {
                total_score += res.score;
                r.total_score += res.score;
                total_moves += res.moves;
                r.best_tile = @max(r.best_tile, res.max_tile);
            }
            r.range_scores[range] = @as(f64, @floatFromInt(total_score)) / @as(f64, @floatFromInt(eval_games));
        }
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const n: f64 = @floatFromInt(games);
        r.avg_moves = @as(f64, @floatFromInt(total_moves)) / n;
    }
    const last_seed = eval_seed + seed_ranges * eval_games - 1;
    std.debug.print("eval: games={d}/range ranges={d} seeds={d}..{d}\n", .{ eval_games, seed_ranges, eval_seed, last_seed });
    for (results) |r| {
        const st = nn.meanStderr(r.range_scores);
        std.debug.print("{s}: avg score={d:.1} ± {d:.1} (stderr over {d} ranges) avg moves={d:.1} best tile={d} elapsed={d:.1} s score/s={d:.1}\n", .{
            r.name,
            st.mean,
            st.stderr,
            seed_ranges,
            r.avg_moves,
            r.best_tile,
            @as(f64, @floatFromInt(r.elapsed_ns)) / 1e9,
            nn.scorePerSecond(@floatFromInt(r.total_score), r.elapsed_ns),
        });
    }
}
