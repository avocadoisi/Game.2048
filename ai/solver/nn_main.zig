const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");
const parallel = @import("parallel");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate] [seed_ranges|diag] [variant|par] [replay_k] [weights_path] [train_seed] [par]
    \\
    \\  train_games   TD(0) self-play games; 0 loads ai/solver/nn_weights.bin (default 2000)
    \\  eval_games    games per seed range per policy, same seeds for every policy (default 100)
    \\  eval_seed     first eval seed; range r, game g uses eval_seed + r*eval_games + g (default 1000)
    \\  learning_rate SGD step for TD training (default 0.003)
    \\  seed_ranges   number of disjoint seed ranges; report mean ± stderr over them (default 1)
    \\  diag          literal "diag": print the TD error / value drop table by empty cells and moves to game over; "-" for none
    \\  variant       base (default), backward (sweep stored afterstates last to first), replay (re-train the last replay_k),
    \\              priority (#34: 32 extra updates per game sampled from a |TD error| buffer, 50% uniform)
    \\              midgame (#59: 32 extra updates per game from buffer entries with 4..7 empty cells, uniform)
    \\              midmix  (#63: same but 50% uniform over the whole buffer, 50% |TD error| in the 4..7 bucket)
    \\              sym     (#72: every per-move update applied to all 8 dihedral transforms of the afterstate)
    \\              symlam  (#72: sym plus TD(lambda) eligibility traces with lambda = 0.7)
    \\              symbatch (#98: the 8 transformed gradients from one net snapshot, averaged and
    \\                       norm-clipped, applied as a single step instead of eight)
    \\              nstep   (#84: online n-step TD with n = 4 instead of one-step returns)
    \\              death   (#101: reward -1 per move so the value learns V ≈ -(moves to game over))
    \\              resume  (#66: load weights_path and train train_games more games from those weights, plain per-move path)
    \\  par           literal "par": train self-play in parallel batches of min(cpu count, 8) games;
    \\                game i uses seed train_seed + i and the batch result is the mean of its games.
    \\                (#99) as arg 10 it batches any stateless variant (base|backward|replay|sym|symlam|
    \\                nstep|resume); as arg 6 it keeps its old meaning, the plain per-move variant only,
    \\                so replay_k is ignored there but its slot must be filled to reach weights_path.
    \\                Not supported for priority|midgame|midmix (cross-game replay buffer)
    \\  replay_k      afterstates re-trained per game for variant replay (default 30)
    \\  weights_path  file to load (train_games 0) or write (default ai/solver/nn_weights.bin); use a temp path for variant runs
    \\  train_seed    seed for weight init and self-play games (default 1, omitted = same weights as before #56)
    \\
    \\Weights are written to `weights_path` only when train_games > 0;
    \\train_games = 0 (eval only) never writes.
    \\
;

const default_weights_path = "ai/solver/nn_weights.bin";

/// Positional arg 9, appended after `weights_path` so existing command lines keep their meaning.
fn parseTrainSeed(args: []const []const u8) !u64 {
    return if (args.len > 9) try std.fmt.parseInt(u64, args[9], 10) else 1;
}

/// Positional arg 10, appended after `train_seed`: the literal "par" runs the variant of arg 6 in
/// parallel batches (issue #99). Arg 6 `par` keeps its meaning (base + parallel).
fn parseParFlag(args: []const []const u8) !bool {
    if (args.len <= 10) return false;
    if (!std.mem.eql(u8, args[10], "par")) return error.InvalidParFlag;
    return true;
}

/// Starting net for `main`: `weights_path` when resuming (or eval-only), otherwise a random init
/// from `train_seed`. Returns whether the file was loaded.
fn startNet(net: *nn.Net, io: std.Io, gpa: std.mem.Allocator, train_games: u32, from_file: bool, train_seed: u64, weights_path: []const u8) !bool {
    if (train_games == 0 or from_file) {
        try nn.load(net, io, gpa, weights_path);
        return true;
    }
    var init_rng = std.Random.DefaultPrng.init(train_seed);
    net.* = nn.Net.init(init_rng.random());
    return false;
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
    // arg 5 is either the seed-range count or the literal "diag" ("-" = no diag, 1 range)
    const arg5: ?[]const u8 = if (args.len > 5) args[5] else null;
    const want_diag = arg5 != null and std.mem.eql(u8, arg5.?, "diag");
    const no_diag = arg5 != null and std.mem.eql(u8, arg5.?, "-");
    const seed_ranges: u32 = if (arg5) |a| (if (want_diag or no_diag) 1 else try std.fmt.parseInt(u32, a, 10)) else 1;
    const variant = if (args.len > 6) args[6] else "base";
    // arg 6 `par` means base + parallel (unchanged); arg 10 `par` batches the variant of arg 6
    const par_variant = std.mem.eql(u8, variant, "par");
    const par_tail = try parseParFlag(args);
    const use_par = par_variant or par_tail;
    const use_resume = std.mem.eql(u8, variant, "resume");
    const replay_k: usize = if (args.len > 7) try std.fmt.parseInt(usize, args[7], 10) else 30;
    const weights_path = if (args.len > 8) args[8] else default_weights_path;
    const train_seed = try parseTrainSeed(args);
    var priority: nn.Priority = .{};
    const opts: nn.Options = if (par_variant or use_resume or std.mem.eql(u8, variant, "base"))
        .{}
    else if (std.mem.eql(u8, variant, "backward"))
        .{ .backward = true }
    else if (std.mem.eql(u8, variant, "replay"))
        .{ .replay_k = replay_k }
    else if (std.mem.eql(u8, variant, "priority"))
        .{ .priority = &priority }
    else if (std.mem.eql(u8, variant, "midgame"))
        .{ .priority = &priority, .midgame = true }
    else if (std.mem.eql(u8, variant, "midmix"))
        .{ .priority = &priority, .midmix = true }
    else if (std.mem.eql(u8, variant, "sym"))
        .{ .symmetry = true }
    else if (std.mem.eql(u8, variant, "symlam"))
        .{ .symmetry = true, .lambda = 0.7 }
    else if (std.mem.eql(u8, variant, "symbatch"))
        .{ .sym_batch = true }
    else if (std.mem.eql(u8, variant, "nstep"))
        .{ .nstep = 4 }
    else if (std.mem.eql(u8, variant, "death"))
        .{ .death = true }
    else {
        std.debug.print("{s}", .{usage});
        return;
    };
    if (args.len > 11 or eval_games == 0 or seed_ranges == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // the priority ring buffer is cross-game state shared by pointer: batching would race on it
    if (use_par and opts.priority != null) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var net: nn.Net = undefined;
    if (try startNet(&net, init.io, init.gpa, train_games, use_resume, train_seed, weights_path))
        std.debug.print("loaded {s}\n", .{weights_path});
    if (train_games > 0) {
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        var last_score: u64 = 0;
        var total_moves: u64 = 0;
        var diag: nn.Diag = .{};
        if (use_par) {
            const batch_size = nn.trainBatchSize();
            var g: usize = 0;
            while (g < train_games) : (g += @min(batch_size, train_games - g)) {
                const games = @min(batch_size, train_games - g);
                const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
                last_score = nn.trainBatchOpts(&net, train_seed, g, games, learning_rate * (1 - 0.9 * frac), if (want_diag) &diag else null, opts);
                if ((g + games) % 5000 == 0) {
                    std.debug.print("train {d}/{d} last game score {d}\n", .{ g + games, train_games, last_score });
                    try nn.save(&net, init.io, weights_path); // checkpoint
                }
            }
        } else {
            for (0..train_games) |g| {
                // linear decay to 10% of the start rate: big steps early, fine tuning late
                const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
                const game = nn.trainGameOpts(&net, rng.random(), learning_rate * (1 - 0.9 * frac), if (want_diag) &diag else null, opts);
                last_score = game.score;
                total_moves += game.moves;
                if ((g + 1) % 5000 == 0) {
                    std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
                    try nn.save(&net, init.io, weights_path); // checkpoint
                }
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const variant_label = if (par_tail) try std.fmt.allocPrint(init.arena.allocator(), "{s}+par", .{variant}) else variant;
        std.debug.print("trained {d} games, {d} moves in {d:.1} s (lr={d}, seed={d}, variant={s})\n", .{
            train_games,
            total_moves,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            learning_rate,
            train_seed,
            variant_label,
        });
        if (want_diag) printDiag(&diag);
        try nn.save(&net, init.io, weights_path);
    }

    var results = [_]Eval{
        .{ .name = "nn 1-ply" },
        .{ .name = "nn + 1 spawn lookahead" },
        .{ .name = "nn 2-ply (deep)" },
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
                2 => nn.play(c.net, prng.random(), .deep),
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
                r.leaves += res.leaves;
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
            nn.scorePerSecond(@floatFromInt(r.total_score), r.elapsed_ns),
            r.leaves,
            leaves_per_move,
            ms_per_move,
        });
    }
}

test "train seed argument is optional and defaults to 1" {
    const no_args = [_][]const u8{"nn"};
    try std.testing.expectEqual(@as(u64, 1), try parseTrainSeed(&no_args));
    const nine_args = [_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin" };
    try std.testing.expectEqual(@as(u64, 1), try parseTrainSeed(&nine_args));
    const with_seed = [_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3" };
    try std.testing.expectEqual(@as(u64, 3), try parseTrainSeed(&with_seed));
    try std.testing.expectError(error.InvalidCharacter, parseTrainSeed(&[_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin", "x" }));
}

/// Test-only replica of the serial training setup in `main` (init + game RNG both from the seed).
fn trainSeedWeights(seed: u64, games: usize) nn.Net {
    var init_rng = std.Random.DefaultPrng.init(seed);
    var net = nn.Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(seed);
    for (0..games) |_| _ = nn.trainGame(&net, rng.random(), 0.003, null);
    return net;
}

test "omitted train seed trains seed 1 and matches the recorded checksum" {
    const no_args = [_][]const u8{"nn"};
    const net_default = trainSeedWeights(try parseTrainSeed(&no_args), 5);
    const net_seed1 = trainSeedWeights(1, 5);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&net_seed1), std.mem.asBytes(&net_default));
    try std.testing.expectEqual(@as(u32, 4199351724), std.hash.Crc32.hash(std.mem.asBytes(&net_default)));
}

test "different train seeds produce different weights" {
    const net1 = trainSeedWeights(1, 5);
    const net2 = trainSeedWeights(2, 5);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&net1), std.mem.asBytes(&net2)));
}

test "resume starts from the saved weights file instead of a random init" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const path = "/tmp/nn_resume_test.bin";

    var init_rng = std.Random.DefaultPrng.init(7);
    const saved = nn.Net.init(init_rng.random());
    try nn.save(&saved, io, path);

    // resume: the net comes out of weights_path byte-for-byte, before any update
    var resumed: nn.Net = undefined;
    try std.testing.expect(try startNet(&resumed, io, std.testing.allocator, 5, true, 7, path));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&saved), std.mem.asBytes(&resumed));

    // default: train_games > 0 without resume still starts from a random init (seed 99, so it must differ)
    var fresh: nn.Net = undefined;
    try std.testing.expect(!try startNet(&fresh, io, std.testing.allocator, 5, false, 99, path));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&saved), std.mem.asBytes(&fresh)));
}

test "usage text lists the resume variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "resume  (#66") != null);
}

test "usage text lists the nstep variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "nstep   (#84") != null);
}

test "usage text lists the death variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "death   (#101") != null);
}

test "par flag is off without arg 10 and on with the literal par" {
    const ten_args = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "sym", "30", "x.bin", "3" };
    try std.testing.expect(!try parseParFlag(&ten_args));
    const eleven_args = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "sym", "30", "x.bin", "3", "par" };
    try std.testing.expect(try parseParFlag(&eleven_args));
    const typo = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "sym", "30", "x.bin", "3", "pa" };
    try std.testing.expectError(error.InvalidParFlag, parseParFlag(&typo));
}

test "usage text documents the par flag for stateless variants" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "#99") != null);
}

test "usage text lists the symbatch variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "symbatch (#98") != null);
}
