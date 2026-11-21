const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const ntuple = @import("ntuple");
const parallel = @import("parallel");
const cli = @import("args.zig");
const bench = @import("bench.zig");
const stats = @import("stats");
const game_limit = @import("game_limit");

const usage =
    \\usage: ntuple [train_games] [eval_games] [eval_seed] [learning_rate] [variant] [table_path] [train_seed]
    \\
    \\  train_games   self-play training games; 0 loads table_path (default 2000)
    \\  eval_games    games per policy on the eval seed range (default 100)
    \\  eval_seed     first eval seed; game i uses eval_seed + i (default 1000)
    \\  learning_rate TD step size (or TC beta) for n-tuple training (default 0.01)
    \\  variant       base (greedy 1-ply) or chance (1 spawn lookahead); deep (2-ply) adds a
    \\                second own move. light / lightdeep use only the 5 base 4-tuples.
    \\                wide / widedeep add the second 2x3 6-tuple (both 2x3 orbits, #106).
    \\                widedelay opts into delayed TD(lambda) on wide tables (lambda=0.5, horizon=3, #290).
    \\                widetc / widedeep_tc opt into TC(lambda) on wide tables (lambda=0.5, horizon=3, #288).
    \\                maxb / maxbchance / maxbdeep are wide with the max-backup TD target (#131).
    \\                lshape / lshaperesume add one non-rectangular 6-cell L tuple, +64 MiB f32 (#323).
    \\                resume loads table_path first, then trains train_games more games.
    \\  table_path    file to load (train_games 0) or write (default ai/solver/ntuple_weights.bin);
    \\                experimental learners require a separate path with a distinct filename
    \\                a path ending in .q16.bin reads/writes the int16 quantized format (#129)
    \\  train_seed    seed for self-play games (default 1)
    \\
    \\  Time caps     60 s/game for training, 30 min/game for evaluation. Append +xtime to variant
    \\                for the single extended retry: 300 s / 120 min. Unfinished games are neither
    \\                saved nor included in a completed evaluation row.
    \\The table is written to `table_path` only when train_games > 0.
    \\
;

const default_table_path = "ai/solver/ntuple_weights.bin";

fn learnerTrainingPathAllowed(
    allocator: std.mem.Allocator,
    cwd: []const u8,
    learner: ntuple.Learner,
    train_games: u32,
    table_path: []const u8,
) !bool {
    if (train_games == 0 or learner == .td_zero) return true;
    const output_path = try std.fs.path.resolveAlloc(allocator, &.{ cwd, table_path });
    defer allocator.free(output_path);
    // Also block shipped basenames outside this worktree (e.g. an absolute path into the main checkout).
    for ([_][]const u8{ default_table_path, "ai/solver/ntuple_wide_q16.bin" }) |shipped| {
        if (std.ascii.eqlIgnoreCase(std.fs.path.basename(output_path), std.fs.path.basename(shipped))) return false;
        const shipped_path = try std.fs.path.resolveAlloc(allocator, &.{ cwd, shipped });
        defer allocator.free(shipped_path);
        if (std.mem.eql(u8, output_path, shipped_path)) return false;
    }
    return true;
}

// Positional defaults, one named constant each (`usage` above is the compatibility surface).
const default_train_games: u32 = 2000;
const default_eval_games: u32 = 100;
const default_eval_seed: u64 = 1000;
const default_learning_rate: f64 = 0.01;
/// Positional arg 7, after `table_path`, appended so existing command lines keep their meaning.
const default_train_seed: u64 = 1;

const Eval = struct {
    name: []const u8,
    range_scores: []f64 = &.{},
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    total_score: u64 = 0,
    game_stderr: f64 = 0,
    reach_8192: u32 = 0,
    leaves: u64 = 0,
    elapsed_ns: i96 = 0,
};

const ExperimentVariant = struct { variant: ntuple.Variant, extended: bool };

/// `variant+xtime` is an explicit single retry, not another training iteration (#336).
fn parseExperimentVariant(name: []const u8) ?ExperimentVariant {
    if (ntuple.parseVariant(name)) |variant| return .{ .variant = variant, .extended = false };
    if (std.mem.endsWith(u8, name, "+xtime")) {
        const variant = ntuple.parseVariant(name[0 .. name.len - "+xtime".len]) orelse return null;
        return .{ .variant = variant, .extended = true };
    }
    return null;
}

fn timeLimit(result: anytype, phase: game_limit.RunPhase, extended: bool) noreturn {
    const extend = game_limit.promising(result.score, result.max_tile, extended);
    const disposition = if (extended) "inconclusive-close" else if (extend) "extend-once" else "no-go-close";
    std.debug.print("{s} game hit the {d:.1}s time cap at {d} moves (partial score={d}, max_tile={d}), {s}; not saving or aggregating it\n", .{
        @tagName(phase), @as(f64, @floatFromInt(game_limit.limitNs(phase, extended))) / std.time.ns_per_s, result.moves, result.score, result.max_tile, disposition,
    });
    std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = try cli.argOr(u32, args, 1, default_train_games);
    const eval_games: u32 = try cli.argOr(u32, args, 2, default_eval_games);
    const eval_seed: u64 = try cli.argOr(u64, args, 3, default_eval_seed);
    const learning_rate: f64 = try cli.argOr(f64, args, 4, default_learning_rate);
    const variant_name: []const u8 = try cli.argOr([]const u8, args, 5, "base");
    const parsed = parseExperimentVariant(variant_name) orelse {
        std.debug.print("{s}", .{usage});
        return;
    };
    const variant = parsed.variant;
    const game_options: ntuple.GameOptions = .{ .game_time_limit_ns = game_limit.limitNs(if (train_games > 0) .train else .evaluate, parsed.extended) };
    const mode = variant.mode;
    const table_path: []const u8 = try cli.argOr([]const u8, args, 6, default_table_path);
    const train_seed: u64 = try cli.argOr(u64, args, 7, default_train_seed);
    if (args.len > 8 or eval_games == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }
    if (train_games > 0 and variant.learner != .td_zero) {
        const cwd = try std.process.currentPathAlloc(init.io, init.gpa);
        defer init.gpa.free(cwd);
        if (!try learnerTrainingPathAllowed(init.gpa, cwd, variant.learner, train_games, table_path)) {
            std.debug.print("Experimental learner training needs a candidate table_path with a distinct filename\n{s}", .{usage});
            return;
        }
    }

    var net = try ntuple.TupleNet.initWith(init.gpa, variant.set);
    defer net.deinit(init.gpa);
    std.debug.print("table: variant={s} patterns={d} entries={d} raw={d:.1} MiB\n", .{
        variant_name,
        net.count,
        net.storage.len,
        @as(f64, @floatFromInt(net.storage.len * @sizeOf(f32))) / (1024 * 1024),
    });
    if (train_games > 0 and variant.learner == .delayed_td_lambda) {
        std.debug.print("delayed TD(lambda) extra memory: {d} bytes (fixed h+1 afterstate/error queue)\n", .{
            ntuple.delayedTDLambdaExtraMemoryBytes(),
        });
    }

    var tc_state: ?ntuple.TemporalCoherence = null;
    defer {
        if (tc_state) |*tc| tc.deinit(init.gpa);
    }
    if (train_games > 0 and variant.learner == .temporal_coherence) {
        tc_state = try ntuple.TemporalCoherence.init(init.gpa, &net);
        const tc = tc_state.?;
        std.debug.print("temporal coherence extra memory: {d:.1} MiB (signed + absolute TD errors + episode trace)\n", .{
            @as(f64, @floatFromInt(tc.extraMemoryBytes())) / (1024 * 1024),
        });
    }

    // A `.q16.bin` path writes (or, for an eval-only run, reads) the int16 quantized file.
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
            const lr = learning_rate * (1 - 0.9 * frac);
            const game = switch (variant.learner) {
                .td_zero => ntuple.trainGameWithOpts(&net, rng.random(), lr, mode, variant.target, game_options),
                .delayed_td_lambda => ntuple.trainGameDelayedTDLambdaOpts(&net, rng.random(), lr, mode, game_options),
                .temporal_coherence => if (tc_state) |*tc|
                    ntuple.trainGameTemporalCoherenceOpts(tc, rng.random(), lr, mode, game_options)
                else
                    unreachable,
            };
            if (game.timed_out) timeLimit(game, .train, parsed.extended);
            last_score = game.score;
            total_moves += game.moves;
            if ((g + 1) % 5000 == 0) {
                std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        // Report throughput for every learner: the A/B of compares TD(0) and widedelay on games/s.
        const rate_seconds = @max(@as(f64, @floatFromInt(elapsed)) / 1e9, 1e-9);
        std.debug.print("trained {d} games, {d} moves in {d:.1} s ({d:.2} games/s, {d:.1} moves/s; lr={d}, seed={d}, variant={s})\n", .{
            train_games,
            total_moves,
            rate_seconds,
            @as(f64, @floatFromInt(train_games)) / rate_seconds,
            @as(f64, @floatFromInt(total_moves)) / rate_seconds,
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
    const EvalCtx = struct { net: *const ntuple.TupleNet, policy: usize, eval_seed: u64, options: ntuple.GameOptions };
    const playGame = struct {
        fn f(c: EvalCtx, i: usize) ntuple.PlayResult {
            var prng = std.Random.DefaultPrng.init(c.eval_seed + i);
            var game_stats: ntuple.Stats = .{};
            return switch (c.policy) {
                0 => ntuple.playWithStatsOpts(c.net, prng.random(), .greedy, &game_stats, c.options),
                1 => ntuple.playWithStatsOpts(c.net, prng.random(), .chance, &game_stats, c.options),
                2 => ntuple.playWithStatsOpts(c.net, prng.random(), .deep, &game_stats, c.options),
                else => blk: {
                    const b = solver.playPolicyOpts(prng.random(), .expectimax, 2, c.options);
                    break :blk ntuple.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves, .timed_out = b.timed_out };
                },
            };
        }
    }.f;
    for (&results, 0..) |*r, idx| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        const res_all = try init.arena.allocator().alloc(ntuple.PlayResult, games);
        const start = std.Io.Clock.awake.now(init.io);
        parallel.map(ntuple.PlayResult, res_all, EvalCtx{ .net = &net, .policy = idx, .eval_seed = eval_seed, .options = game_options }, playGame);
        for (res_all) |result| {
            if (result.timed_out) timeLimit(result, .evaluate, parsed.extended);
        }
        stats.rangeTotals(ntuple.PlayResult, Eval, r, res_all, eval_games);
        // `seed_ranges` is 1, so the between-range stderr above is always zero; a single-range row is
        // only judgeable with the per-game standard error.
        const game_scores = try init.arena.allocator().alloc(f64, games);
        for (res_all, 0..) |res, i| game_scores[i] = @as(f64, @floatFromInt(res.score));
        r.game_stderr = stats.meanStderr(game_scores).stderr;
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
    }
    const last_seed = eval_seed + seed_ranges * eval_games - 1;
    std.debug.print("eval: games={d}/range ranges={d} seeds={d}..{d}\n", .{ eval_games, seed_ranges, eval_seed, last_seed });
    for (results) |r| {
        const st = stats.meanStderr(r.range_scores);
        // Same row as `zig build nn`: `avg_score` holds the mean over seed ranges.
        bench.printEvalRow(r.name, st.stderr, seed_ranges, .{
            .games = games,
            .avg_score = st.mean,
            .avg_moves = r.avg_moves,
            .best_tile = r.best_tile,
            .elapsed_ns = r.elapsed_ns,
        }, r.total_score, r.leaves);
        std.debug.print("eval per-game score stderr={d:.1} (n={d}); reach8192={d}/{d} ({d:.1}%)\n\n", .{
            r.game_stderr,
            games,
            r.reach_8192,
            games,
            100 * @as(f64, @floatFromInt(r.reach_8192)) / @as(f64, @floatFromInt(games)),
        });
    }
}

test "experiment variants and time budgets agree with NN (#336)" {
    const base = parseExperimentVariant("base").?;
    try std.testing.expect(!base.extended);
    try std.testing.expect(parseExperimentVariant("base+xtime").?.extended);
    try std.testing.expect(parseExperimentVariant("base+xtime+xtime") == null);
    try std.testing.expectEqual(@as(u64, 60 * std.time.ns_per_s), game_limit.limitNs(.train, false));
    try std.testing.expectEqual(@as(u64, 300 * std.time.ns_per_s), game_limit.limitNs(.train, true));
    try std.testing.expectEqual(@as(u64, 30 * std.time.ns_per_min), game_limit.limitNs(.evaluate, false));
    try std.testing.expectEqual(@as(u64, 120 * std.time.ns_per_min), game_limit.limitNs(.evaluate, true));
    try std.testing.expect(game_limit.promising(100_000, 0, false));
    try std.testing.expect(!game_limit.promising(100_000, 0, true));
}

test "train seed argument (position 7) is optional and defaults to 1" {
    const no_args = [_][]const u8{"ntuple"};
    try std.testing.expectEqual(@as(u64, 1), try cli.argOr(u64, &no_args, 7, default_train_seed));
    const path_only = [_][]const u8{ "ntuple", "0", "20", "2000", "0.01", "base", "x.bin" };
    try std.testing.expectEqual(@as(u64, 1), try cli.argOr(u64, &path_only, 7, default_train_seed));
    const with_seed = [_][]const u8{ "ntuple", "0", "20", "2000", "0.01", "base", "x.bin", "3" };
    try std.testing.expectEqual(@as(u64, 3), try cli.argOr(u64, &with_seed, 7, default_train_seed));
}

test "experimental n-tuple learners cannot overwrite the shipped table" {
    const gpa = std.testing.allocator;
    const cwd = "/repo";
    const experimental = [_]ntuple.Learner{ .delayed_td_lambda, .temporal_coherence };
    const refused = [_][]const u8{
        default_table_path,
        "./ai/solver/ntuple_weights.bin",
        "ai/solver/../solver/ntuple_weights.bin",
        "/repo/ai/solver/ntuple_weights.bin",
        "ai/solver/ntuple_wide_q16.bin",
        "/other/repo/ai/solver/ntuple_weights.bin",
        "/tmp/ntuple_wide_q16.bin",
        "/tmp/NTUPLE_WEIGHTS.BIN",
    };
    for (experimental) |learner| {
        for (refused) |path| {
            try std.testing.expect(!(try learnerTrainingPathAllowed(gpa, cwd, learner, 1, path)));
        }
        // Loading never writes, and a distinct candidate filename is the intended output.
        try std.testing.expect(try learnerTrainingPathAllowed(gpa, cwd, learner, 0, default_table_path));
        try std.testing.expect(try learnerTrainingPathAllowed(gpa, cwd, learner, 1, "candidate.bin"));
    }
    try std.testing.expect(try learnerTrainingPathAllowed(gpa, cwd, .td_zero, 1, default_table_path));
}

test "usage text lists both variants" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "base (greedy") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "chance (1 spawn") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "lightdeep") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "widedeep") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "widedelay") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "widetc") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "widedeep_tc") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "lshaperesume") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "separate path with a distinct filename") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "resume") != null);
}
