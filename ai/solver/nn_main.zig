const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");
const parallel = @import("parallel");
const cli = @import("args.zig");

const usage =
    \\usage: nn [train_games] [eval_games] [eval_seed] [learning_rate] [seed_ranges|diag] [variant|par] [replay_k] [weights_path] [train_seed] [par|-] [dump_path]
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
    \\              t2      (#158: TD target = score delta + the 2-ply value of the post-spawn board,
    \\                       i.e. what the eval search looks at, instead of V(next afterstate))
    \\              p2      (#158: self-play move = the eval's 2-ply search (deep) instead of the 1-ply argmax)
    \\              tp2     (#158: t2 + p2, both sides aligned with the 2-ply eval)
    \\              d2      (#167: distillation, 2-ply teacher: the self-play move is the deep search's
    \\                       and the TD target is the search value of the post-spawn board)
    \\              d3      (#167: distillation, 3-ply teacher: same, from the deep3 tree (1e-2 spawn cut),
    \\                       so a 2-ply eval learns what the deeper search thinks the position is worth)
    \\              oi / oi16 (#107: multiply the initial weights by 4 / 16 = optimistic initialization,
    \\                       then train exactly like base; only the init differs, so `par` works)
    \\              resume  (#66: load weights_path and train train_games more games from those weights, plain per-move path)
    \\              mstd    (#107 MS-TD: one MLP per empty-cell stage (0..2, 3..4, 5..6, 7..8, 9..10,
    \\                       11..12, 13..16 empties); each TD step trains only the stage of the afterstate
    \\                       it reached. Weights go to a staged file (its own magic), so eval needs the same file)
    \\              dump    (#159: trains from weights_path like resume and also stores every afterstate
    \\                       with max tile >= 4096 into a bounded buffer, written to dump_path at the end)
    \\              ft2     (#159: fine tuning from dump_path, loaded like resume; every self-play step also
    \\                       trains one uniformly drawn deep afterstate, so 50% of the updates stay on the
    \\                       self-play trajectory (shallow positions included) and 50% land on the deep tail)
    \\              restart (#159: fine tuning that starts every self-play game on a uniformly drawn deep
    \\                       afterstate of dump_path plus one fresh spawn; otherwise trains like resume)
    \\  par           literal "par": train self-play in parallel batches of min(cpu count, 8) games;
    \\                game i uses seed train_seed + i and the batch result is the mean of its games.
    \\                (#99) as arg 10 it batches any stateless variant (base|backward|replay|sym|symlam|
    \\                nstep|resume); as arg 6 it keeps its old meaning, the plain per-move variant only,
    \\                so replay_k is ignored there but its slot must be filled to reach weights_path.
    \\                Not supported for priority|midgame|midmix (cross-game replay buffer)
    \\                or t2|p2|tp2 (#158: the extra search per move is the whole cost, batching it too
    \\                would only multiply the work), or d2|d3 (#167: the teacher search is the whole cost). "-" here means the same as leaving it out, which
    \\                is how dump_path is reached.
    \\  replay_k      afterstates re-trained per game for variant replay (default 30)
    \\  weights_path  file to load (train_games 0) or write (default ai/solver/nn_weights.bin); use a temp path for variant runs
    \\  train_seed    seed for weight init and self-play games (default 1, omitted = same weights as before #56)
    \\  dump_path     file for the deep-position buffer (#159); required by dump|ft2|restart, after a "-" par slot
    \\  batch_scale   (#137) multiplier on the merged batch delta for `par` runs (default 1 = mean;
    \\                `8` = linear scaling rule, `2.8284` = sqrt). Sits after the "-" par dump slot
    \\
    \\Weights are written to `weights_path` only when train_games > 0;
    \\train_games = 0 (eval only) never writes.
    \\
;

const default_weights_path = "ai/solver/nn_weights.bin";

// Positional defaults, one named constant each (`usage` above is the compatibility surface).
const default_train_games: u32 = 2000;
const default_eval_games: u32 = 100;
const default_eval_seed: u64 = 1000;
const default_learning_rate: f64 = 0.003;
const default_replay_k: usize = 30;
const default_seed_ranges: u32 = 1;
/// Appended after `weights_path` so existing command lines keep their meaning.
const default_train_seed: u64 = 1;

/// Positional arg 10, appended after `train_seed`: the literal "par" runs the variant of arg 6 in
/// parallel batches (issue #99). "-" is the explicit "no par" placeholder that opens the dump slots
/// of #159; any other value is a typo.
/// (#137) Batch delta multiplier for `par` runs, arg 12 (after the `par` slot and its `-` dump slot).
/// Omitted = 1.0, which is the historical "mean of the per-game deltas" step.
fn parseBatchScale(args: []const []const u8) !f64 {
    if (args.len <= 12) return 1.0;
    if (std.mem.eql(u8, args[12], "-")) return 1.0;
    return std.fmt.parseFloat(f64, args[12]) catch error.InvalidBatchScale;
}

fn parseParFlag(args: []const []const u8) !bool {
    if (args.len <= 10) return false;
    if (std.mem.eql(u8, args[10], "-")) return false;
    if (!std.mem.eql(u8, args[10], "par")) return error.InvalidParFlag;
    return true;
}

/// `par` forks whole games onto threads, so it cannot carry the cross-game priority buffer (shared by
/// pointer), the staged net (no batch path, one net is ~1 MB), or the #158 2-ply variants and their #167
/// teacher cells (their search is the whole cost of the run; batching would only multiply it by the
/// thread count).
fn parSupported(opts: nn.Options, is_mstd: bool) bool {
    return opts.priority == null and !is_mstd and opts.target == .td1 and opts.selfplay == .greedy1 and opts.teacher == null;
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

/// Starting staged value for `mstd`: `weights_path` when resuming (train_games 0) or otherwise 7 random
/// inits from `train_seed`. Returns whether the file was loaded.
fn startStaged(staged: *nn.Staged, io: std.Io, gpa: std.mem.Allocator, train_games: u32, train_seed: u64, weights_path: []const u8) !bool {
    if (train_games == 0) {
        try staged.loadFile(io, gpa, weights_path);
        return true;
    }
    var init_rng = std.Random.DefaultPrng.init(train_seed);
    staged.* = nn.Staged.init(init_rng.random());
    return false;
}

const Eval = struct {
    name: []const u8,
    /// Which entry of the `playGame` switch this row runs. Set from the row table below so a
    /// measurement run can pick a subset (see `selectedRows`) instead of paying for all nine.
    policy: usize = 0,
    range_scores: []f64 = &.{},
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    total_score: u64 = 0,
    leaves: u64 = 0,
    elapsed_ns: i96 = 0,
    /// Games whose final `max_tile` is at or above the tile (counts, not rates); the avg score is a
    /// central measure, so how often a run actually gets there has to be counted separately (#156).
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    /// Games that ended below 2048, i.e. died early.
    dead_below_2048: u32 = 0,
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
    printTileDiag(diag);
}

/// Max-tile half of the diagnostic (#159): how many of the trained afterstates ever reach the deep tail.
fn printTileDiag(diag: *const nn.Diag) void {
    const tile_names = [_][]const u8{ "<=256", "512", "1024", "2048", "4096", "8192+" };
    var total: u64 = 0;
    for (diag.tiles) |s| total += s.count;
    std.debug.print("diag: steps by max tile of the trained afterstate\n", .{});
    std.debug.print("{s:<10} {s:>10} {s:>12} {s:>12}\n", .{ "max tile", "steps", "share steps", "mean |err|" });
    for (diag.tiles, 0..) |s, t| {
        const n: f64 = @floatFromInt(@max(s.count, 1));
        std.debug.print("{s:<10} {d:>10} {d:>11.1}% {d:>12.4}\n", .{
            tile_names[t],
            s.count,
            100 * @as(f64, @floatFromInt(s.count)) / @max(@as(f64, @floatFromInt(total)), 1e-300),
            s.abs_err / n,
        });
    }
}

/// Loads the #159 deep-position file for the variants that train from it. A missing, foreign or empty
/// dump is a command-line error (usage + exit 1), never a silent no-op.
fn loadDump(dump: *nn.Dump, io: std.Io, gpa: std.mem.Allocator, path: []const u8) void {
    dump.load(io, gpa, path) catch |err| {
        std.debug.print("error: cannot read dump {s}: {s}\n{s}", .{ path, @errorName(err), usage });
        std.process.exit(1);
    };
    if (dump.len == 0) {
        std.debug.print("error: dump {s} holds no deep afterstates\n{s}", .{ path, usage });
        std.process.exit(1);
    }
    std.debug.print("dump {s}: {d} deep afterstates\n", .{ path, dump.len });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const train_games: u32 = try cli.argOr(u32, args, 1, default_train_games);
    const eval_games: u32 = try cli.argOr(u32, args, 2, default_eval_games);
    const eval_seed: u64 = try cli.argOr(u64, args, 3, default_eval_seed);
    const learning_rate: f64 = try cli.argOr(f64, args, 4, default_learning_rate);
    // arg 5 is either the seed-range count or the literal "diag" ("-" = no diag, 1 range)
    const arg5: ?[]const u8 = if (args.len > 5) args[5] else null;
    const want_diag = arg5 != null and std.mem.eql(u8, arg5.?, "diag");
    const no_diag = arg5 != null and std.mem.eql(u8, arg5.?, "-");
    const seed_ranges: u32 = if (arg5) |a| (if (want_diag or no_diag) 1 else try std.fmt.parseInt(u32, a, 10)) else default_seed_ranges;
    const variant = if (args.len > 6) args[6] else "base";
    // arg 6 `par` means base + parallel (unchanged); arg 10 `par` batches the variant of arg 6
    const par_variant = std.mem.eql(u8, variant, "par");
    const par_tail = try parseParFlag(args);
    const use_par = par_variant or par_tail;
    const is_mstd = std.mem.eql(u8, variant, "mstd");
    const is_dump = std.mem.eql(u8, variant, "dump");
    const is_ft2 = std.mem.eql(u8, variant, "ft2");
    const is_restart = std.mem.eql(u8, variant, "restart");
    // Paired comparison (#168): two policies play the same spawn tape.
    const is_pair = std.mem.eql(u8, variant, "pair");
    // #159 variants train from the net in weights_path (the deep positions come from that net), like `resume`
    const use_resume = std.mem.eql(u8, variant, "resume") or is_dump or is_ft2 or is_restart;
    // Variants that take dump_path at index 11: `dump` writes it, `ft2` and `restart` read it.
    const uses_dump = is_dump or is_ft2 or is_restart;
    const replay_k: usize = try cli.argOr(usize, args, 7, default_replay_k);
    const weights_path: []const u8 = try cli.argOr([]const u8, args, 8, default_weights_path);
    const train_seed: u64 = try cli.argOr(u64, args, 9, default_train_seed);
    // Deep-position buffer (#159). Index 10 must say "par" or "-" (see `parseParFlag`); index 11 is the
    // dump path.
    const dump_path: ?[]const u8 = if (args.len > 11) args[11] else null;
    var dump: nn.Dump = .{};
    var priority: nn.Priority = .{};
    // (#137) `par` merges whole-game deltas into one step; arg 12 scales that step (1 = historical mean).
    const batch_scale: f64 = if (use_par) try parseBatchScale(args) else 1.0;
    const opts: nn.Options = if (par_variant or use_resume or is_pair or std.mem.eql(u8, variant, "base"))
        .{ .batch_scale = batch_scale }
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
    else if (std.mem.eql(u8, variant, "t2"))
        .{ .target = .search2 }
    else if (std.mem.eql(u8, variant, "p2"))
        .{ .selfplay = .deep2 }
    else if (std.mem.eql(u8, variant, "tp2"))
        .{ .target = .search2, .selfplay = .deep2 }
    else if (std.mem.eql(u8, variant, "d2"))
        .{ .teacher = .deep }
    else if (std.mem.eql(u8, variant, "d3"))
        .{ .teacher = .deep3 }
    else if (is_dump)
        .{ .dump_record = true, .dump = &dump }
    else if (is_ft2)
        .{ .deep_mix = true, .dump = &dump }
    else if (is_restart)
        .{ .restart = true, .dump = &dump }
    else if (std.mem.eql(u8, variant, "oi") or std.mem.eql(u8, variant, "oi16") or is_mstd)
        .{}
    else {
        std.debug.print("{s}", .{usage});
        return;
    };
    // `dump` is the shared layout plus one slot: index 11 carries the dump path.
    // `par` runs may carry the #137 batch_scale at index 12, after the `-` dump slot.
    const max_args: usize = if (uses_dump) 12 else if (use_par) 13 else 11;
    if (args.len > max_args or eval_games == 0 or seed_ranges == 0 or (uses_dump and dump_path == null)) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // the priority ring buffer, the dump, the deep mix and the restart buffer are cross-game state:
    // batching would race on them. `mstd` has no batch path either (one staged net is ~1 MB).
    if (use_par and (!parSupported(opts, is_mstd) or opts.dump_record or opts.deep_mix or opts.restart)) {
        std.debug.print("{s}", .{usage});
        return;
    }
    if (is_ft2 or is_restart) loadDump(&dump, init.io, init.gpa, dump_path.?);

    var net: nn.Net = undefined;
    var staged: nn.Staged = undefined;
    const loaded = if (is_mstd)
        try startStaged(&staged, init.io, init.gpa, train_games, train_seed, weights_path)
    else
        try startNet(&net, init.io, init.gpa, train_games, use_resume, train_seed, weights_path);
    if (loaded) std.debug.print("loaded {s}\n", .{weights_path});
    // Optimistic initialization (#107) only rescales the random init, so it stays out of `nn.Options`.
    if (std.mem.eql(u8, variant, "oi")) nn.scaleInit(&net, 4)
    else if (std.mem.eql(u8, variant, "oi16")) nn.scaleInit(&net, 16);
    if (train_games > 0) {
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        var last_score: u64 = 0;
        var total_moves: u64 = 0;
        var diag: nn.Diag = .{};
        if (is_mstd) {
            for (0..train_games) |g| {
                const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
                const game = nn.trainGameStaged(&staged, rng.random(), learning_rate * (1 - 0.9 * frac));
                last_score = game.score;
                total_moves += game.moves;
                if ((g + 1) % 5000 == 0) {
                    std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
                    try staged.saveFile(init.io, init.gpa, weights_path); // checkpoint
                }
            }
        } else if (use_par) {
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
        if (want_diag and !is_mstd) printDiag(&diag);
        if (is_dump) {
            const path = dump_path.?;
            try dump.save(init.io, init.gpa, path);
            std.debug.print("dump: {d} deep afterstates ({d} dropped) -> {s}\n", .{ dump.len, dump.dropped, path });
        }
        if (is_mstd) try staged.saveFile(init.io, init.gpa, weights_path) else try nn.save(&net, init.io, weights_path);
    }

    // Every row the binary knows about. A measurement run edits `row_selection` down to the rows it
    // needs (e.g. policy 0 and 2) instead of paying for the 3-ply rows twice over.
    const all_rows = [_]struct { name: []const u8, policy: usize }{
        .{ .name = "nn 1-ply", .policy = 0 },
        .{ .name = "nn + 1 spawn lookahead", .policy = 1 },
        .{ .name = "nn 2-ply (deep)", .policy = 2 },
        .{ .name = "expectimax depth 2", .policy = 3 },
        .{ .name = "nn 3-ply (deep3, #108)", .policy = 4 },
        .{ .name = "nn phase h1 (#157)", .policy = 5 },
        .{ .name = "nn phase h2 (#157)", .policy = 6 },
        .{ .name = "nn phase h3 (#157)", .policy = 7 },
        .{ .name = "nn phase h4 (#171)", .policy = 8 },
        // #182: the same 2-ply tree as `nn 2-ply (deep)`, minus each chance node's lowest-probability
        // spawn branches up to 5% of its mass. Last row because `playGame` indexes by position.
        .{ .name = "nn 2-ply + mass cutoff (#182)", .policy = 9 },
    };
    // Rows are all on by default; a measurement run edits this line down to the two it needs
    // (0 = 1-ply, 2 = 2-ply) instead of paying for the 3-ply rows twice over.
    const row_selection: [all_rows.len]bool = @splat(true);
    var results_buf: [all_rows.len]Eval = undefined;
    var result_count: usize = 0;
    for (all_rows, 0..) |row, k| {
        if (!row_selection[k]) continue;
        results_buf[result_count] = .{ .name = row.name, .policy = row.policy };
        result_count += 1;
    }
    const results = results_buf[0..result_count];
    const games = eval_games * seed_ranges;
    // `mstd` evaluates through the staged value, everything else through the single net (#107)
    const eval_value: nn.Value = if (is_mstd) .{ .staged = &staged } else .{ .single = &net };

    // Paired comparison (#168): play one game with A (`deep`), then replay the same spawns with B
    // (`deep3`), so both see identical tiles. The per-game difference has much less noise than two
    // independent runs. Opt-in with variant `pair` (train_games 0, eval as usual).
    if (is_pair) {
        const pair_games = eval_games * seed_ranges;
        const spawns = try init.gpa.alloc(nn.Spawn, 20_000);
        defer init.gpa.free(spawns);
        const diffs = try init.gpa.alloc(f64, pair_games);
        defer init.gpa.free(diffs);
        var pair_rng = std.Random.DefaultPrng.init(eval_seed);
        const start_pair = std.Io.Clock.awake.now(init.io);
        for (0..pair_games) |i| {
            var tape = nn.Tape.initRecord(spawns);
            const a = nn.playTapedValue(eval_value, pair_rng.random(), .deep, &tape);
            var replay = nn.Tape.initReplay(spawns[0..tape.len]);
            const b = nn.playTapedValue(eval_value, pair_rng.random(), .deep3, &replay);
            diffs[i] = @as(f64, @floatFromInt(b.score)) - @as(f64, @floatFromInt(a.score));
            std.debug.print("pair {d}: deep={d} ({d}) deep3={d} ({d}) diff={d:.0}\n", .{ i, a.score, a.moves, b.score, b.moves, diffs[i] });
        }
        const st = nn.pairStats(diffs);
        const elapsed = start_pair.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("pair deep (A) vs deep3 (B): games={d} mean diff={d:.1} ± {d:.1} (stderr) better={d} worse={d} elapsed={d:.1} s\n", .{ st.games, st.mean, st.stderr, st.better, st.worse, @as(f64, @floatFromInt(elapsed)) / 1e9 });
        return;
    }
    const EvalCtx = struct { vfn: nn.Value, policy: usize, eval_seed: u64 };
    // Game i uses seed eval_seed + i (= eval_seed + range*eval_games + g); games run in parallel.
    const playGame = struct {
        fn f(c: EvalCtx, i: usize) nn.PlayResult {
            var prng = std.Random.DefaultPrng.init(c.eval_seed + i);
            return switch (c.policy) {
                0 => nn.playValue(c.vfn, prng.random(), .greedy),
                1 => nn.playValue(c.vfn, prng.random(), .chance),
                2 => nn.playValue(c.vfn, prng.random(), .deep),
                3 => blk: {
                    const b = solver.playPolicy(prng.random(), .expectimax, 2);
                    break :blk nn.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves };
                },
                // nn3 baseline for the phase rows below (#157): same seeds, so the cost and score compare directly
                4 => nn.playValue(c.vfn, prng.random(), .deep3),
                5 => nn.playValue(c.vfn, prng.random(), .phase_h1),
                6 => nn.playValue(c.vfn, prng.random(), .phase_h2),
                7 => nn.playValue(c.vfn, prng.random(), .phase_h3),
                8 => nn.playValue(c.vfn, prng.random(), .phase_h4),
                // #182: the same 2-ply tree with the low-probability chance branches cut
                9 => nn.playValue(c.vfn, prng.random(), .deep_mass),
                else => unreachable,
            };
        }
    }.f;
    for (results) |*r| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        const res_all = try init.arena.allocator().alloc(nn.PlayResult, games);
        const start = std.Io.Clock.awake.now(init.io);
        parallel.map(nn.PlayResult, res_all, EvalCtx{ .vfn = eval_value, .policy = r.policy, .eval_seed = eval_seed }, playGame);
        var total_moves: u64 = 0;
        for (0..seed_ranges) |range| {
            var total_score: u64 = 0;
            for (res_all[range * eval_games ..][0..eval_games]) |res| {
                total_score += res.score;
                r.total_score += res.score;
                total_moves += res.moves;
                r.leaves += res.leaves;
                r.best_tile = @max(r.best_tile, res.max_tile);
                if (res.max_tile >= 4096) r.reach_4096 += 1;
                if (res.max_tile >= 8192) r.reach_8192 += 1;
                if (res.max_tile >= 16384) r.reach_16384 += 1;
                if (res.max_tile < 2048) r.dead_below_2048 += 1;
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
        std.debug.print("{s}: avg score={d:.1} ± {d:.1} (stderr over {d} ranges) avg moves={d:.1} best tile={d} elapsed={d:.1} s score/s={d:.1} leaves={d} leaves/move={d:.1} ms/move={d:.2} reach4096={d}/{d} reach8192={d}/{d} reach16384={d}/{d} dead<2048={d}/{d}\n", .{
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
            r.reach_4096,
            games,
            r.reach_8192,
            games,
            r.reach_16384,
            games,
            r.dead_below_2048,
            games,
        });
    }
}

test "train seed argument (position 9) is optional and defaults to 1" {
    const no_args = [_][]const u8{"nn"};
    try std.testing.expectEqual(@as(u64, 1), try cli.argOr(u64, &no_args, 9, default_train_seed));
    const nine_args = [_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin" };
    try std.testing.expectEqual(@as(u64, 1), try cli.argOr(u64, &nine_args, 9, default_train_seed));
    const with_seed = [_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3" };
    try std.testing.expectEqual(@as(u64, 3), try cli.argOr(u64, &with_seed, 9, default_train_seed));
    try std.testing.expectError(error.InvalidCharacter, cli.argOr(u64, &[_][]const u8{ "nn", "0", "20", "2000", "0.003", "-", "base", "30", "x.bin", "x" }, 9, default_train_seed));
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
    const net_default = trainSeedWeights(try cli.argOr(u64, &no_args, 9, default_train_seed), 5);
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
    // "-" is the explicit "no par" placeholder that #159 uses to reach the dump path
    const dash = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "dump", "30", "x.bin", "3", "-", "x.dmp" };
    try std.testing.expect(!try parseParFlag(&dash));
}

test "usage text documents the par flag for stateless variants" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "#99") != null);
}

test "usage text lists the symbatch variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "symbatch (#98") != null);
}

test "usage text lists the restart variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "restart (#159") != null);
}

test "usage text lists the dump variant and its dump_path slot" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "dump    (#159") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "[dump_path]") != null);
}

test "usage text lists the mstd variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "mstd    (#107") != null);
}

test "usage text lists the aligned 2-ply variants (#158)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "t2      (#158") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "p2      (#158") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "tp2     (#158") != null);
}

test "usage text lists the distillation variants (#167)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "d2      (#167") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "d3      (#167") != null);
}

test "par is refused for the priority buffer, the staged net and the 2-ply variants (#158)" {
    var pr: nn.Priority = .{};
    try std.testing.expect(parSupported(.{}, false));
    try std.testing.expect(parSupported(.{ .symmetry = true, .replay_k = 30 }, false));
    try std.testing.expect(!parSupported(.{}, true));
    try std.testing.expect(!parSupported(.{ .priority = &pr }, false));
    try std.testing.expect(!parSupported(.{ .target = .search2 }, false));
    try std.testing.expect(!parSupported(.{ .selfplay = .deep2 }, false));
    try std.testing.expect(!parSupported(.{ .target = .search2, .selfplay = .deep2 }, false));
    // #167: the teacher search is the whole cost, so a batch would only multiply it
    try std.testing.expect(!parSupported(.{ .teacher = .deep }, false));
    try std.testing.expect(!parSupported(.{ .teacher = .deep3 }, false));
}

test "mstd starts from the staged weights file on eval and from 7 random nets otherwise" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/mstd.bin", .{tmp.sub_path});
    defer std.testing.allocator.free(path);

    var prng = std.Random.DefaultPrng.init(41);
    const saved = nn.Staged.init(prng.random());
    try saved.saveFile(io, std.testing.allocator, path);

    // eval-only (train_games 0) and resume load the staged file byte for byte
    var resumed: nn.Staged = undefined;
    try std.testing.expect(try startStaged(&resumed, io, std.testing.allocator, 0, 41, path));
    for (saved.nets, resumed.nets) |a, b| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));

    // training without resume starts from a random init (seed 99, so it must differ)
    var fresh: nn.Staged = undefined;
    try std.testing.expect(!try startStaged(&fresh, io, std.testing.allocator, 5, 99, path));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&saved.nets), std.mem.asBytes(&fresh.nets)));
}
