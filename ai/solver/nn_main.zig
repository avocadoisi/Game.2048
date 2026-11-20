const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");
const parallel = @import("parallel");
const cli = @import("args.zig");
const bench = @import("bench.zig");
const stats = @import("stats");

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
    \\              tnet    (#262: replay plus a frozen bootstrap net; `par` only. The target is a copy of the
    \\                       live net refreshed every arg-13 merged batches, default 100. Self-play moves and
    \\                       the spawn tape still use the live net, only the next-state V in every TD target is frozen)
    \\              priority (#34: 32 extra updates per game sampled from a |TD error| buffer, 50% uniform)
    \\              midgame (#59: 32 extra updates per game from buffer entries with 4..7 empty cells, uniform)
    \\              midmix  (#63: same but 50% uniform over the whole buffer, 50% |TD error| in the 4..7 bucket)
    \\              sym     (#72: every per-move update applied to all 8 dihedral transforms of the afterstate)
    \\              rsym    (#271: one per-move TD update, on a uniformly random dihedral transform of
    \\                       the afterstate; same cost as base, unlike sym's 8 updates)
    \\              rtgt    (#273: rsym plus the target's next board rotated by the same transform, so
    \\                       the update is the TD update of a rotated transition; same cost as base)
    \\              symlam  (#72: sym plus TD(lambda) eligibility traces with lambda = 0.7)
    \\              lam     (#300: `replay_k` from scratch, with the replay sweep itself run as TD(lambda) over one
    \\                              accumulating trace; index 13 = lambda, default 0.7, `0` = the TD(0) control)
    \\              rlam    (#300: resume + replay_k, with the replay sweep itself run as TD(lambda) over one
    \\                       accumulating trace instead of K independent TD(0) steps; lambda at index 13,
    \\                       default 0.7, `0` is the plain replay sweep = the control for the comparison)
    \\              symbatch (#98: the 8 transformed gradients from one net snapshot, averaged and
    \\                       norm-clipped, applied as a single step instead of eight)
    \\              nstep   (#84: online n-step TD with n = 4 instead of one-step returns)
    \\              explore (#224: 5% of the self-play moves are uniformly random instead of greedy,
    \\                       explore20: 20% — widens the boards the value is trained on)
    \\              open8   (#224: the first 8 moves of every game are uniformly random and train
    \\                       nothing, open16: 16 — widens the start boards only)
    \\              death   (#101: reward -1 per move so the value learns V ≈ -(moves to game over))
    \\              t2      (#158: TD target = score delta + the 2-ply value of the post-spawn board,
    \\                       i.e. what the eval search looks at, instead of V(next afterstate))
    \\              p2      (#158: self-play move = the eval's 2-ply search (deep) instead of the 1-ply argmax)
    \\              tp2     (#158: t2 + p2, both sides aligned with the 2-ply eval)
    \\              d2      (#167: distillation, 2-ply teacher: the self-play move is the deep search's
    \\                       and the TD target is the search value of the post-spawn board)
    \\              d3      (#167: distillation, 3-ply teacher: same, from the deep3 tree (1e-2 spawn cut),
    \\                       so a 2-ply eval learns what the deeper search thinks the position is worth)
    \\              dumpp2  (#175: `dump` from 2-ply self-play, i.e. the boards the 2-ply eval visits)
    \\              tch     (#175: teacher pass over a #159 dump: one spawn per stored afterstate, deep3 value,
    \\                       written to <dump_path>.tch; train_games is ignored and nothing is trained)
    \\              tchadd  (#207: same pass, appended to the <dump_path>.tch that must already exist, so
    \\                       several dumps (one per seed) build one teacher file; the cap is 32 dumps)
    \\              dist    (#175: offline distillation: base's per-move TD plus one regression step per step
    \\                       on a uniformly drawn teacher pair from <dump_path>.tch, so 50/50 the update)
    \\              dson    (#175: the teacher regression only: no self-play TD, so nothing but the deep
    \\                       positions trains the net, which shares its parameters with the early game)
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
    \\              adam    (#231: `par` only. The batch gradient is stepped by Adam instead of plain SGD, so
    \\                       every parameter moves about adam_lr per step whatever its gradient scale is)
    \\              tailw   (#232: weights every per-move TD sample by how few empty cells its board has, so
    \\                       the late game takes up to `1 + tail_alpha` times the step of an open board)
    \\              clamp   (#234: bounds the TD target at +/- `clamp` (default 200, i.e. score 200,000) so a
    \\                       diverging value cannot bootstrap itself; `par` compatible)
    \\              shape   (#277: potential-based reward shaping with the empty-cell potential
    \\                       Phi = shape_c * empties / 15, so the target is reward + Phi(after) -
    \\                       Phi(before) + V(after); `par` compatible, coefficient in the row-spec slot,
    \\                       default 1, 0 refused. Only the per-move td1 + greedy1 path)
    \\              clip    (#246: bounds the update itself, cap in the row-spec slot, mode at arg 14:
    \\                       `norm` (default, `par` only) scales the merged batch delta to an L2 norm of
    \\                       the cap; `step` caps every |dw| of a step at the cap; `par` compatible)
    \\  pairw         (#210) paired comparison of two weight files: A = weights_path with `deep`,
    \\                B = the path after the `-` par slot with `deep`, same spawn tape per game
    \\  pairleaf      (#260) A = current, B = omit the last reward before a leaf; same spawn tape,
    \\                optional search mode `deep` (default) or `deep3` at arg 12
    \\  pairrow       (#267) paired comparison of two eval rows: A = the row index at the dump slot
    \\                (arg 11), B = the index at arg 12, same net and spawn tape; prints per-side
    \\                avg score, reach8192, leaves/move and serially measured ms/move
    \\  leaf          (#260) eval-only candidate; rows 2 (`deep`) and 4 (`deep3`) omit the leaf reward
    \\  par           literal "par": train self-play in parallel batches of min(cpu count, 8) games;
    \\                game i uses seed train_seed + i and the batch result is the mean of its games.
    \\                (#99) as arg 10 it batches any stateless variant (base|backward|replay|tnet|sym|symlam|
    \\                nstep|resume); as arg 6 it keeps its old meaning, the plain per-move variant only,
    \\                so replay_k is ignored there but its slot must be filled to reach weights_path.
    \\                Not supported for priority|midgame|midmix (cross-game replay buffer)
    \\                or d2|d3 (#167: the teacher search is the whole cost). t2|p2|tp2 (#158) are plain
    \\                per-move runs, so (#228) they batch like base. "-" here means the same as leaving
    \\                it out, which is how dump_path is reached.
    \\  replay_k      afterstates re-trained per game for variant replay|tnet (default 30)
    \\  weights_path  file to load (train_games 0) or write (default ai/solver/nn_weights.bin); use a temp path for variant runs
    \\  train_seed    seed for weight init and self-play games (default 1, omitted = same weights as before #56)
    \\  dump_path     file for the deep-position buffer (#159); required by dump|ft2|restart, after a "-" par slot
    \\  deep_every    (#207) for ft2: one deep update per this many self-play moves (default 1 = the #159
    \\                mix, which diverges; 4..8 copies what the stable offline twin `dist` uses).
    \\                For tch: the teacher search mode, `deep3` (default) or `h4` (3-ply + 4-ply in the endgame)
    \\  batch_scale   (#137) multiplier on the merged batch delta for `par` runs (default = sqrt of the
    \\                batch size, #203; `-` = 1 = the old mean, `8` = the #208 linear rule). Sits after
    \\                the "-" par dump slot
    \\  rows          (#203) eval-table rows to run, comma list after batch_scale (`-` = all), e.g.
    \\                "0,2" for 1-ply and 2-ply only; the 3-ply rows dominate a measurement run.
    \\                For variant adam this slot is adam_lr instead (default 0.01), like dist's divisor,
    \\                tailw's tail_alpha (default 1: twice the step on a full board), clamp's ceiling,
    \\                tnet's sync interval in merged batches (default 100) and clip's cap (default 1, the
    \\                sym_batch_clip scale). Arg 14 then picks clip's mode, `norm` (default) or `step`.
    \\
    \\Weights are written to `weights_path` only when train_games > 0;
    \\train_games = 0 (eval only) never writes.
    \\
    \\No arguments (e.g. `zig build nn` without `--`) prints this and exits: it must not retrain and
    \\overwrite ai/solver/nn_weights.bin (#207).
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

/// Every accepted `variant` string, one tag per variant. The table below owns the facts and the options
/// builder that used to be repeated at the hand-written dispatch sites in `main` (#314, #315).
const Variant = enum {
    base,
    backward,
    replay,
    tnet,
    priority,
    midgame,
    midmix,
    sym,
    symlam,
    symbatch,
    rsym,
    rtgt,
    explore,
    explore20,
    open8,
    open16,
    nstep,
    death,
    t2,
    p2,
    tp2,
    d2,
    d3,
    dumpp2,
    tch,
    tchadd,
    dist,
    dson,
    oi,
    oi16,
    @"resume",
    mstd,
    dump,
    ft2,
    restart,
    adam,
    tailw,
    clamp,
    shape,
    clip,
    pairw,
    pairleaf,
    pairrow,
    leaf,
    pair,
    par,
    lam,
    rlam,
    /// Not in `variant_specs`: the placeholder for a name no entry matched. It carries every default
    /// fact and no builder, so `main` prints the usage where the old chain's final `else` did - the
    /// same place, and after the same parses, as before the table existed (the old code parsed later
    /// slots before it noticed the name was unknown).
    unknown,
};

/// Which arm of the old `max_args` if-chain a variant used. `rows` is the arm shared by the variants
/// that read the eval-row slot; the others carry one variant-specific ceiling.
const MaxArgs = enum { clip, value, rows, ft2_tch, dump, pairw_pairleaf, pairrow };

/// Everything a variant's `nn.Options` builder reads. The buffers the options point at (`adam`,
/// `priority`, `dump`, `teachers`, the `tnet` snapshot) belong to `main`, so they are borrowed here;
/// `args` is the raw command line, read only through the `parse*` helpers.
const BuildCtx = struct {
    args: []const []const u8,
    replay_k: usize,
    learning_rate: f64,
    adam: *nn.Adam,
    priority: *nn.Priority,
    dump: *nn.Dump,
    teachers: *nn.Teachers,
    /// (#262) `tnet`: the snapshot the next-state value of every TD target is read from.
    target_net: *const nn.Net,
    /// (#246) parsed in `main`, which also needs it for the `par` rejection and the `clip:<mode>` log.
    clip_mode: ClipMode,
    /// (#206) parsed in `main` before the options are built, where the old chain parsed it: reading it
    /// inside the builder would reorder it after `replay_k`/`batch_scale` and turn an `InvalidDistill*`
    /// on a line with two bad slots into their error.
    dist_mix: DistillMix,
};

/// One variant's `nn.Options`, built from the command line. The tail every variant shares
/// (`batch_scale`, `nn.validateOptions`) stays in `main`.
const BuildFn = *const fn (*const BuildCtx) anyerror!nn.Options;

/// Facts that depend on the variant name. `build` is the entry's options builder (`null` only for the
/// `.unknown` placeholder, which exists to print the usage), `load_weights` loads `weights_path`
/// before training, `dump` requires the index-11 dump path, `rows` reads index 13 as eval rows, and
/// `max_args` selects the cap.
const VariantSpec = struct {
    variant: Variant,
    build: ?BuildFn = null,
    load_weights: bool = false,
    dump: bool = false,
    rows: bool = true,
    max_args: MaxArgs = .rows,
};

/// One entry per accepted variant. The order is only for readers; lookup uses the tag name.
const variant_specs = [_]VariantSpec{
    .{ .variant = .base, .build = &buildPlain },
    .{ .variant = .backward, .build = &buildBackward },
    .{ .variant = .replay, .build = &buildReplay },
    .{ .variant = .tnet, .build = &buildTnet, .rows = false, .max_args = .value },
    .{ .variant = .priority, .build = &buildPriority },
    .{ .variant = .midgame, .build = &buildMidgame },
    .{ .variant = .midmix, .build = &buildMidmix },
    .{ .variant = .sym, .build = &buildSym },
    .{ .variant = .symlam, .build = &buildSymLambda },
    .{ .variant = .symbatch, .build = &buildSymBatch },
    .{ .variant = .rsym, .build = &buildRsym },
    .{ .variant = .rtgt, .build = &buildRtgt },
    .{ .variant = .explore, .build = &buildExplore },
    .{ .variant = .explore20, .build = &buildExplore20 },
    .{ .variant = .open8, .build = &buildOpen8 },
    .{ .variant = .open16, .build = &buildOpen16 },
    .{ .variant = .nstep, .build = &buildNstep },
    .{ .variant = .death, .build = &buildDeath },
    .{ .variant = .t2, .build = &buildT2, .load_weights = true },
    .{ .variant = .p2, .build = &buildP2, .load_weights = true },
    .{ .variant = .tp2, .build = &buildTp2, .load_weights = true },
    .{ .variant = .d2, .build = &buildD2 },
    .{ .variant = .d3, .build = &buildD3 },
    .{ .variant = .dumpp2, .build = &buildDumpP2, .load_weights = true, .dump = true, .max_args = .dump },
    .{ .variant = .tch, .build = &buildPlain, .load_weights = true, .dump = true, .max_args = .ft2_tch },
    .{ .variant = .tchadd, .build = &buildPlain, .load_weights = true, .dump = true, .max_args = .ft2_tch },
    .{ .variant = .dist, .build = &buildDist, .load_weights = true, .dump = true, .rows = false, .max_args = .value },
    .{ .variant = .dson, .build = &buildDson, .load_weights = true, .dump = true, .rows = false, .max_args = .value },
    .{ .variant = .oi, .build = &buildPlain },
    .{ .variant = .oi16, .build = &buildPlain },
    .{ .variant = .@"resume", .build = &buildPlain, .load_weights = true },
    .{ .variant = .mstd, .build = &buildPlain },
    .{ .variant = .dump, .build = &buildDump, .load_weights = true, .dump = true, .max_args = .dump },
    .{ .variant = .ft2, .build = &buildFt2, .load_weights = true, .dump = true, .max_args = .ft2_tch },
    .{ .variant = .restart, .build = &buildRestart, .load_weights = true, .dump = true, .max_args = .dump },
    .{ .variant = .adam, .build = &buildAdam, .rows = false, .max_args = .value },
    .{ .variant = .tailw, .build = &buildTailw, .rows = false, .max_args = .value },
    .{ .variant = .clamp, .build = &buildClamp, .rows = false, .max_args = .value },
    .{ .variant = .shape, .build = &buildShape, .rows = false, .max_args = .value },
    .{ .variant = .clip, .build = &buildClip, .rows = false, .max_args = .clip },
    .{ .variant = .pairw, .build = &buildPlain, .rows = false, .max_args = .pairw_pairleaf },
    .{ .variant = .pairleaf, .build = &buildPlain, .rows = false, .max_args = .pairw_pairleaf },
    .{ .variant = .pairrow, .build = &buildPlain, .rows = false, .max_args = .pairrow },
    .{ .variant = .leaf, .build = &buildLeaf },
    .{ .variant = .pair, .build = &buildPlain, .rows = false },
    .{ .variant = .par, .build = &buildPlain },
    .{ .variant = .lam, .build = &buildSweepReplay, .max_args = .value },
    .{ .variant = .rlam, .build = &buildSweepReplay, .load_weights = true, .max_args = .value },
};

// One builder per options family: the arms of the options if-chain #315 moved out of `main`, each
// keeping the variant's `parse*` call next to the field it fills. A builder that ignores the command
// line takes `_` instead of the context.

/// The plain per-move TD path: `base`, `resume`, `par`, the paired runs and the teacher pass all run
/// it with the default options, and `oi`/`oi16`/`mstd` differ from it outside `nn.Options`.
fn buildPlain(_: *const BuildCtx) anyerror!nn.Options {
    return .{};
}

/// (#260) `leaf` is the eval-only omit-leaf candidate: the rows take the flag up in the eval loop.
fn buildLeaf(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .omit_leaf_reward = true };
}

/// Re-train every stored afterstate from last to first after the game.
fn buildBackward(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .backward = true };
}

/// `replay`: re-train the last `replay_k` afterstates once more after the game.
fn buildReplay(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .replay_k = ctx.replay_k };
}

/// (#300) `lam` / `rlam`: the replay sweep run as TD(lambda) over one accumulating trace, so index
/// 13 is the lambda and `0` is the plain TD(0) sweep (the control).
fn buildSweepReplay(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .replay_k = ctx.replay_k, .sweep_lambda = try parseSweepLambda(ctx.args) };
}

/// (#262) `tnet`: `replay` whose next-state value comes from the frozen snapshot in `TargetSync`.
fn buildTnet(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .replay_k = ctx.replay_k, .target_net = ctx.target_net };
}

/// (#34) `priority`: 32 extra updates per game drawn from the shared |TD error| buffer.
fn buildPriority(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .priority = ctx.priority };
}

/// (#59) `midgame`: the buffer's extra updates draw from the 4..7 empty-cell bucket only.
fn buildMidgame(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .priority = ctx.priority, .midgame = true };
}

/// (#63) `midmix`: uniform over the whole buffer, |TD error| inside the 4..7 bucket.
fn buildMidmix(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .priority = ctx.priority, .midmix = true };
}

/// (#72) `sym`: every per-move update applied to all 8 dihedral transforms of the afterstate.
fn buildSym(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .symmetry = true };
}

/// (#72) `symlam`: `sym` plus TD(lambda) eligibility traces.
fn buildSymLambda(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .symmetry = true, .lambda = 0.7 };
}

/// (#98) `symbatch`: the 8 transformed gradients averaged and norm-clipped into one step.
fn buildSymBatch(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .sym_batch = true };
}

/// (#271) `rsym`: one update per move, on a uniformly random dihedral transform.
fn buildRsym(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .rsym = true };
}

/// (#273) `rtgt`: `rsym` with the TD target's next board rotated by the same transform.
fn buildRtgt(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .rtgt = true };
}

/// (#224) `explore`: 5% of the self-play moves are uniformly random instead of greedy.
fn buildExplore(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .explore = 0.05 };
}

fn buildExplore20(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .explore = 0.20 };
}

/// (#224) `open8`: the first 8 moves are uniformly random and train nothing.
fn buildOpen8(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .warm_open = 8 };
}

fn buildOpen16(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .warm_open = 16 };
}

/// (#84) `nstep`: online n-step TD with n = 4.
fn buildNstep(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .nstep = 4 };
}

/// (#101) `death`: -1 per move plus a backward sweep, so the value learns moves to game over.
fn buildDeath(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .death = true };
}

/// (#158) `t2`: the TD target is the eval's 2-ply search value of the post-spawn board.
fn buildT2(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .target = .search2 };
}

/// (#158) `p2`: the self-play move is the eval's 2-ply search instead of the 1-ply argmax.
fn buildP2(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .selfplay = .deep2 };
}

/// (#158) `tp2`: both sides aligned with the 2-ply eval.
fn buildTp2(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .target = .search2, .selfplay = .deep2 };
}

/// (#167) `d2` / `d3`: the per-move target and the self-play move are the deep search's, at 2 or 3 ply.
fn buildD2(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .teacher = .deep };
}

fn buildD3(_: *const BuildCtx) anyerror!nn.Options {
    return .{ .teacher = .deep3 };
}

/// (#231) `adam`: Adam on the merged batch gradient; `par` is required, checked in `main`.
fn buildAdam(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .adam = ctx.adam, .adam_lr = try parseAdamLr(ctx.args) };
}

/// (#232) `tailw`: per-sample step weight from the afterstate's empty-cell count.
fn buildTailw(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .tail_alpha = try parseTailAlpha(ctx.args) };
}

/// (#234) `clamp`: ceiling on the TD target, so a diverging value cannot bootstrap itself.
fn buildClamp(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .clamp = try parseClamp(ctx.args) };
}

/// (#277) `shape`: potential-based shaping with the empty-cell potential, coefficient at index 13.
fn buildShape(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .shape_c = try parseShapeC(ctx.args) };
}

/// (#246) `clip`: `norm` caps the L2 norm of one merged batch delta, `step` every |dw| of a step.
/// Both read the cap at index 13; the mode at index 14 was parsed in `main`, which refuses a `norm`
/// cap without `par`.
fn buildClip(ctx: *const BuildCtx) anyerror!nn.Options {
    var opts: nn.Options = .{};
    const cap = try parseStepCap(ctx.args);
    switch (ctx.clip_mode) {
        .norm => opts.grad_clip = cap,
        .step => opts.step_clip = cap,
    }
    return opts;
}

/// (#159) `dump`: record every afterstate of the plain per-move path whose max tile reached 4096.
fn buildDump(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .dump_record = true, .dump = ctx.dump, .selfplay = .greedy1 };
}

/// (#175) `dumpp2`: the same buffer from 2-ply self-play boards, the distribution the 2-ply eval
/// actually visits, which is what the teacher regression is supposed to fix.
fn buildDumpP2(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .dump_record = true, .dump = ctx.dump, .selfplay = .deep2 };
}

/// (#159) `ft2`: one deep update per `deep_every` self-play moves (that count at index 12).
fn buildFt2(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .deep_mix = true, .dump = ctx.dump, .deep_every = try parseDeepEvery(ctx.args) };
}

/// (#159) `restart`: every self-play game starts on a deep afterstate of `dump` plus one spawn.
fn buildRestart(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{ .restart = true, .dump = ctx.dump };
}

/// (#175) `dist`: the per-move TD plus one teacher regression per `distill_every` steps at
/// `distillRate` (the TD rate over the index-13 divisor, see #207).
fn buildDist(ctx: *const BuildCtx) anyerror!nn.Options {
    return .{
        .distill = ctx.teachers,
        .distill_every = ctx.dist_mix.every,
        .distill_lr = distillRate(ctx.learning_rate, ctx.dist_mix.div),
    };
}

/// (#175) `dson`: the teacher regression alone, with no self-play TD update.
fn buildDson(ctx: *const BuildCtx) anyerror!nn.Options {
    var opts = try buildDist(ctx);
    opts.distill_only = true;
    return opts;
}

/// Resolve a command-line variant name exactly once. This is the only variant-name comparison left.
fn findVariant(name: []const u8) ?VariantSpec {
    for (variant_specs) |spec| {
        if (std.mem.eql(u8, name, @tagName(spec.variant))) return spec;
    }
    return null;
}

/// Reproduce the old `max_args` if-chain from the table facts and the two runtime slots that feed it.
fn maxArgs(spec: VariantSpec, row_spec: ?[]const u8, use_par: bool) usize {
    return switch (spec.max_args) {
        .clip => 15,
        .value => 14,
        .rows => if (row_spec != null) 14 else if (use_par) 13 else 11,
        .ft2_tch => if (row_spec != null) 14 else 13,
        .dump => if (row_spec != null) 14 else 12,
        .pairw_pairleaf => if (use_par) 13 else 12,
        .pairrow => 13,
    };
}

/// (#203) Fill `sel` from the `rows` argument: comma-separated indices, any bad token or an index
/// past the table is an error, so a typo never silently runs the full table.
fn parseRows(sel: []bool, spec: []const u8) !void {
    @memset(sel, false);
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |tok| {
        const idx = std.fmt.parseInt(usize, std.mem.trim(u8, tok, " "), 10) catch return error.InvalidRow;
        if (idx >= sel.len) return error.InvalidRow;
        sel[idx] = true;
    }
}

/// Default `par` step multiplier (#203): sqrt of the batch size. The linear rule (= the batch size)
/// matches serial over 20k games but oversteps on a 100k run at lr 0.003 (three seeds land 26k-68k
/// behind the shipped net in the paired test), while sqrt stays at the shipped net's level.
fn batchScaleFor(batch_size: usize) f64 {
    return @sqrt(@as(f64, @floatFromInt(batch_size)));
}

fn defaultBatchScale() f64 {
    return batchScaleFor(nn.trainBatchSize());
}

/// Positional arg 10, appended after `train_seed`: the literal "par" runs the variant of arg 6 in
/// parallel batches (issue #99). "-" is the explicit "no par" placeholder that opens the dump slots
/// of #159; any other value is a typo.
///
/// (#137) Batch delta multiplier for `par` runs, arg 12 (after the `par` slot and its `-` dump slot):
/// omitted = `defaultBatchScale()`, `-` = the historical 1.0 ("mean of the per-game deltas", which
/// under-trains by 1/games), a number = that multiplier (`8` = the #208 linear rule).
fn parseBatchScale(args: []const []const u8) !f64 {
    if (args.len <= 12) return defaultBatchScale();
    if (std.mem.eql(u8, args[12], "-")) return 1.0;
    return std.fmt.parseFloat(f64, args[12]) catch error.InvalidBatchScale;
}

/// (#207) `tch` takes the teacher's search mode at arg 12: `deep3` (the default, #175/#206) or `h4`
/// (`phase_h4`: 3-ply everywhere plus 4-ply in the endgame, #171), which prices the deep boards the
/// dump actually holds with one more ply. Typo = error, so a label pass never silently uses the old mode.
fn parseTeacherMode(args: []const []const u8) !nn.Mode {
    if (args.len <= 12) return .deep3;
    const name = args[12];
    if (std.mem.eql(u8, name, "-") or std.mem.eql(u8, name, "deep3")) return .deep3;
    if (std.mem.eql(u8, name, "h4")) return .phase_h4;
    if (std.mem.eql(u8, name, "deep")) return .deep;
    return error.InvalidTeacherMode;
}

/// (#207) `ft2` takes the deep-half thinning after the dump path: arg 12 is `deep_every`, one deep
/// update per that many self-play moves. `1` is the #159 50% mix, which diverges to NaN; 4..8 is the
/// mix the stable offline twin (`dist`) uses.
fn parseDeepEvery(args: []const []const u8) !u32 {
    if (args.len <= 12) return 1;
    if (std.mem.eql(u8, args[12], "-")) return 1;
    const every = std.fmt.parseInt(u32, args[12], 10) catch return error.InvalidDeepEvery;
    if (every == 0) return error.InvalidDeepEvery;
    return every;
}

/// (#206) `dist` / `dson` take the teacher mix after the dump path: arg 12 is `distill_every`
/// (1 = every step) and arg 13 is the divisor applied to the TD rate for teacher updates (0 = same rate).
const DistillMix = struct { every: u32 = 1, div: f64 = 0 };

fn parseDistillMix(args: []const []const u8) !DistillMix {
    var mix: DistillMix = .{};
    if (args.len > 12 and !std.mem.eql(u8, args[12], "-")) {
        mix.every = std.fmt.parseInt(u32, args[12], 10) catch return error.InvalidDistillEvery;
    }
    if (args.len > 13 and !std.mem.eql(u8, args[13], "-")) {
        mix.div = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidDistillLr;
        if (!(mix.div >= 0)) return error.InvalidDistillLr;
    }
    return mix;
}

/// Absolute teacher step: the TD rate divided by the arg-13 divisor, or 0 for "same as TD".
///
/// `nn.Options.distill_lr` is an absolute rate (see its tests in `nn.zig`), so the divisor has to be
/// applied here, where the TD rate is known. The #206 code wrote `1 / div` into the field instead, and
/// `dist` with the documented divisor of 10 therefore took steps 300x the TD rate at lr 3e-4: the net
/// blew up to NaN within a few thousand games (#207).
fn distillRate(learning_rate: f64, div: f64) f64 {
    return if (div > 0) learning_rate / div else 0;
}

fn parseParFlag(args: []const []const u8) !bool {
    if (args.len <= 10) return false;
    if (std.mem.eql(u8, args[10], "-")) return false;
    if (!std.mem.eql(u8, args[10], "par")) return error.InvalidParFlag;
    return true;
}

/// (#231) Adam's own step size for variant `adam`, at the row-spec slot (dist owns that slot for its
/// teacher divisor, so the two variants never collide).
fn parseAdamLr(args: []const []const u8) !f64 {
    if (args.len <= 13) return 0.01;
    if (std.mem.eql(u8, args[13], "-")) return 0.01;
    const lr = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidAdamLr;
    if (!(lr > 0)) return error.InvalidAdamLr;
    return lr;
}

/// (#234) Ceiling on the TD target for variant `clamp`, at the same row-spec slot as Adam's step size and
/// `tailw`'s alpha. 0 or `-` leaves the target alone (i.e. a plain `base` run), so a value is required.
fn parseClamp(args: []const []const u8) !f64 {
    if (args.len <= 13) return 200;
    if (std.mem.eql(u8, args[13], "-")) return 200;
    const cap = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidClamp;
    if (!(cap > 0)) return error.InvalidClamp;
    return cap;
}

/// (#300) `sweep_lambda` of the `rlam` variant, at the row-spec slot (index 13). Default 0.7; `0`
/// restores the plain TD(0) replay sweep, which is the control of the #300 comparison, so 0 is allowed
/// (unlike `tailw`, where 0 would silently make the variant a plain `base`) but negatives are refused.
fn parseSweepLambda(args: []const []const u8) !f64 {
    if (args.len <= 13) return default_sweep_lambda;
    if (std.mem.eql(u8, args[13], "-")) return default_sweep_lambda;
    const lambda = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidSweepLambda;
    if (!(lambda >= 0 and lambda <= 1)) return error.InvalidSweepLambda;
    return lambda;
}

/// (#232) `tail_alpha` for variant `tailw`, at the same row-spec slot as Adam's step size: the step of a
/// per-move sample is `lr * (1 + alpha * (15 - empties) / 15)`, i.e. up to `1 + alpha` times `lr` on a
/// board with no empty cell. 0 would make the variant a plain `base` run, so it is refused.
fn parseTailAlpha(args: []const []const u8) !f64 {
    if (args.len <= 13) return 1;
    if (std.mem.eql(u8, args[13], "-")) return 1;
    const alpha = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidTailAlpha;
    if (!(alpha > 0)) return error.InvalidTailAlpha;
    return alpha;
}

/// (#277) `shape_c` for variant `shape`, at the same row-spec slot as Adam's step size, `tailw`'s
/// alpha, `clamp`'s target bound and `clip`'s cap. The default and `-` are 1; 0 would make the variant
/// a plain `base` run, so it is refused. Negative coefficients are valid potential shaping.
fn parseShapeC(args: []const []const u8) !f64 {
    if (args.len <= 13) return 1;
    if (std.mem.eql(u8, args[13], "-")) return 1;
    const shape_c = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidShapeC;
    if (!std.math.isFinite(shape_c) or shape_c == 0) return error.InvalidShapeC;
    return shape_c;
}

/// (#246) Which ceiling the `clip` variant applies: the L2 norm of one merged `par` batch delta
/// (`norm`, the `sym_batch_clip` style) or every element of one step's `|dw|` (`step`).
const ClipMode = enum { norm, step };

/// (#246) `clip`'s default ceiling: the unit L2 norm `sym_batch_clip` already uses for an averaged
/// symmetry gradient, so `clip` without a value bounds a batch step to a single sample's scale.
const default_clip_cap: f64 = nn.sym_batch_clip;

/// (#246) `clip`'s ceiling, at the same row-spec slot as Adam's step size, `tailw`'s alpha and
/// `clamp`'s target bound. 0 or `-` would be a plain `base` run, so it is refused.
fn parseStepCap(args: []const []const u8) !f64 {
    if (args.len <= 13) return default_clip_cap;
    if (std.mem.eql(u8, args[13], "-")) return default_clip_cap;
    const cap = std.fmt.parseFloat(f64, args[13]) catch return error.InvalidStepCap;
    if (!(cap > 0)) return error.InvalidStepCap;
    return cap;
}

/// (#246) `clip`'s mode at arg 14: `norm` (default) bounds the merged batch delta's L2 norm, `step`
/// bounds every element of a step. A stray token is an error, never a silent fallback to `norm`.
fn parseClipMode(args: []const []const u8) !ClipMode {
    if (args.len <= 14) return .norm;
    if (std.mem.eql(u8, args[14], "-") or std.mem.eql(u8, args[14], "norm")) return .norm;
    if (std.mem.eql(u8, args[14], "step")) return .step;
    return error.InvalidClipMode;
}

/// `par` forks whole games onto threads, so it cannot carry the cross-game priority buffer (shared by
/// pointer), the staged net (no batch path, one net is ~1 MB), or the #167 teacher cells (their search
/// is the whole cost of the run; batching would only multiply it by the thread count).
/// The #158 aligned variants t2/p2/tp2 use the plain per-move path, so they batch like base; the
/// value cache is threadlocal and cleared per move (#228), so the games do not share stale leaves.
fn parSupported(opts: nn.Options, is_mstd: bool) bool {
    // `distill` stays out: it is read-only and race-free, but a 20,000-game `dist` net trained under `par`
    // is 30,000 points behind the same serial run at every batch_scale tried (#207), so the batched path
    // is refused rather than silently offered.
    // `-Dfeat` builds stay out too (#226): a feature net trained under `par` lands far below the same
    // serial run (feat 1, 100k games, sqrt rule: 27,433 under `par` vs 81,959 serial), so the batched
    // path would silently produce a broken net.
    return opts.priority == null and !is_mstd and opts.teacher == null and opts.distill == null and
        nn.feature_count == 0 and
        (opts.target == .td1 or opts.target == .search2) and
        (opts.selfplay == .greedy1 or opts.selfplay == .deep2);
}

/// Foreign rerank score of the #189 eval row: the snake heuristic of `solver.evaluate` (the #117
/// `baseline` weights), divided by `nn.reward_scale` so it is on the NN value scale.
fn snakeScore(board: core.Board) f64 {
    return solver.evaluate(board) / nn.reward_scale;
}

/// Loads the #175 teacher file (`tch` output) for `dist`. A missing, foreign or empty file is a
/// command-line error (usage + exit 1), never a silent fallback to plain self-play.
fn loadTeachers(buffer: *nn.Teachers, io: std.Io, gpa: std.mem.Allocator, path: []const u8) void {
    buffer.load(io, gpa, path) catch |err| {
        std.debug.print("error: cannot read teacher file {s}: {s}\n{s}", .{ path, @errorName(err), usage });
        std.process.exit(1);
    };
    if (buffer.len == 0) {
        std.debug.print("error: teacher file {s} holds no samples\n{s}", .{ path, usage });
        std.process.exit(1);
    }
    std.debug.print("teacher file {s}: {d} samples\n", .{ path, buffer.len });
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

/// The search one eval row runs. Every row but the handcrafted expectimax baseline is an NN
/// search; `pairrow` refuses the expectimax row.
const NnEvalRow = struct {
    mode: nn.Mode,
    opts: nn.Options = .{},
    /// (#260) The eval-only `leaf` variant reads `omit_leaf_reward` from the run options. Only
    /// the rows whose leaves it changes set this; the row's other options stay in `opts`.
    leaf_override: bool = false,
};

const EvalRowSpec = union(enum) {
    nn: NnEvalRow,
    /// Handcrafted expectimax baseline; the payload is its depth.
    expectimax: u8,
};

/// One eval row: the name that is printed and the search that produces its numbers. The array index
/// is the row number used by `rows` and `pairrow`, so this is the only place a row is defined.
const EvalRow = struct {
    name: []const u8,
    spec: EvalRowSpec,
};

// Every row the binary knows about. `rows` selects the rows a measurement run needs (e.g. 0 and 2)
// instead of paying for the 3-ply rows twice over.
const all_rows = [_]EvalRow{
    .{ .name = "nn 1-ply", .spec = .{ .nn = .{ .mode = .greedy } } },
    .{ .name = "nn + 1 spawn lookahead", .spec = .{ .nn = .{ .mode = .chance } } },
    .{ .name = "nn 2-ply (deep)", .spec = .{ .nn = .{ .mode = .deep, .leaf_override = true } } },
    .{ .name = "expectimax depth 2", .spec = .{ .expectimax = 2 } },
    .{ .name = "nn 3-ply (deep3, #108)", .spec = .{ .nn = .{ .mode = .deep3, .opts = .{ .node_tt = false }, .leaf_override = true } } },
    .{ .name = "nn phase h1 (#157)", .spec = .{ .nn = .{ .mode = .phase_h1 } } },
    .{ .name = "nn phase h2 (#157)", .spec = .{ .nn = .{ .mode = .phase_h2 } } },
    .{ .name = "nn phase h3 (#157)", .spec = .{ .nn = .{ .mode = .phase_h3 } } },
    .{ .name = "nn phase h4 (#171)", .spec = .{ .nn = .{ .mode = .phase_h4 } } },
    // #182: the same 2-ply tree as `nn 2-ply (deep)`, minus each chance node's lowest-probability
    // spawn branches up to 5% of its mass.
    .{ .name = "nn 2-ply + mass cutoff (#182)", .spec = .{ .nn = .{ .mode = .deep_mass } } },
    // #178: the .deep row with every search leaf averaged over the 8 dihedral transforms.
    .{ .name = "nn 2-ply + sym leaf (#178)", .spec = .{ .nn = .{ .mode = .deep, .opts = .{ .sym_leaf = true } } } },
    // #183: iterative deepening from the .deep row, stopping at a leaf-evaluation budget instead of
    // a fixed ply, so a position it can afford searches deeper than 2 and still costs the same.
    .{ .name = "nn iterative deepening (#183)", .spec = .{ .nn = .{ .mode = .deep, .opts = .{ .id_budget = 200_000 } } } },
    // #189: the .deep row, reranked between the top two moves by the snake heuristic when the
    // net's own gap is within 0.05 (NN values are score deltas / reward_scale, so 50 points of
    // one-move score); the snake score enters at half its `reward_scale` units.
    .{ .name = "nn 2-ply + rerank (#189)", .spec = .{ .nn = .{ .mode = .deep, .opts = .{ .rerank = .{ .margin = 0.05, .weight = 0.5, .score = &snakeScore } } } } },
    // #187: the .deep row with every chance node scored 30% by its CVaR below the lower 25%
    // quantile, so a move that is usually good but sometimes losing ranks below a stable one.
    .{ .name = "nn 2-ply + downside risk (#187)", .spec = .{ .nn = .{ .mode = .deep, .opts = .{ .risk_weight = 0.3 } } } },
    // #188: the #183 budget, but a root move that already beats its runner-up by more than 0.05
    // ends the deepening after the first iteration, so only near-tie positions pay for a deeper tree.
    .{ .name = "nn deepening on near ties (#188)", .spec = .{ .nn = .{ .mode = .deep, .opts = .{ .id_budget = 200_000, .id_margin = 0.05 } } } },
    // #263: the .deep3 row with completed interior nodes reused inside one search. #266 made this
    // the default for 3-ply, so the row passes no options and measures the shipped path; row 4 is
    // the explicit-off baseline the two are compared on.
    .{ .name = "nn 3-ply + node TT default (#266)", .spec = .{ .nn = .{ .mode = .deep3 } } },
    // #263: the same table at a 1e-3 spawn cut: the time the table saved, spent on a wider tree.
    .{ .name = "nn 3-ply 1e-3 + node TT (#263)", .spec = .{ .nn = .{ .mode = .deep3, .opts = .{ .cutoff = 1e-3 } } } },
    // #267: `deep4`, one more own move and spawn layer than row 15 at the same 1e-2 cut, with the
    // node table on. A/B against row 15 answers whether the saved time buys a deeper tree.
    .{ .name = "nn 4-ply + node TT (#267)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .node_tt = true } } } },
    // #281: row 15 with a higher spawn cut, so the 4 branches at empty cells >= 6 (0.02) or >= 3
    // (0.05) are dropped. A/B against row 15 measures whether the saved leaves buy a better score.
    .{ .name = "nn 3-ply cutoff 0.02 + node TT (#281)", .spec = .{ .nn = .{ .mode = .deep3, .opts = .{ .cutoff = 0.02, .node_tt = true } } } },
    .{ .name = "nn 3-ply cutoff 0.05 + node TT (#281)", .spec = .{ .nn = .{ .mode = .deep3, .opts = .{ .cutoff = 0.05, .node_tt = true } } } },
    // #261: the `deep3` tree, but a spawn branch under the fixed cut is evaluated on its real
    // child board one own-move/reward layer shallower instead of being dropped and renormalized.
    .{ .name = "nn 3-ply cut branches recovered 1 ply shallow (#261)", .spec = .{ .nn = .{ .mode = .deep3_recover } } },
    // #320: `deep4`'s extra ply, paid for by keeping only a finite weighted sample of each chance
    // node's spawns (0.9/0.1 kept, deterministic seed). A/B against row 15 answers whether the deeper
    // tree beats the full-width 3-ply at a comparable leaf budget.
    .{ .name = "nn 4-ply spawn sample 2 + node TT (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 2 } } } },
    .{ .name = "nn 4-ply spawn sample 3 + node TT (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 3 } } } },
    .{ .name = "nn 4-ply spawn sample 4 + node TT (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 4 } } } },
    .{ .name = "nn 4-ply spawn sample 6 + node TT (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 6 } } } },
    .{ .name = "nn 4-ply spawn sample 2, full last spawn layer (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 2, .sample_min_depth = 2 } } } },
    .{ .name = "nn 4-ply spawn sample 3, full last spawn layer (#320)", .spec = .{ .nn = .{ .mode = .deep4, .opts = .{ .sample = 3, .sample_min_depth = 2 } } } },
};

const Eval = struct {
    name: []const u8,
    /// The search this row runs, copied from `all_rows` so eval and `pairrow` cannot drift.
    spec: EvalRowSpec,
    range_scores: []f64 = &.{},
    avg_moves: f64 = 0,
    best_tile: core.Tile = 0,
    total_score: u64 = 0,
    leaves: u64 = 0,
    /// #263 interior-node probes and hits over the whole run (0 when the row has no node table).
    node_probes: u64 = 0,
    node_hits: u64 = 0,
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

fn parsePairLeafMode(arg: ?[]const u8) ?nn.Mode {
    const mode = arg orelse return .deep;
    if (std.mem.eql(u8, mode, "deep")) return .deep;
    if (std.mem.eql(u8, mode, "deep3")) return .deep3;
    return null;
}

/// `pairrow` row index (#267): a 0-based index into `all_rows`, so `15` is the shipped 3-ply row.
fn parsePairRow(arg: ?[]const u8) ?usize {
    const text = arg orelse return null;
    return std.fmt.parseInt(usize, std.mem.trim(u8, text, " "), 10) catch null;
}

/// (#262) Frozen bootstrap net for `tnet`: a copy of the live net, refreshed only between merged
/// `par` batches. Every game inside a batch therefore prices its next afterstate from the same
/// snapshot; the target cannot chase the weights it is training, and no batch sees a mid-batch copy.
const TargetSync = struct {
    every: usize,
    batches: usize = 0,
    net: nn.Net,

    fn init(net: *const nn.Net, every: usize) TargetSync {
        return .{ .every = every, .net = net.* };
    }

    /// One merged batch finished; refresh the snapshot when the interval elapsed.
    fn afterBatch(self: *TargetSync, net: *const nn.Net) void {
        self.batches += 1;
        if (self.batches % self.every == 0) self.net = net.*;
    }
};

/// (#262) `tnet` sync interval: merged `par` batches between target snapshots, at the row-spec slot.
/// 0 would freeze the initial random net forever, so it is refused instead of silently meaning off.
const default_target_every: usize = 100;

/// (#300) TD(lambda) of the `rlam` replay sweep when index 13 is omitted or `-`.
const default_sweep_lambda: f64 = 0.7;

fn parseTargetEvery(args: []const []const u8) !usize {
    if (args.len <= 13) return default_target_every;
    if (std.mem.eql(u8, args[13], "-")) return default_target_every;
    const every = std.fmt.parseInt(usize, args[13], 10) catch return error.InvalidTargetEvery;
    if (every == 0) return error.InvalidTargetEvery;
    return every;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    // `zig build nn` without `--` used to run the defaults (2000 games, write weights_path = the shipped
    // file). It silently retrained the shipped net twice (#207), so no arguments now means "show me how".
    if (args.len <= 1) {
        std.debug.print("{s}", .{usage});
        return;
    }
    const train_games: u32 = try cli.argOr(u32, args, 1, default_train_games);
    const eval_games: u32 = try cli.argOr(u32, args, 2, default_eval_games);
    const eval_seed: u64 = try cli.argOr(u64, args, 3, default_eval_seed);
    const learning_rate: f64 = try cli.argOr(f64, args, 4, default_learning_rate);
    // arg 5 is either the seed-range count or the literal "diag" ("-" = no diag, 1 range)
    const arg5: ?[]const u8 = if (args.len > 5) args[5] else null;
    const want_diag = arg5 != null and std.mem.eql(u8, arg5.?, "diag");
    const no_diag = arg5 != null and std.mem.eql(u8, arg5.?, "-");
    const seed_ranges: u32 = if (arg5) |a| (if (want_diag or no_diag) 1 else try std.fmt.parseInt(u32, a, 10)) else default_seed_ranges;
    const variant_name = if (args.len > 6) args[6] else "base";
    // arg 6 `par` means base + parallel (unchanged); arg 10 `par` batches the variant of arg 6
    const par_tail = try parseParFlag(args);
    const spec = findVariant(variant_name) orelse VariantSpec{ .variant = .unknown };
    const variant = @tagName(spec.variant);
    const use_par = spec.variant == .par or par_tail;
    const is_mstd = spec.variant == .mstd;
    const is_death = spec.variant == .death;
    const is_oi = spec.variant == .oi;
    const is_oi16 = spec.variant == .oi16;
    // `dump` and `dumpp2` record the same buffer (#159, #175) and are refused under `par` alike; their
    // options differ in the self-play policy and live in `buildDump`/`buildDumpP2`.
    const is_dump = spec.variant == .dump or spec.variant == .dumpp2;
    const is_ft2 = spec.variant == .ft2;
    const is_restart = spec.variant == .restart;
    // Paired comparison (#168): two policies play the same spawn tape.
    const is_pair = spec.variant == .pair;
    // Paired comparison of two weight files (#210): A = weights_path, B = the file at index 12, both with
    // the same policy, so a candidate net is judged against the shipped one on identical spawns.
    const is_pairw = spec.variant == .pairw;
    const is_pairleaf = spec.variant == .pairleaf;
    const is_pairrow = spec.variant == .pairrow;
    const is_leaf = spec.variant == .leaf;
    const leaf_mode: nn.Mode = if (is_pairleaf) parsePairLeafMode(if (args.len > 11) args[11] else null) orelse {
        std.debug.print("pairleaf mode must be deep or deep3\\n{s}", .{usage});
        return;
    } else .deep;
    const is_tch = spec.variant == .tch or spec.variant == .tchadd;
    const is_tchadd = spec.variant == .tchadd;
    const is_dist = spec.variant == .dist;
    const is_dson = spec.variant == .dson;
    // (#231) `par` only: Adam steps on the merged batch gradient, refused below without it.
    const is_adam = spec.variant == .adam;
    // (#262) replay with a frozen bootstrap net: a `par` variable, see `TargetSync`.
    const is_tnet = spec.variant == .tnet;
    const target_every: usize = if (is_tnet) try parseTargetEvery(args) else 0;
    // (#246) the step ceiling: `norm` bounds the merged `par` delta, `step` every element of a step.
    const is_clip = spec.variant == .clip;
    const clip_mode: ClipMode = if (is_clip) try parseClipMode(args) else .norm;
    // (#206) the teacher mix: `distill_every` (arg 12) and the rate divisor (arg 13). Handed to the
    // `dist`/`dson` builder instead of parsed there, to keep the old parse order (see `BuildCtx`).
    const dist_mix: DistillMix = if (is_dist or is_dson) try parseDistillMix(args) else .{};
    // `use_resume` only decides whether `weights_path` is loaded; the variant's options come from its own
    // builder, never from this flag.
    const use_resume = spec.load_weights;
    // Variants that take dump_path at index 11: `dump` writes it, `ft2` and `restart` read it.
    const uses_dump = spec.dump;
    const replay_k: usize = try cli.argOr(usize, args, 7, default_replay_k);
    const weights_path: []const u8 = try cli.argOr([]const u8, args, 8, default_weights_path);
    const train_seed: u64 = try cli.argOr(u64, args, 9, default_train_seed);
    // Deep-position buffer (#159). Index 10 must say "par" or "-" (see `parseParFlag`); index 11 is the
    // dump path.
    const dump_path: ?[]const u8 = if (args.len > 11) args[11] else null;
    var dump: nn.Dump = .{};
    var priority: nn.Priority = .{};
    var adam: nn.Adam = .{};
    // (#262) Declared before `opts`, which points at its `net`; initialised from the live net after
    // `startNet` and refreshed between batches only.
    var target_sync: TargetSync = undefined;
    // `nn.Teachers` is ~40 MB (32 dumps worth, #207), so it is heap allocated, unlike the small buffers.
    const teachers = try init.gpa.create(nn.Teachers);
    defer init.gpa.destroy(teachers);
    teachers.len = 0;
    teachers.draws = 0;
    // (#137) `par` merges whole-game deltas into one step; arg 12 scales that step (1 = historical mean).
    const batch_scale: f64 = if (use_par) try parseBatchScale(args) else 1.0;
    // Every variant owns its `nn.Options`: `findVariant` picked the entry and its `build` reads the
    // command line and the buffers above. `.unknown` has no builder, so a name no entry matched still
    // prints the usage here, where the old chain's final `else` did.
    const build = spec.build orelse {
        std.debug.print("{s}", .{usage});
        return;
    };
    var opts: nn.Options = try build(&.{
        .args = args,
        .replay_k = replay_k,
        .learning_rate = learning_rate,
        .adam = &adam,
        .priority = &priority,
        .dump = &dump,
        .teachers = teachers,
        .target_net = &target_sync.net,
        .clip_mode = clip_mode,
        .dist_mix = dist_mix,
    });
    // (#137) the merge multiplier applies to every variant that reached a batched path; the builders never
    // set it, so it is assigned once here for all of them.
    opts.batch_scale = batch_scale;
    // (#277) Refuse a shaped run mixed with a variant that retrains afterstates later or replaces the
    // TD target: those paths do not implement the shaped reward, and silently training on the raw
    // reward would hide the mistake.
    nn.validateOptions(opts) catch {
        std.debug.print("{s}", .{usage});
        return;
    };
    // `dump` is the shared layout plus one slot: index 11 carries the dump path.
    // `par` runs may carry the #137 batch_scale at index 12, after the `-` dump slot.
    // (#203) index 13 narrows the eval table to the rows a measurement needs (`0,2`). The variants that
    // own 13 for another parameter are marked `rows = false` in `variant_specs` (including `lam`/`rlam`,
    // whose current double-read is deliberately preserved).
    const row_spec: ?[]const u8 = if (spec.rows and args.len > 13) args[13] else null;
    const max_args = maxArgs(spec, row_spec, use_par);
    if (args.len > max_args or seed_ranges == 0 or (uses_dump and dump_path == null) or ((is_pairleaf or is_leaf or is_pairrow) and train_games != 0)) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // the priority ring buffer, the dump, the deep mix and the restart buffer are cross-game state:
    // batching would race on them. `mstd` has no batch path either (one staged net is ~1 MB).
    if (use_par and (!parSupported(opts, is_mstd) or opts.dump_record or opts.deep_mix or opts.restart or opts.distill != null)) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // (#231) Adam steps on the merged batch gradient only, so the variant needs `par` at arg 10.
    if (is_adam and !use_par) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // (#246) the norm ceiling is on the delta one merged batch builds, which only `par` has.
    if (is_clip and clip_mode == .norm and !use_par) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // (#262) the frozen snapshot is per merged batch, so the serial loop has no equivalent interval.
    if (is_tnet and !use_par) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // `dist` has no replay / death / restart / dump half to combine with: refuse instead of ignoring.
    // `dist` keeps the plain TD half (so replay applies to it); `dson` has no TD half, so a replay
    // buffer would be dropped on the floor.
    if ((is_dson and replay_k > 0) or ((is_dist or is_dson) and (is_death or is_restart or is_dump))) {
        std.debug.print("{s}", .{usage});
        return;
    }
    if (is_ft2 or is_restart or is_tch) loadDump(&dump, init.io, init.gpa, dump_path.?);
    if (is_dist or is_dson) loadTeachers(teachers, init.io, init.gpa, dump_path.?);

    var net: nn.Net = undefined;
    // `nn.Staged` is 7 nets (~2 MB at the 128/32 default), too much for the stack (#233).
    const staged = try init.gpa.create(nn.Staged);
    defer init.gpa.destroy(staged);
    const loaded = if (is_mstd)
        try startStaged(staged, init.io, init.gpa, train_games, train_seed, weights_path)
    else
        try startNet(&net, init.io, init.gpa, train_games, use_resume, train_seed, weights_path);
    if (loaded) std.debug.print("loaded {s}\n", .{weights_path});
    // (#262) The initial snapshot is the starting net; the first batch trains against it.
    if (is_tnet) target_sync = TargetSync.init(&net, target_every);
    // Optimistic initialization (#107) only rescales the random init, so it stays out of `nn.Options`.
    if (is_oi) nn.scaleInit(&net, 4) else if (is_oi16) nn.scaleInit(&net, 16);
    if (is_tch) {
        // #175 teacher pass: price every stored deep afterstate once, write `<dump_path>.tch`, train nothing.
        // #207 `tchadd` appends to that file instead of replacing it, so N dumps give one N x 65,536 teacher.
        const path = try std.fmt.allocPrint(init.arena.allocator(), "{s}.tch", .{dump_path.?});
        if (is_tchadd) loadTeachers(teachers, init.io, init.gpa, path);
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        const extra = try init.gpa.create(nn.Teachers);
        defer init.gpa.destroy(extra);
        nn.computeTeachers(&net, extra, &dump, rng.random(), try parseTeacherMode(args));
        const pass_len = extra.len;
        const dropped = teachers.appendBuffer(extra);
        try teachers.save(init.io, init.gpa, path);
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("teacher pass: {d} samples in {d:.1} s ({d:.1} ms/sample), file {s}: {d} samples, {d} dropped\n", .{
            pass_len,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            @as(f64, @floatFromInt(elapsed)) / 1e6 / @max(@as(f64, @floatFromInt(pass_len)), 1),
            path,
            teachers.len,
            dropped,
        });
        return;
    }
    if (train_games > 0) {
        var rng = std.Random.DefaultPrng.init(train_seed);
        const start = std.Io.Clock.awake.now(init.io);
        var last_score: u64 = 0;
        var total_moves: u64 = 0;
        var diag: nn.Diag = .{};
        if (is_mstd) {
            for (0..train_games) |g| {
                const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
                const game = nn.trainGameStaged(staged, rng.random(), learning_rate * (1 - 0.9 * frac));
                last_score = game.score;
                total_moves += game.moves;
                if (game.capped) return moveCap(g + 1, train_games, game.moves, nn.default_move_cap, weights_path);
                if (!stagedFinite(staged)) return diverged(g + 1, train_games, weights_path);
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
                const batch = try nn.trainBatchOpts(&net, train_seed, g, games, learning_rate * (1 - 0.9 * frac), if (want_diag) &diag else null, opts);
                last_score = batch.score;
                total_moves += batch.moves;
                // (#242) One pathological game stalls the whole batch (seed 3 spent 53 min in one at 60k),
                // so stop on the first capped game instead of waiting for the other seven.
                if (batch.capped_game) |i| return moveCap(g + i + 1, train_games, batch.capped_moves, opts.move_cap, weights_path);
                if (!nn.weightsFinite(&net)) return diverged(g + games, train_games, weights_path);
                // (#262) Refresh only after the merged batch is complete: every game in it saw the
                // same target, and no thread observed a half-written copy.
                if (is_tnet) target_sync.afterBatch(&net);
                if ((g + games) % 5000 == 0) {
                    std.debug.print("train {d}/{d} last game score {d}\n", .{ g + games, train_games, last_score });
                    try nn.save(&net, init.io, weights_path); // checkpoint
                }
            }
        } else {
            for (0..train_games) |g| {
                // linear decay to 10% of the start rate: big steps early, fine tuning late
                const frac = @as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(train_games));
                const game = try nn.trainGameOpts(&net, rng.random(), learning_rate * (1 - 0.9 * frac), if (want_diag) &diag else null, opts);
                last_score = game.score;
                total_moves += game.moves;
                if (game.capped) return moveCap(g + 1, train_games, game.moves, opts.move_cap, weights_path);
                if (!nn.weightsFinite(&net)) return diverged(g + 1, train_games, weights_path);
                if ((g + 1) % 5000 == 0) {
                    std.debug.print("train {d}/{d} last game score {d}\n", .{ g + 1, train_games, last_score });
                    try nn.save(&net, init.io, weights_path); // checkpoint
                }
            }
        }
        const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const variant_label = label: {
            // (#246) `clip` has two ceilings, so the log names the one that ran.
            const base_label = if (is_clip) try std.fmt.allocPrint(init.arena.allocator(), "{s}:{s}", .{ variant, @tagName(clip_mode) }) else if (is_tnet) try std.fmt.allocPrint(init.arena.allocator(), "tnet:{d}", .{target_every}) else variant;
            break :label if (par_tail) try std.fmt.allocPrint(init.arena.allocator(), "{s}+par", .{base_label}) else base_label;
        };
        std.debug.print("trained {d} games, {d} moves in {d:.1} s (lr={d}, seed={d}, variant={s})\n", .{
            train_games,
            total_moves,
            @as(f64, @floatFromInt(elapsed)) / 1e9,
            learning_rate,
            train_seed,
            variant_label,
        });
        if (want_diag and !is_mstd) printDiag(&diag);
        if (is_dist or is_dson) std.debug.print("distill: {d} teacher samples, {d} regression steps\n", .{ teachers.len, teachers.draws });
        if (is_dump) {
            const path = dump_path.?;
            try dump.save(init.io, init.gpa, path);
            std.debug.print("dump: {d} deep afterstates ({d} dropped) -> {s}\n", .{ dump.len, dump.dropped, path });
        }
        if (is_mstd) try staged.saveFile(init.io, init.gpa, weights_path) else try nn.save(&net, init.io, weights_path);
    }

    // `eval_games = 0` is "train only": the table below plays every row on every eval game, and one
    // `deep3`/iterative-deepening row can cost minutes, so a short training run must be able to skip it.
    if (eval_games == 0) return;

    // Rows are all on by default; `rows` (arg 13, #203) keeps the two a measurement needs
    // (0 = 1-ply, 2 = 2-ply) instead of paying for the 3-ply rows twice over. A stray token is a
    // command-line error, never a silent fallback to the full table.
    var row_selection: [all_rows.len]bool = @splat(true);
    if (row_spec) |rows_arg| {
        if (!std.mem.eql(u8, rows_arg, "-"))
            parseRows(&row_selection, rows_arg) catch {
                std.debug.print("error: rows must be a comma list of row indices (e.g. 0,2)\n{s}", .{usage});
                return;
            };
    }
    var results_buf: [all_rows.len]Eval = undefined;
    var result_count: usize = 0;
    for (all_rows, 0..) |row, k| {
        if (!row_selection[k]) continue;
        results_buf[result_count] = .{ .name = row.name, .spec = row.spec };
        result_count += 1;
    }
    const results = results_buf[0..result_count];
    const games = eval_games * seed_ranges;
    // `mstd` evaluates through the staged value, everything else through the single net (#107)
    const eval_value: nn.Value = if (is_mstd) .{ .staged = staged } else .{ .single = &net };

    // Paired comparison (#168): play one game with A (`deep`), then replay the same spawns with B
    // (`deep3`), so both see identical tiles. The per-game difference has much less noise than two
    // independent runs. Opt-in with variant `pair` (train_games 0, eval as usual).
    // (#210) A second net, so `pairw` can put a candidate file against the reference on the same spawns.
    var net_b: nn.Net = undefined;
    const pair_b_value: nn.Value = if (is_pairw) blk: {
        // The second path sits in the dump slot (index 11): `pairw` has no dump to read.
        if (args.len <= 11) {
            std.debug.print("pairw needs the second weights path after the `-` par slot\n{s}", .{usage});
            return;
        }
        try nn.load(&net_b, init.io, init.gpa, args[11]);
        break :blk .{ .single = &net_b };
    } else eval_value;
    var pair_spec_a: ?NnEvalRow = null;
    var pair_spec_b: ?NnEvalRow = null;
    var pair_name_a: []const u8 = "";
    var pair_name_b: []const u8 = "";
    if (is_pairrow) {
        if (args.len <= 12) {
            std.debug.print("pairrow needs two eval row indices after the `-` par slot (e.g. `- 15 17`)\n{s}", .{usage});
            return;
        }
        const row_a = parsePairRow(args[11]) orelse {
            std.debug.print("pairrow row A must be an eval row index, got `{s}`\n{s}", .{ args[11], usage });
            return;
        };
        const row_b = parsePairRow(args[12]) orelse {
            std.debug.print("pairrow row B must be an eval row index, got `{s}`\n{s}", .{ args[12], usage });
            return;
        };
        if (row_a >= all_rows.len or row_b >= all_rows.len) {
            std.debug.print("pairrow rows must be 0..{d}, got {d} and {d}\n{s}", .{ all_rows.len - 1, row_a, row_b, usage });
            return;
        }
        pair_name_a = all_rows[row_a].name;
        pair_name_b = all_rows[row_b].name;
        pair_spec_a = switch (all_rows[row_a].spec) {
            .nn => |nn_spec| nn_spec,
            .expectimax => {
                std.debug.print("pairrow row {d} ({s}) is not an NN search row\n{s}", .{ row_a, pair_name_a, usage });
                return;
            },
        };
        pair_spec_b = switch (all_rows[row_b].spec) {
            .nn => |nn_spec| nn_spec,
            .expectimax => {
                std.debug.print("pairrow row {d} ({s}) is not an NN search row\n{s}", .{ row_b, pair_name_b, usage });
                return;
            },
        };
    }
    if (is_pair or is_pairw or is_pairleaf or is_pairrow) {
        const pair_games = eval_games * seed_ranges;
        const spawns = try init.gpa.alloc(nn.Spawn, 20_000);
        defer init.gpa.free(spawns);
        const diffs = try init.gpa.alloc(f64, pair_games);
        defer init.gpa.free(diffs);
        var pair_rng = std.Random.DefaultPrng.init(eval_seed);
        const start_pair = std.Io.Clock.awake.now(init.io);
        var a_ns: u64 = 0;
        var b_ns: u64 = 0;
        var a_moves: u64 = 0;
        var b_moves: u64 = 0;
        var a_score: u64 = 0;
        var b_score: u64 = 0;
        var a_leaves: u64 = 0;
        var b_leaves: u64 = 0;
        var a_reach_8192: usize = 0;
        var b_reach_8192: usize = 0;
        for (0..pair_games) |i| {
            var tape = nn.Tape.initRecord(spawns);
            const pair_mode = if (is_pairleaf) leaf_mode else .deep;
            const a_start = std.Io.Clock.awake.now(init.io);
            const a = if (is_pairrow)
                nn.playTapedValueOpts(eval_value, pair_rng.random(), pair_spec_a.?.mode, &tape, pair_spec_a.?.opts)
            else
                nn.playTapedValue(eval_value, pair_rng.random(), pair_mode, &tape);
            const a_ns_i: u64 = @intCast(a_start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds());
            var replay = nn.Tape.initReplay(spawns[0..tape.len]);
            const b_start = std.Io.Clock.awake.now(init.io);
            const b = if (is_pairrow)
                nn.playTapedValueOpts(eval_value, pair_rng.random(), pair_spec_b.?.mode, &replay, pair_spec_b.?.opts)
            else if (is_pairw)
                nn.playTapedValue(pair_b_value, pair_rng.random(), .deep, &replay)
            else if (is_pairleaf)
                nn.playTapedValueOpts(eval_value, pair_rng.random(), leaf_mode, &replay, .{ .omit_leaf_reward = true })
            else
                nn.playTapedValue(eval_value, pair_rng.random(), .deep3, &replay);
            const b_ns_i: u64 = @intCast(b_start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds());
            diffs[i] = @as(f64, @floatFromInt(b.score)) - @as(f64, @floatFromInt(a.score));
            a_ns += a_ns_i;
            b_ns += b_ns_i;
            a_moves += a.moves;
            b_moves += b.moves;
            a_score += a.score;
            b_score += b.score;
            a_leaves += a.leaves;
            b_leaves += b.leaves;
            if (a.max_tile >= 8192) a_reach_8192 += 1;
            if (b.max_tile >= 8192) b_reach_8192 += 1;
            if (is_pairrow) {
                const a_ms = @as(f64, @floatFromInt(a_ns_i)) / 1e6 / @as(f64, @floatFromInt(a.moves));
                const b_ms = @as(f64, @floatFromInt(b_ns_i)) / 1e6 / @as(f64, @floatFromInt(b.moves));
                std.debug.print("pair {d}: A={d} ({d}) B={d} ({d}) diff={d:.0} A={d:.2} ms/move B={d:.2} ms/move\n", .{ i, a.score, a.moves, b.score, b.moves, diffs[i], a_ms, b_ms });
            } else {
                std.debug.print("pair {d}: A={d} ({d}) B={d} ({d}) diff={d:.0}\n", .{ i, a.score, a.moves, b.score, b.moves, diffs[i] });
            }
        }
        const st = nn.pairStats(diffs);
        const elapsed = start_pair.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        std.debug.print("pair A vs B: games={d} mean diff={d:.1} ± {d:.1} (stderr) better={d} worse={d} elapsed={d:.1} s\n", .{ st.games, st.mean, st.stderr, st.better, st.worse, @as(f64, @floatFromInt(elapsed)) / 1e9 });
        if (is_pairrow) {
            const n: f64 = @floatFromInt(pair_games);
            const a_ms = @as(f64, @floatFromInt(a_ns)) / 1e6 / @as(f64, @floatFromInt(a_moves));
            const b_ms = @as(f64, @floatFromInt(b_ns)) / 1e6 / @as(f64, @floatFromInt(b_moves));
            const a_lpm = @as(f64, @floatFromInt(a_leaves)) / @as(f64, @floatFromInt(a_moves));
            const b_lpm = @as(f64, @floatFromInt(b_leaves)) / @as(f64, @floatFromInt(b_moves));
            std.debug.print("pairrow A={s}: avg={d:.1} reach8192={d}/{d} leaves/move={d:.1} ms/move={d:.2}\n", .{ pair_name_a, @as(f64, @floatFromInt(a_score)) / n, a_reach_8192, pair_games, a_lpm, a_ms });
            std.debug.print("pairrow B={s}: avg={d:.1} reach8192={d}/{d} leaves/move={d:.1} ms/move={d:.2}\n", .{ pair_name_b, @as(f64, @floatFromInt(b_score)) / n, b_reach_8192, pair_games, b_lpm, b_ms });
        }
        return;
    }
    const EvalCtx = struct { vfn: nn.Value, spec: EvalRowSpec, eval_seed: u64, opts: nn.Options };
    // Game i uses seed eval_seed + i (= eval_seed + range*eval_games + g); games run in parallel.
    const playGame = struct {
        fn f(c: EvalCtx, i: usize) nn.PlayResult {
            var prng = std.Random.DefaultPrng.init(c.eval_seed + i);
            return switch (c.spec) {
                .expectimax => |depth| blk: {
                    const b = solver.playPolicy(prng.random(), .expectimax, depth);
                    break :blk nn.PlayResult{ .score = b.score, .max_tile = b.max_tile, .moves = b.moves };
                },
                .nn => |row| blk: {
                    var row_opts = row.opts;
                    // (#260) `leaf` is eval-only: rows 2 and 4 take its omit-leaf option, and row 4
                    // keeps its TT-off baseline from the table.
                    if (row.leaf_override) row_opts.omit_leaf_reward = c.opts.omit_leaf_reward;
                    break :blk nn.playValueOpts(c.vfn, prng.random(), row.mode, row_opts);
                },
            };
        }
    }.f;
    for (results) |*r| {
        r.range_scores = try init.arena.allocator().alloc(f64, seed_ranges);
        const res_all = try init.arena.allocator().alloc(nn.PlayResult, games);
        const start = std.Io.Clock.awake.now(init.io);
        parallel.map(nn.PlayResult, res_all, EvalCtx{ .vfn = eval_value, .spec = r.spec, .eval_seed = eval_seed, .opts = opts }, playGame);
        stats.rangeTotals(nn.PlayResult, Eval, r, res_all, eval_games);
        r.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
    }
    const last_seed = eval_seed + seed_ranges * eval_games - 1;
    std.debug.print("eval: games={d}/range ranges={d} seeds={d}..{d}\n", .{ eval_games, seed_ranges, eval_seed, last_seed });
    for (results) |r| {
        const st = stats.meanStderr(r.range_scores);
        // `avg score` is the mean over seed ranges (not the pooled average), which is exactly what
        // `avg_score` of the shared row holds. The reach counts are this runner's own tail (#141).
        bench.printEvalRow(r.name, st.stderr, seed_ranges, .{
            .games = games,
            .avg_score = st.mean,
            .avg_moves = r.avg_moves,
            .best_tile = r.best_tile,
            .elapsed_ns = r.elapsed_ns,
            .reach_4096 = r.reach_4096,
            .reach_8192 = r.reach_8192,
            .reach_16384 = r.reach_16384,
            .dead_below_2048 = r.dead_below_2048,
        }, r.total_score, r.leaves);
        std.debug.print(" reach4096={d}/{d} reach8192={d}/{d} reach16384={d}/{d} dead<2048={d}/{d}\n", .{ r.reach_4096, games, r.reach_8192, games, r.reach_16384, games, r.dead_below_2048, games });
        if (r.node_probes > 0) {
            const probes: f64 = @floatFromInt(r.node_probes);
            std.debug.print(" nodeTT probes={d} hits={d} hit_rate={d:.1}%\n", .{ r.node_probes, r.node_hits, 100 * @as(f64, @floatFromInt(r.node_hits)) / probes });
        }
    }
}

test "pairrow reads two eval row indices and maps row 17 to deep4 (#267)" {
    try std.testing.expectEqual(@as(?usize, 15), parsePairRow("15"));
    try std.testing.expectEqual(@as(?usize, 17), parsePairRow(" 17 "));
    try std.testing.expectEqual(@as(?usize, null), parsePairRow("deep4"));
    try std.testing.expectEqual(@as(?usize, null), parsePairRow(null));
    try std.testing.expectEqual(nn.Mode.deep3, all_rows[15].spec.nn.mode);
    try std.testing.expectEqual(nn.Mode.deep4, all_rows[17].spec.nn.mode);
    try std.testing.expectEqual(@as(?bool, true), all_rows[17].spec.nn.opts.node_tt);
    try std.testing.expectEqual(@as(u8, 2), all_rows[3].spec.expectimax);
}

test "high spawn-cut rows 18 and 19 keep the default row 15 path (#281)" {
    try std.testing.expectEqual(nn.Mode.deep3, all_rows[15].spec.nn.mode);
    try std.testing.expectEqual(@as(?f64, null), all_rows[15].spec.nn.opts.cutoff);
    try std.testing.expectEqual(nn.Mode.deep3, all_rows[18].spec.nn.mode);
    try std.testing.expectEqual(@as(?f64, 0.02), all_rows[18].spec.nn.opts.cutoff);
    try std.testing.expectEqual(@as(?bool, true), all_rows[18].spec.nn.opts.node_tt);
    try std.testing.expectEqual(nn.Mode.deep3, all_rows[19].spec.nn.mode);
    try std.testing.expectEqual(@as(?f64, 0.05), all_rows[19].spec.nn.opts.cutoff);
    try std.testing.expectEqual(@as(?bool, true), all_rows[19].spec.nn.opts.node_tt);
}

test "recovered cut branch row 20 stays the default deep3 path (#261)" {
    try std.testing.expectEqual(nn.Mode.deep3_recover, all_rows[20].spec.nn.mode);
    try std.testing.expectEqual(@as(?f64, null), all_rows[20].spec.nn.opts.cutoff);
}

test "eval row table is the single row source (#310)" {
    try std.testing.expectEqual(@as(usize, 27), all_rows.len);
    try std.testing.expectEqualStrings("nn 1-ply", all_rows[0].name);
    try std.testing.expectEqualStrings("nn 3-ply + node TT default (#266)", all_rows[15].name);
    try std.testing.expectEqualStrings("nn 3-ply cut branches recovered 1 ply shallow (#261)", all_rows[20].name);
    try std.testing.expectEqualStrings("nn 4-ply spawn sample 2, full last spawn layer (#320)", all_rows[25].name);
}

test "spawn-sample rows 21..26 only change the sample (#320)" {
    for ([_]usize{ 2, 3, 4, 6 }, 0..) |sample, i| {
        const row = all_rows[21 + i].spec.nn;
        try std.testing.expectEqual(nn.Mode.deep4, row.mode);
        try std.testing.expectEqual(@as(?usize, sample), row.opts.sample);
        try std.testing.expectEqual(@as(u8, 1), row.opts.sample_min_depth);
        // the cut and the node table are the ones `deep4` already uses
        try std.testing.expectEqual(@as(?f64, null), row.opts.cutoff);
        try std.testing.expectEqual(@as(?bool, null), row.opts.node_tt);
    }
    // rows 25/26 spend the same budget on the two deepest layers and keep the last spawn layer exact
    for ([_]usize{ 2, 3 }, 0..) |sample, i| {
        const row = all_rows[25 + i].spec.nn;
        try std.testing.expectEqual(nn.Mode.deep4, row.mode);
        try std.testing.expectEqual(@as(?usize, sample), row.opts.sample);
        try std.testing.expectEqual(@as(u8, 2), row.opts.sample_min_depth);
    }
}

test "leaf keeps its omit-leaf override on rows 2 and 4 (#260)" {
    try std.testing.expect(all_rows[2].spec.nn.leaf_override);
    try std.testing.expect(all_rows[4].spec.nn.leaf_override);
    try std.testing.expectEqual(@as(?bool, false), all_rows[4].spec.nn.opts.node_tt);
}

test "usage text lists the pairrow variant (#267)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "pairrow") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "15") != null);
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

/// `mstd` keeps seven nets, so the divergence check has to look at every stage (#234).
fn stagedFinite(staged: *const nn.Staged) bool {
    for (0..nn.stage_count) |k| {
        if (!nn.weightsFinite(&staged.nets[k])) return false;
    }
    return true;
}

/// Stop before the checkpoint write: a diverged run must leave the last good weights file in place (#234).
fn diverged(done: usize, total: usize, weights_path: []const u8) error{Diverged} {
    std.debug.print("diverged at {d}/{d} games (non-finite or |w| > 1e6), not saving {s}\n", .{ done, total, weights_path });
    return error.Diverged;
}

/// (#242) Same policy for a game that hit the move cap: report which game and how many moves, then
/// stop without overwriting the last good checkpoint. The game is pathological, not a result to keep.
fn moveCap(done: usize, total: usize, moves: u32, cap: u32, weights_path: []const u8) error{MoveCap} {
    std.debug.print("game {d}/{d} hit the {d}-move cap at {d} moves, stopping the run and not saving {s}\n", .{ done, total, cap, moves, weights_path });
    return error.MoveCap;
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
    // re-recorded at #233: the shipped width changed (64/32 -> 128/32), so the same 5 games land elsewhere
    try std.testing.expectEqual(@as(u32, 3379822721), std.hash.Crc32.hash(std.mem.asBytes(&net_default)));
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

test "pairleaf accepts deep and deep3 and defaults to deep (#260)" {
    try std.testing.expectEqual(nn.Mode.deep, parsePairLeafMode(null).?);
    try std.testing.expectEqual(nn.Mode.deep, parsePairLeafMode("deep").?);
    try std.testing.expectEqual(nn.Mode.deep3, parsePairLeafMode("deep3").?);
    try std.testing.expectEqual(@as(?nn.Mode, null), parsePairLeafMode("greedy"));
}

test "usage text lists the leaf-reward evaluation modes (#260)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "pairleaf      (#260)") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "leaf          (#260)") != null);
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

test "omitted batch_scale is the sqrt rule, '-' is the old mean (#203)" {
    const omitted = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3", "par", "-" };
    try std.testing.expectEqual(defaultBatchScale(), try parseBatchScale(&omitted));
    // The rule itself is checked at a fixed batch size; trainBatchSize() is host-dependent, so the
    // default is then checked against whatever this machine reports.
    try std.testing.expectApproxEqAbs(@as(f64, 2.8284271247461903), batchScaleFor(8), 1e-12);
    try std.testing.expectEqual(batchScaleFor(nn.trainBatchSize()), defaultBatchScale());
    const dash = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3", "par", "-", "-" };
    try std.testing.expectEqual(@as(f64, 1), try parseBatchScale(&dash));
    const explicit = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3", "par", "-", "2.8284" };
    try std.testing.expectEqual(@as(f64, 2.8284), try parseBatchScale(&explicit));
    const typo = [_][]const u8{ "nn", "2000", "20", "2000", "0.003", "-", "base", "30", "x.bin", "3", "par", "-", "two" };
    try std.testing.expectError(error.InvalidBatchScale, parseBatchScale(&typo));
}

test "clamp variant reads its TD target ceiling from the row-spec slot (#234)" {
    const omitted = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clamp", "30", "x.bin", "3", "par" };
    try std.testing.expectEqual(@as(f64, 200), try parseClamp(&omitted));
    const dash = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clamp", "30", "x.bin", "3", "par", "-", "-", "-" };
    try std.testing.expectEqual(@as(f64, 200), try parseClamp(&dash));
    const explicit = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clamp", "30", "x.bin", "3", "par", "-", "2.8284", "50" };
    try std.testing.expectEqual(@as(f64, 50), try parseClamp(&explicit));
    const zero = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clamp", "30", "x.bin", "3", "par", "-", "-", "0" };
    try std.testing.expectError(error.InvalidClamp, parseClamp(&zero));
}

test "clip variant reads its step ceiling and mode from args 13 and 14 (#246)" {
    const omitted = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clip", "30", "x.bin", "3", "par" };
    try std.testing.expectEqual(default_clip_cap, try parseStepCap(&omitted));
    try std.testing.expectEqual(ClipMode.norm, try parseClipMode(&omitted));
    // the ceiling of a step is much smaller than the unit norm a batch delta is capped to
    const step = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clip", "30", "x.bin", "3", "par", "-", "2.8284", "0.001", "step" };
    try std.testing.expectEqual(@as(f64, 0.001), try parseStepCap(&step));
    try std.testing.expectEqual(ClipMode.step, try parseClipMode(&step));
    const dash = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clip", "30", "x.bin", "3", "par", "-", "-", "-", "-" };
    try std.testing.expectEqual(default_clip_cap, try parseStepCap(&dash));
    try std.testing.expectEqual(ClipMode.norm, try parseClipMode(&dash));
    const zero = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clip", "30", "x.bin", "3", "par", "-", "-", "0" };
    try std.testing.expectError(error.InvalidStepCap, parseStepCap(&zero));
    const typo = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "clip", "30", "x.bin", "3", "par", "-", "-", "1", "l2" };
    try std.testing.expectError(error.InvalidClipMode, parseClipMode(&typo));
}

test "usage text lists the clip variant and its mode slot (#246)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "clip    (#246") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "`norm` (default)") != null);
}

test "tnet sync interval defaults to 100 and rejects zero (#262)" {
    const omitted = [_][]const u8{ "nn", "2000", "0", "1", "0.003", "-", "tnet", "30", "/tmp/w.bin", "1", "par" };
    try std.testing.expectEqual(default_target_every, try parseTargetEvery(&omitted));
    const dash = [_][]const u8{ "nn", "2000", "0", "1", "0.003", "-", "tnet", "30", "/tmp/w.bin", "1", "par", "-", "-", "-" };
    try std.testing.expectEqual(default_target_every, try parseTargetEvery(&dash));
    const thousand = [_][]const u8{ "nn", "2000", "0", "1", "0.003", "-", "tnet", "30", "/tmp/w.bin", "1", "par", "-", "-", "1000" };
    try std.testing.expectEqual(@as(usize, 1000), try parseTargetEvery(&thousand));
    const zero = [_][]const u8{ "nn", "2000", "0", "1", "0.003", "-", "tnet", "30", "/tmp/w.bin", "1", "par", "-", "-", "0" };
    try std.testing.expectError(error.InvalidTargetEvery, parseTargetEvery(&zero));
}

test "frozen target snapshot refreshes only on the interval (#262)" {
    var live = std.mem.zeroes(nn.Net);
    var next = live;
    next.b3 = 1;
    var sync = TargetSync.init(&live, 2);
    try std.testing.expectEqual(@as(f64, 0), sync.net.b3);
    sync.afterBatch(&next);
    // between refreshes the snapshot must not chase the live net
    try std.testing.expectEqual(@as(f64, 0), sync.net.b3);
    sync.afterBatch(&next);
    try std.testing.expectEqual(@as(f64, 1), sync.net.b3);
    // a change inside the next interval is invisible until its end
    next.b3 = 2;
    sync.afterBatch(&next);
    try std.testing.expectEqual(@as(f64, 1), sync.net.b3);
    sync.afterBatch(&next);
    try std.testing.expectEqual(@as(f64, 2), sync.net.b3);
}

test "usage text lists the frozen target variant and its sync slot (#262)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "tnet    (#262") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "sync interval") != null);
}

test "rows argument keeps the listed eval rows and rejects junk (#203)" {
    var sel: [15]bool = @splat(true);
    try parseRows(&sel, "0, 2");
    try std.testing.expect(sel[0] and sel[2]);
    try std.testing.expect(!sel[1] and !sel[3] and !sel[14]);
    try std.testing.expectError(error.InvalidRow, parseRows(&sel, "0,2,x"));
    try std.testing.expectError(error.InvalidRow, parseRows(&sel, "15"));
}

test "usage text documents the par flag for stateless variants" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "#99") != null);
}

test "usage text lists the symbatch variant" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "symbatch (#98") != null);
}

test "usage text lists the rsym variant (#271)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "rsym    (#271") != null);
}

test "usage text lists the rtgt variant (#273)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "rtgt    (#273") != null);
}

test "par is refused for the global-feature build (#226)" {
    // the default build has no features, so the guard is a no-op there; a `-Dfeat` build must refuse
    try std.testing.expectEqual(nn.feature_count != 0, !parSupported(.{}, false));
}

test "usage text lists the exploration variants (#224)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "explore (#224") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "open8   (#224") != null);
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

test "ft2 deep_every parses, defaults to 1 and rejects 0 (#207)" {
    const none = [_][]const u8{ "nn", "2000", "0", "9000", "0.0003", "1", "ft2", "30", "/tmp/w.bin", "1", "-", "/tmp/d.bin" };
    try std.testing.expectEqual(@as(u32, 1), try parseDeepEvery(&none));
    const dash = [_][]const u8{ "nn", "2000", "0", "9000", "0.0003", "1", "ft2", "30", "/tmp/w.bin", "1", "-", "/tmp/d.bin", "-" };
    try std.testing.expectEqual(@as(u32, 1), try parseDeepEvery(&dash));
    const eight = [_][]const u8{ "nn", "2000", "0", "9000", "0.0003", "1", "ft2", "30", "/tmp/w.bin", "1", "-", "/tmp/d.bin", "8" };
    try std.testing.expectEqual(@as(u32, 8), try parseDeepEvery(&eight));
    const zero = [_][]const u8{ "nn", "2000", "0", "9000", "0.0003", "1", "ft2", "30", "/tmp/w.bin", "1", "-", "/tmp/d.bin", "0" };
    try std.testing.expectError(error.InvalidDeepEvery, parseDeepEvery(&zero));
}

test "usage text lists the distillation variants (#167)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "d2      (#167") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "d3      (#167") != null);
}

test "par is refused for the priority buffer, the staged net and the teacher variants (#158, #228)" {
    var pr: nn.Priority = .{};
    // heap, because `Teachers` is ~40 MB
    const teachers = try std.testing.allocator.create(nn.Teachers);
    defer std.testing.allocator.destroy(teachers);
    try std.testing.expect(parSupported(.{}, false));
    try std.testing.expect(parSupported(.{ .symmetry = true, .replay_k = 30 }, false));
    // #271: rsym is one per-move update with a random input orientation, stateless like base
    try std.testing.expect(parSupported(.{ .rsym = true }, false));
    // #273: rtgt is still one per-move update, with the same random orientation on both sides
    try std.testing.expect(parSupported(.{ .rtgt = true }, false));
    try std.testing.expect(!parSupported(.{}, true));
    try std.testing.expect(!parSupported(.{ .priority = &pr }, false));
    // #228: t2/p2/tp2 are plain per-move runs (the value cache is thread-local and cleared per move),
    // so they batch like base
    try std.testing.expect(parSupported(.{ .target = .search2 }, false));
    try std.testing.expect(parSupported(.{ .selfplay = .deep2 }, false));
    try std.testing.expect(parSupported(.{ .target = .search2, .selfplay = .deep2 }, false));
    // #167: the teacher search is the whole cost, so a batch would only multiply it
    try std.testing.expect(!parSupported(.{ .teacher = .deep }, false));
    try std.testing.expect(!parSupported(.{ .teacher = .deep3 }, false));
    // `dist` is read-only, but batched it loses ~30,000 points over 20,000 games (#207).
    try std.testing.expect(!parSupported(.{ .distill = teachers, .distill_every = 8 }, false));
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
    const saved = try std.testing.allocator.create(nn.Staged);
    defer std.testing.allocator.destroy(saved);
    saved.* = nn.Staged.init(prng.random());
    try saved.saveFile(io, std.testing.allocator, path);

    // eval-only (train_games 0) and resume load the staged file byte for byte
    const resumed = try std.testing.allocator.create(nn.Staged);
    defer std.testing.allocator.destroy(resumed);
    try std.testing.expect(try startStaged(resumed, io, std.testing.allocator, 0, 41, path));
    for (saved.nets, resumed.nets) |a, b| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));

    // training without resume starts from a random init (seed 99, so it must differ)
    const fresh = try std.testing.allocator.create(nn.Staged);
    defer std.testing.allocator.destroy(fresh);
    try std.testing.expect(!try startStaged(fresh, io, std.testing.allocator, 5, 99, path));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&saved.nets), std.mem.asBytes(&fresh.nets)));
}

test "teacher mix parses the divisor and the teacher rate divides the TD rate (#207)" {
    const mix = try parseDistillMix(&.{ "nn", "0", "0", "9000", "0.0003", "1", "dist", "30", "w", "1", "-", "t.tch", "4", "3" });
    try std.testing.expectEqual(@as(u32, 4), mix.every);
    try std.testing.expectApproxEqAbs(@as(f64, 3), mix.div, 1e-12);
    // the teacher step is `learning_rate / div`, never `1 / div` (the #206 code passed 0.1 to `train`)
    try std.testing.expectApproxEqRel(@as(f64, 1e-4), distillRate(3e-4, 3), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 3e-5), distillRate(3e-4, 10), 1e-12);
    // divisor 0 / "-" keeps the teacher step at the TD rate (0 = "same as TD" in nn.Options)
    try std.testing.expectEqual(@as(f64, 0), distillRate(3e-4, 0));
    try std.testing.expectError(error.InvalidDistillLr, parseDistillMix(&.{ "nn", "0", "0", "9000", "0.0003", "1", "dist", "30", "w", "1", "-", "t.tch", "4", "-1" }));
}

test "teacher mode is deep3 by default and h4 on request (#207)" {
    try std.testing.expectEqual(nn.Mode.deep3, try parseTeacherMode(&.{"nn"}));
    try std.testing.expectEqual(nn.Mode.deep3, try parseTeacherMode(&.{ "nn", "0", "0", "9000", "0.0003", "1", "tch", "30", "w", "1", "-", "d.dmp" }));
    try std.testing.expectEqual(nn.Mode.deep3, try parseTeacherMode(&.{ "nn", "0", "0", "9000", "0.0003", "1", "tch", "30", "w", "1", "-", "d.dmp", "-" }));
    try std.testing.expectEqual(nn.Mode.phase_h4, try parseTeacherMode(&.{ "nn", "0", "0", "9000", "0.0003", "1", "tch", "30", "w", "1", "-", "d.dmp", "h4" }));
    try std.testing.expectEqual(nn.Mode.deep, try parseTeacherMode(&.{ "nn", "0", "0", "9000", "0.0003", "1", "tch", "30", "w", "1", "-", "d.dmp", "deep" }));
    try std.testing.expectError(error.InvalidTeacherMode, parseTeacherMode(&.{ "nn", "0", "0", "9000", "0.0003", "1", "tch", "30", "w", "1", "-", "d.dmp", "h5" }));
}

test "usage says no arguments must not retrain the shipped weights (#207)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "must not retrain") != null);
}

test "usage text lists the tchadd variant (#207)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "tchadd  (#207") != null);
}

test "shape variant reads its coefficient from the row-spec slot (#277)" {
    const omitted = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par" };
    try std.testing.expectEqual(@as(f64, 1), try parseShapeC(&omitted));
    const dash = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "-" };
    try std.testing.expectEqual(@as(f64, 1), try parseShapeC(&dash));
    const explicit = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "0.5" };
    try std.testing.expectEqual(@as(f64, 0.5), try parseShapeC(&explicit));
    // negative potential shaping is still potential-based shaping
    const negative = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "-0.25" };
    try std.testing.expectEqual(@as(f64, -0.25), try parseShapeC(&negative));
    const zero = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "0" };
    try std.testing.expectError(error.InvalidShapeC, parseShapeC(&zero));
    const infinite = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "inf" };
    try std.testing.expectError(error.InvalidShapeC, parseShapeC(&infinite));
    const typo = [_][]const u8{ "nn", "2000", "0", "2000", "0.003", "-", "shape", "30", "x.bin", "3", "par", "-", "-", "two" };
    try std.testing.expectError(error.InvalidShapeC, parseShapeC(&typo));
}

test "usage text lists the shape variant and its coefficient slot (#277)" {
    try std.testing.expect(std.mem.indexOf(u8, usage, "shape   (#277") != null);
    try std.testing.expect(std.mem.indexOf(u8, usage, "empty-cell potential") != null);
}

test "variant table lists every accepted variant name once (#314)" {
    const names = [_][]const u8{
        "base",   "backward", "replay",  "tnet", "priority", "midgame",   "midmix", "sym",
        "symlam", "symbatch", "rsym",    "rtgt", "explore",  "explore20", "open8",  "open16",
        "nstep",  "death",    "t2",      "p2",   "tp2",      "d2",        "d3",     "dumpp2",
        "tch",    "tchadd",   "dist",    "dson", "oi",       "oi16",      "resume", "mstd",
        "dump",   "ft2",      "restart", "adam", "tailw",    "clamp",     "shape",  "clip",
        "pairw",  "pairleaf", "pairrow", "leaf", "pair",     "par",       "lam",    "rlam",
    };
    try std.testing.expectEqual(names.len, variant_specs.len);
    for (variant_specs, 0..) |spec, i| {
        try std.testing.expectEqual(spec, findVariant(@tagName(spec.variant)).?);
        for (variant_specs[0..i]) |prev| {
            try std.testing.expect(!std.mem.eql(u8, @tagName(spec.variant), @tagName(prev.variant)));
        }
    }
    for (names) |name| try std.testing.expect(findVariant(name) != null);
    try std.testing.expect(findVariant("no-such-variant") == null);
}

test "variant table resume and dump sets match the old hand lists (#314)" {
    const resumes = [_][]const u8{
        "resume", "dump", "dumpp2", "ft2", "restart", "dist", "dson", "tch", "tchadd", "t2", "p2", "tp2", "rlam",
    };
    const dumps = [_][]const u8{
        "dump", "dumpp2", "ft2", "restart", "tch", "tchadd", "dist", "dson",
    };
    var resume_count: usize = 0;
    var dump_count: usize = 0;
    for (variant_specs) |spec| {
        var wants_resume = false;
        for (resumes) |name| {
            if (std.mem.eql(u8, name, @tagName(spec.variant))) wants_resume = true;
        }
        var wants_dump = false;
        for (dumps) |name| {
            if (std.mem.eql(u8, name, @tagName(spec.variant))) wants_dump = true;
        }
        try std.testing.expectEqual(wants_resume, spec.load_weights);
        try std.testing.expectEqual(wants_dump, spec.dump);
        if (spec.load_weights) resume_count += 1;
        if (spec.dump) dump_count += 1;
    }
    try std.testing.expectEqual(resumes.len, resume_count);
    try std.testing.expectEqual(dumps.len, dump_count);
}

test "variant table rows flag matches the old row_spec exclusion (#314)" {
    const no_rows = [_][]const u8{
        "dist", "dson", "adam", "tailw", "clamp", "shape", "tnet", "clip", "pairw", "pair", "pairleaf", "pairrow",
    };
    var no_rows_count: usize = 0;
    for (variant_specs) |spec| {
        var expects_no_rows = false;
        for (no_rows) |name| {
            if (std.mem.eql(u8, name, @tagName(spec.variant))) expects_no_rows = true;
        }
        try std.testing.expectEqual(!expects_no_rows, spec.rows);
        if (!spec.rows) no_rows_count += 1;
    }
    try std.testing.expectEqual(no_rows.len, no_rows_count);
}

test "variant table max_args matches the old if-chain (#314)" {
    const MaxExpect = struct { name: []const u8, no_rows: usize, no_rows_par: usize, rows: ?usize };
    const expected = [_]MaxExpect{
        .{ .name = "base", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "backward", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "replay", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "tnet", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "priority", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "midgame", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "midmix", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "sym", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "symlam", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "symbatch", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "rsym", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "rtgt", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "explore", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "explore20", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "open8", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "open16", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "nstep", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "death", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "t2", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "p2", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "tp2", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "d2", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "d3", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "dumpp2", .no_rows = 12, .no_rows_par = 12, .rows = 14 },
        .{ .name = "tch", .no_rows = 13, .no_rows_par = 13, .rows = 14 },
        .{ .name = "tchadd", .no_rows = 13, .no_rows_par = 13, .rows = 14 },
        .{ .name = "dist", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "dson", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "oi", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "oi16", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "resume", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "mstd", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "dump", .no_rows = 12, .no_rows_par = 12, .rows = 14 },
        .{ .name = "ft2", .no_rows = 13, .no_rows_par = 13, .rows = 14 },
        .{ .name = "restart", .no_rows = 12, .no_rows_par = 12, .rows = 14 },
        .{ .name = "adam", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "tailw", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "clamp", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "shape", .no_rows = 14, .no_rows_par = 14, .rows = null },
        .{ .name = "clip", .no_rows = 15, .no_rows_par = 15, .rows = null },
        .{ .name = "pairw", .no_rows = 12, .no_rows_par = 13, .rows = null },
        .{ .name = "pairleaf", .no_rows = 12, .no_rows_par = 13, .rows = null },
        .{ .name = "pairrow", .no_rows = 13, .no_rows_par = 13, .rows = null },
        .{ .name = "leaf", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "pair", .no_rows = 11, .no_rows_par = 13, .rows = null },
        .{ .name = "par", .no_rows = 11, .no_rows_par = 13, .rows = 14 },
        .{ .name = "lam", .no_rows = 14, .no_rows_par = 14, .rows = 14 },
        .{ .name = "rlam", .no_rows = 14, .no_rows_par = 14, .rows = 14 },
    };
    try std.testing.expectEqual(variant_specs.len, expected.len);
    for (expected) |e| {
        const spec = findVariant(e.name).?;
        try std.testing.expectEqual(e.no_rows, maxArgs(spec, null, false));
        try std.testing.expectEqual(e.no_rows_par, maxArgs(spec, null, true));
        if (e.rows) |rows_cap| {
            try std.testing.expect(spec.rows);
            try std.testing.expectEqual(rows_cap, maxArgs(spec, "0,2", false));
            try std.testing.expectEqual(rows_cap, maxArgs(spec, "0,2", true));
        } else {
            try std.testing.expect(!spec.rows);
        }
    }
}

test "lam and rlam still read index 13 as rows before lambda (#314)" {
    var selected: [all_rows.len]bool = @splat(false);
    for ([_][]const u8{ "lam", "rlam" }) |name| {
        const spec = findVariant(name).?;
        const row_spec: ?[]const u8 = if (spec.rows and 14 > 13) "0.7" else null;
        try std.testing.expectEqual(@as(usize, 14), maxArgs(spec, row_spec, false));
        try std.testing.expectError(error.InvalidRow, parseRows(&selected, row_spec.?));
    }
}

test "an unknown variant keeps the old parse order and usage path (#314)" {
    // The old code parsed the later slots before it noticed the name was unknown, then printed the
    // usage from the options chain's final `else`. `.unknown` carries every default fact so dispatch
    // runs the same path; this pins those facts and the cap the placeholder computes.
    const spec = VariantSpec{ .variant = .unknown };
    try std.testing.expect(!spec.load_weights and !spec.dump and spec.rows);
    try std.testing.expectEqual(@as(usize, 14), maxArgs(spec, "0,2", false));
    try std.testing.expectEqual(@as(usize, 13), maxArgs(spec, null, true));
    try std.testing.expectEqual(@as(usize, 11), maxArgs(spec, null, false));
    // A borrowed name (`unknown`) is not in the table either, so it resolves to the placeholder.
    try std.testing.expect(findVariant("unknown") == null);
}

/// Buffers one `BuildCtx` borrows, so a builder test can build options from a command line and compare
/// them against a literal.
const BuildFixture = struct {
    adam: nn.Adam = .{},
    priority: nn.Priority = .{},
    dump: nn.Dump = .{},
    teachers: *nn.Teachers,
    /// Only the address is read (the `tnet` builder hands it to `nn.Options.target_net`), so a builder
    /// test never initialises a net it does not train.
    net: nn.Net = undefined,
    clip_mode: ClipMode = .norm,
    dist_mix: DistillMix = .{},
    replay_k: usize = default_replay_k,
    learning_rate: f64 = default_learning_rate,

    fn init() !BuildFixture {
        // `nn.Teachers` is ~40 MB, so it is heap allocated like `main` does.
        const teachers = try std.testing.allocator.create(nn.Teachers);
        teachers.len = 0;
        teachers.draws = 0;
        return .{ .teachers = teachers };
    }

    fn deinit(self: *BuildFixture) void {
        std.testing.allocator.destroy(self.teachers);
    }

    fn ctx(self: *BuildFixture, args: []const []const u8) BuildCtx {
        return .{
            .args = args,
            .replay_k = self.replay_k,
            .learning_rate = self.learning_rate,
            .adam = &self.adam,
            .priority = &self.priority,
            .dump = &self.dump,
            .teachers = self.teachers,
            .target_net = &self.net,
            .clip_mode = self.clip_mode,
            .dist_mix = self.dist_mix,
        };
    }
};

/// An empty command line: every `parse*` helper defaults an absent slot, so a build from this is a
/// bare `<variant>` invocation. `main`'s shared tail (`batch_scale`, `validateOptions`) is not part of
/// the build, so the expectations below carry `batch_scale` at its struct default.
const no_build_args: []const []const u8 = &.{};

test "each variant builds the options of its own branch family (#315)" {
    var fixture = try BuildFixture.init();
    defer fixture.deinit();
    const ctx = fixture.ctx(no_build_args);
    const Expect = struct { name: []const u8, opts: nn.Options };
    const expected = [_]Expect{
        .{ .name = "base", .opts = .{} },
        .{ .name = "backward", .opts = .{ .backward = true } },
        .{ .name = "replay", .opts = .{ .replay_k = default_replay_k } },
        .{ .name = "tnet", .opts = .{ .replay_k = default_replay_k, .target_net = &fixture.net } },
        .{ .name = "priority", .opts = .{ .priority = &fixture.priority } },
        .{ .name = "midgame", .opts = .{ .priority = &fixture.priority, .midgame = true } },
        .{ .name = "midmix", .opts = .{ .priority = &fixture.priority, .midmix = true } },
        .{ .name = "sym", .opts = .{ .symmetry = true } },
        .{ .name = "symlam", .opts = .{ .symmetry = true, .lambda = 0.7 } },
        .{ .name = "symbatch", .opts = .{ .sym_batch = true } },
        .{ .name = "rsym", .opts = .{ .rsym = true } },
        .{ .name = "rtgt", .opts = .{ .rtgt = true } },
        .{ .name = "explore", .opts = .{ .explore = 0.05 } },
        .{ .name = "explore20", .opts = .{ .explore = 0.20 } },
        .{ .name = "open8", .opts = .{ .warm_open = 8 } },
        .{ .name = "open16", .opts = .{ .warm_open = 16 } },
        .{ .name = "nstep", .opts = .{ .nstep = 4 } },
        .{ .name = "death", .opts = .{ .death = true } },
        .{ .name = "t2", .opts = .{ .target = .search2 } },
        .{ .name = "p2", .opts = .{ .selfplay = .deep2 } },
        .{ .name = "tp2", .opts = .{ .target = .search2, .selfplay = .deep2 } },
        .{ .name = "d2", .opts = .{ .teacher = .deep } },
        .{ .name = "d3", .opts = .{ .teacher = .deep3 } },
        .{ .name = "dumpp2", .opts = .{ .dump_record = true, .dump = &fixture.dump, .selfplay = .deep2 } },
        .{ .name = "tch", .opts = .{} },
        .{ .name = "tchadd", .opts = .{} },
        .{ .name = "dist", .opts = .{ .distill = fixture.teachers } },
        .{ .name = "dson", .opts = .{ .distill = fixture.teachers, .distill_only = true } },
        .{ .name = "oi", .opts = .{} },
        .{ .name = "oi16", .opts = .{} },
        .{ .name = "resume", .opts = .{} },
        .{ .name = "mstd", .opts = .{} },
        .{ .name = "dump", .opts = .{ .dump_record = true, .dump = &fixture.dump, .selfplay = .greedy1 } },
        .{ .name = "ft2", .opts = .{ .deep_mix = true, .dump = &fixture.dump, .deep_every = 1 } },
        .{ .name = "restart", .opts = .{ .restart = true, .dump = &fixture.dump } },
        .{ .name = "adam", .opts = .{ .adam = &fixture.adam, .adam_lr = 0.01 } },
        .{ .name = "tailw", .opts = .{ .tail_alpha = 1 } },
        .{ .name = "clamp", .opts = .{ .clamp = 200 } },
        .{ .name = "shape", .opts = .{ .shape_c = 1 } },
        .{ .name = "clip", .opts = .{ .grad_clip = default_clip_cap } },
        .{ .name = "pairw", .opts = .{} },
        .{ .name = "pairleaf", .opts = .{} },
        .{ .name = "pairrow", .opts = .{} },
        .{ .name = "leaf", .opts = .{ .omit_leaf_reward = true } },
        .{ .name = "pair", .opts = .{} },
        .{ .name = "par", .opts = .{} },
        .{ .name = "lam", .opts = .{ .replay_k = default_replay_k, .sweep_lambda = default_sweep_lambda } },
        .{ .name = "rlam", .opts = .{ .replay_k = default_replay_k, .sweep_lambda = default_sweep_lambda } },
    };
    try std.testing.expectEqual(variant_specs.len, expected.len);
    for (expected) |e| {
        const build = findVariant(e.name).?.build.?;
        try std.testing.expectEqual(e.opts, try build(&ctx));
    }
}

test "every table variant resolves and builds without the usage path (#315)" {
    var fixture = try BuildFixture.init();
    defer fixture.deinit();
    const ctx = fixture.ctx(no_build_args);
    for (variant_specs) |spec| {
        const resolved = findVariant(@tagName(spec.variant)) orelse {
            std.debug.print("variant {s} does not resolve\n", .{@tagName(spec.variant)});
            return error.TestExpectedEqual;
        };
        try std.testing.expectEqual(spec.variant, resolved.variant);
        const build = resolved.build orelse {
            std.debug.print("variant {s} has no options builder\n", .{@tagName(spec.variant)});
            return error.TestExpectedEqual;
        };
        _ = try build(&ctx);
    }
    // `.unknown` is the only variant without an entry, and it must stay builder-less: that `null` is
    // what makes `main` print the usage for a name no entry matched.
    try std.testing.expect((VariantSpec{ .variant = .unknown }).build == null);
}

test "builders take the slots' values and the borrowed buffers from the context (#315)" {
    var fixture = try BuildFixture.init();
    defer fixture.deinit();
    fixture.replay_k = 7;
    fixture.dist_mix = .{ .every = 4, .div = 10 };
    const ctx = fixture.ctx(no_build_args);
    // The ctx fields, not the defaults, reach the options.
    try std.testing.expectEqual(@as(usize, 7), (try buildReplay(&ctx)).replay_k);
    const tnet = try buildTnet(&ctx);
    try std.testing.expectEqual(@as(usize, 7), tnet.replay_k);
    try std.testing.expect(tnet.target_net.? == &fixture.net);
    try std.testing.expect((try buildAdam(&ctx)).adam.? == &fixture.adam);
    try std.testing.expect((try buildPriority(&ctx)).priority.? == &fixture.priority);
    try std.testing.expect((try buildDump(&ctx)).dump.? == &fixture.dump);
    try std.testing.expect((try buildFt2(&ctx)).dump.? == &fixture.dump);
    try std.testing.expect((try buildDist(&ctx)).distill.? == fixture.teachers);
    const dist = try buildDist(&ctx);
    try std.testing.expectEqual(@as(u32, 4), dist.distill_every);
    // The teacher rate is the TD rate over the arg-13 divisor (#207); read the rate at runtime so the
    // comparison divides in f64 like `main` does, not in comptime precision.
    const td_rate: f64 = fixture.learning_rate;
    try std.testing.expectEqual(td_rate / 10, dist.distill_lr);
    try std.testing.expect((try buildDson(&ctx)).distill_only);
}

test "clip puts the cap in the field its mode names (#315)" {
    var fixture = try BuildFixture.init();
    defer fixture.deinit();
    // The cap sits at index 13 like Adam's step size, `tailw`'s alpha and `clamp`'s bound; the mode at
    // index 14 decides whether it bounds the merged batch delta or every element of a step.
    const args = [_][]const u8{ "nn", "-", "0", "9000", "0.003", "-", "clip", "-", "-", "-", "-", "-", "-", "0.25", "step" };
    fixture.clip_mode = .step;
    const step_ctx = fixture.ctx(&args);
    const step = try buildClip(&step_ctx);
    try std.testing.expectEqual(@as(f64, 0.25), step.step_clip);
    try std.testing.expectEqual(@as(f64, 0), step.grad_clip);
    fixture.clip_mode = .norm;
    const norm_ctx = fixture.ctx(&args);
    const norm = try buildClip(&norm_ctx);
    try std.testing.expectEqual(@as(f64, 0.25), norm.grad_clip);
    try std.testing.expectEqual(@as(f64, 0), norm.step_clip);
}
