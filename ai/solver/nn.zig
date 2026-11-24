const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const game_limit = @import("game_limit");
const parallel = @import("parallel");
const eval_stats = @import("stats");
const blob = @import("blob");
/// Hidden layer widths. Compile-time because the net is an `extern struct` whose layout is the weights
/// file; `zig build nn -Dh1=… -Dh2=…` sweeps capacity without editing copies of the net.
const width = @import("nn_width");

const Board = core.Board;
const Direction = core.Direction;
const GameState = core.GameState;

/// Cell encoding: log2 tile level (0 = empty), one-hot per cell. Tiles >= 2^15 share the top level.
pub const levels = 16;
pub const input_count = core.BoardCellCount * levels;
pub const h1_size: u32 = width.h1;
pub const h2_size: u32 = width.h2;
/// Gate tier: false under `zig build test -Dslow-tests=false` (pre-push). A slow full-game /
/// training / deep-search regression test starts with `if (!slow_tests) return error.SkipZigTest;`
/// so it runs only in CI (`gate.sh all`). Re-exported by `solver` for the targets that import it.
pub const slow_tests = width.slow_tests;
/// Hand-engineered global board features: 0 (the default build) keeps the 256 one-hot inputs and the current
/// weights file; `zig build nn -Dfeat=6` adds the first `feature_count` of `feature_all` as extra input rows
/// (`Net.wf`), so the net need not learn empty-count/monotonicity from scratch.
pub const feature_count: u32 = width.feat;
/// Every feature is normalised to 0..1 so one learning rate serves all of them, in this order:
/// empties, max level, monotonicity, smoothness, max tile in a corner, total level mass.
pub const feature_all = 6;
/// Score deltas are divided by this before they enter the value target, so values stay near unit scale.
pub const reward_scale: f64 = 1000;
/// `-Dnn-f32`: the shipped net is evaluated from `nn_weights.f32.bin` (#348). Off by default, so the
/// shipped policy code and the f64 weight file stay the reference.
pub const f32_weights: bool = width.f32_weights;

/// Plain f64 weights. `extern` keeps the layout fixed, so the file is the raw struct bytes.
pub const Net = extern struct {
    /// First layer, one row per (cell, level) pair. Only 16 rows are active for any board.
    w1: [input_count][h1_size]f64,
    b1: [h1_size]f64,
    w2: [h2_size][h1_size]f64,
    b2: [h2_size]f64,
    w3: [h2_size]f64,
    b3: f64,
    /// One row per global feature; zero width unless the build sets `-Dfeat`.
    wf: [feature_count][h1_size]f64,

    pub fn init(rng: std.Random) Net {
        var net = std.mem.zeroes(Net);
        for (&net.w1) |*row| for (row) |*w| {
            w.* = uniform(rng, 1.0 / 4.0);
        };
        for (&net.w2) |*row| for (row) |*w| {
            w.* = uniform(rng, 1.0 / 8.0);
        };
        for (&net.w3) |*w| w.* = uniform(rng, 1.0 / @sqrt(@as(f64, h2_size)));
        // `wf` stays zero: these rows contribute nothing to a fresh net, so a `-Dfeat` build reproduces
        // the plain one until training moves them, and they still get their gradient on the first step
        // because the update uses `cache.feat`, not the current weight.
        return net;
    }
};

fn uniform(rng: std.Random, scale: f64) f64 {
    return (rng.float(f64) * 2 - 1) * scale;
}

/// Optimistic initialization: multiply every weight by `k`, so the initial value surface has more spread and
/// the self-play policy explores more early on. Only the spread can matter: a constant shift of `b3` adds the
/// same amount to every afterstate, so the argmax over moves is unchanged.
pub fn scaleInit(net: *Net, k: f64) void {
    for (&net.w1) |*row| {
        for (row) |*w| w.* *= k;
    }
    for (&net.w2) |*row| {
        for (row) |*w| w.* *= k;
    }
    for (&net.w3) |*w| w.* *= k;
}

/// Activations kept from one forward pass; needed by `train`.
pub const Cache = struct {
    rows: [core.BoardCellCount]usize = undefined,
    z1: [h1_size]f64 = undefined,
    a1: [h1_size]f64 = undefined,
    z2: [h2_size]f64 = undefined,
    a2: [h2_size]f64 = undefined,
    out: f64 = 0,
    /// The global features of the same board (empty array unless the build sets `-Dfeat`).
    feat: [feature_count]f64 = undefined,
};

pub fn levelOf(value: core.Tile) usize {
    if (value == 0) return 0;
    return @min(levels - 1, std.math.log2_int(core.Tile, value));
}

/// The global features of `board`, in `feature_all` order, each in 0..1.
pub fn featureValues(board: Board) [feature_all]f64 {
    var empties: f64 = 0;
    var max_level: f64 = 0;
    var mass: f64 = 0;
    var monotone_penalty: f64 = 0;
    var roughness: f64 = 0;
    for (0..core.BoardCellCount) |cell| {
        const level: f64 = @floatFromInt(levelOf(board[cell / core.BoardSize][cell % core.BoardSize]));
        if (level == 0) empties += 1;
        max_level = @max(max_level, level);
        mass += level;
    }
    // Rows and columns as 8 lines of 4 log2 levels (0 = empty, and empties count as 0 in both terms).
    for (0..core.BoardSize) |fixed| {
        for (0..2) |axis| {
            var inc: f64 = 0;
            var dec: f64 = 0;
            var prev: f64 = 0;
            for (0..core.BoardSize) |k| {
                const level: f64 = @floatFromInt(if (axis == 0)
                    levelOf(board[fixed][k])
                else
                    levelOf(board[k][fixed]));
                const diff = level - prev;
                if (k > 0) {
                    inc += @max(diff, 0);
                    dec += @max(-diff, 0);
                    if (level > 0 and prev > 0) roughness += @abs(diff);
                }
                prev = level;
            }
            monotone_penalty += @min(inc, dec);
        }
    }
    const max_tile = @max(1, max_level);
    const corners = [_]usize{ 0, core.BoardSize - 1, core.BoardCellCount - core.BoardSize, core.BoardCellCount - 1 };
    var in_corner = false;
    if (max_level > 0) for (corners) |c| {
        const level = levelOf(board[c / core.BoardSize][c % core.BoardSize]);
        if (@as(f64, @floatFromInt(level)) == max_tile) in_corner = true;
    };
    const scale: f64 = @floatFromInt(core.BoardSize * core.BoardSize);
    return .{
        empties / scale,
        max_level / @as(f64, @floatFromInt(levels - 1)),
        @max(0, 1 - monotone_penalty / (scale / 2 * @as(f64, @floatFromInt(levels - 1)))),
        @max(0, 1 - roughness / (scale * @as(f64, @floatFromInt(levels - 1)))),
        if (in_corner) 1 else 0,
        mass / (scale * @as(f64, @floatFromInt(levels - 1))),
    };
}

/// The first `feature_count` values of `featureValues`, which is what the net consumes.
pub fn features(board: Board) [feature_count]f64 {
    const all = featureValues(board);
    var out: [feature_count]f64 = undefined;
    for (0..feature_count) |k| out[k] = all[k];
    return out;
}

/// Each lane follows the same row-addition order as the scalar loop.
fn accumulateRowsSimd(dst: []f64, src: []const f64) void {
    const V = @Vector(2, f64);
    var i: usize = 0;
    while (i + 2 <= dst.len) : (i += 2) {
        const a: V = dst[i..][0..2].*;
        const b: V = src[i..][0..2].*;
        dst[i..][0..2].* = a + b;
    }
    while (i < dst.len) : (i += 1) dst[i] += src[i];
}

/// Afterstate value estimate (scalar, unit-scale). Fills `cache` for a following `train`.
pub fn forward(net: *const Net, board: Board, cache: *Cache) f64 {
    return forwardImpl(net, board, cache, width.simd);
}

fn forwardImpl(net: *const Net, board: Board, cache: *Cache, comptime simd: bool) f64 {
    for (0..core.BoardCellCount) |cell| {
        const value = board[cell / core.BoardSize][cell % core.BoardSize];
        cache.rows[cell] = cell * levels + levelOf(value);
    }
    cache.z1 = net.b1;
    for (cache.rows) |row| {
        if (comptime simd) {
            accumulateRowsSimd(&cache.z1, &net.w1[row]);
        } else {
            for (0..h1_size) |i| cache.z1[i] += net.w1[row][i];
        }
    }
    if (comptime feature_count > 0) {
        cache.feat = features(board);
        for (cache.feat, 0..) |f, k| {
            if (f == 0) continue;
            for (0..h1_size) |i| cache.z1[i] += net.wf[k][i] * f;
        }
    }
    for (0..h1_size) |i| cache.a1[i] = @max(cache.z1[i], 0);

    var out = net.b3;
    for (0..h2_size) |j| {
        var z = net.b2[j];
        for (0..h1_size) |i| z += net.w2[j][i] * cache.a1[i];
        cache.z2[j] = z;
        cache.a2[j] = @max(z, 0);
        out += net.w3[j] * cache.a2[j];
    }
    cache.out = out;
    return out;
}

/// One SGD step on 0.5 * (out - target)^2, using the activations in `cache` (from forward on the same board).
pub fn train(net: *Net, cache: *const Cache, target: f64, lr: f64) void {
    trainCapped(net, cache, target, lr, 0);
}

/// One element of a step, cut to +-`cap` when `cap > 0`. A non-positive cap returns the delta unchanged, so
/// `train` (which passes 0) is a plain SGD step.
fn capStep(delta: f64, cap: f64) f64 {
    return if (cap > 0) std.math.clamp(delta, -cap, cap) else delta;
}

/// `train` with a per-element ceiling on `|dw|`: the serial per-move step and the merged `par` step are the
/// two places that apply a step, and both pass their `step_clip` here.
pub fn trainCapped(net: *Net, cache: *const Cache, target: f64, lr: f64, step_clip: f64) void {
    const d = cache.out - target;
    var dz2: [h2_size]f64 = undefined;
    var da1 = std.mem.zeroes([h1_size]f64);
    for (0..h2_size) |j| {
        dz2[j] = if (cache.z2[j] > 0) d * net.w3[j] else 0;
        for (0..h1_size) |i| da1[i] += net.w2[j][i] * dz2[j];
    }

    net.b3 -= capStep(lr * d, step_clip);
    for (0..h2_size) |j| {
        net.w3[j] -= capStep(lr * d * cache.a2[j], step_clip);
        net.b2[j] -= capStep(lr * dz2[j], step_clip);
        for (0..h1_size) |i| net.w2[j][i] -= capStep(lr * dz2[j] * cache.a1[i], step_clip);
    }

    for (0..h1_size) |i| {
        const dz1 = if (cache.z1[i] > 0) da1[i] else 0;
        net.b1[i] -= capStep(lr * dz1, step_clip);
        for (cache.rows) |row| net.w1[row][i] -= capStep(lr * dz1, step_clip);
        if (comptime feature_count > 0) {
            for (0..feature_count) |k| if (cache.feat[k] != 0) {
                net.wf[k][i] -= capStep(lr * dz1 * cache.feat[k], step_clip);
            };
        }
    }
}

/// PopArt uses exponential target moments in the existing reward_scale units; the weight file still
/// stores the ordinary raw-value network, while the optimizer sidecar stores these statistics.
pub const popart_beta: f64 = 1e-3;
const popart_min_variance: f64 = 1e-6;
const popart_state_magic = "2048-PA\x00";
const popart_state_version: u32 = 1;
const popart_state_file_len = 72;

pub const PopArtState = struct {
    mu: f64 = 0,
    nu: f64 = 1,
    updates: u64 = 0,
    value_mu: f64 = 0,
    value_nu: f64 = 1,
    value_updates: u64 = 0,
    beta: f64 = popart_beta,

    pub fn variance(self: *const PopArtState) f64 {
        return @max(self.nu - self.mu * self.mu, popart_min_variance);
    }

    pub fn sigma(self: *const PopArtState) f64 {
        return @sqrt(self.variance());
    }

    /// EMA variance of raw predictions, for diagnostics only (not the normalization target).
    pub fn valueVariance(self: *const PopArtState) f64 {
        return @max(self.value_nu - self.value_mu * self.value_mu, popart_min_variance);
    }

    pub fn valueSigma(self: *const PopArtState) f64 {
        return @sqrt(self.valueVariance());
    }

    pub fn denormalize(self: *const PopArtState, normalized: f64) f64 {
        return self.sigma() * normalized + self.mu;
    }

    /// Update target moments and temporarily convert the raw-value output layer to the new normalized
    /// coordinates. `cache` was computed before the conversion; its hidden activations remain valid.
    pub fn startTrainingUpdate(self: *PopArtState, net: *Net, cache: *Cache, raw_target: f64) f64 {
        const raw_value = cache.out;
        self.value_mu = (1 - self.beta) * self.value_mu + self.beta * raw_value;
        self.value_nu = @max((1 - self.beta) * self.value_nu + self.beta * raw_value * raw_value, self.value_mu * self.value_mu + popart_min_variance);
        self.value_updates += 1;
        self.mu = (1 - self.beta) * self.mu + self.beta * raw_target;
        self.nu = @max((1 - self.beta) * self.nu + self.beta * raw_target * raw_target, self.mu * self.mu + popart_min_variance);
        self.updates += 1;
        const sigma_now = self.sigma();
        for (&net.w3) |*weight| weight.* /= sigma_now;
        net.b3 = (net.b3 - self.mu) / sigma_now;
        cache.out = (cache.out - self.mu) / sigma_now;
        return (raw_target - self.mu) / sigma_now;
    }

    /// Restore the raw-value parameterization after the normalized SGD step, preserving the existing
    /// Net/weight-file contract for play, evaluation, and non-PopArt variants.
    pub fn finishTrainingUpdate(self: *const PopArtState, net: *Net) void {
        const sigma_now = self.sigma();
        for (&net.w3) |*weight| weight.* *= sigma_now;
        net.b3 = sigma_now * net.b3 + self.mu;
    }

    pub fn encode(self: *const PopArtState) [popart_state_file_len]u8 {
        var bytes: [popart_state_file_len]u8 = undefined;
        @memcpy(bytes[0..8], popart_state_magic);
        std.mem.writeInt(u32, bytes[8..12], popart_state_version, .little);
        std.mem.writeInt(u32, bytes[12..16], endian_marker, .little);
        std.mem.writeInt(u64, bytes[16..24], self.updates, .little);
        std.mem.writeInt(u64, bytes[24..32], @bitCast(self.mu), .little);
        std.mem.writeInt(u64, bytes[32..40], @bitCast(self.nu), .little);
        std.mem.writeInt(u64, bytes[40..48], @bitCast(self.beta), .little);
        std.mem.writeInt(u64, bytes[48..56], self.value_updates, .little);
        std.mem.writeInt(u64, bytes[56..64], @bitCast(self.value_mu), .little);
        std.mem.writeInt(u64, bytes[64..72], @bitCast(self.value_nu), .little);
        return bytes;
    }

    pub fn parse(bytes: []const u8) !PopArtState {
        if (bytes.len != popart_state_file_len) return error.InvalidPopArtStateSize;
        if (!std.mem.eql(u8, bytes[0..8], popart_state_magic)) return error.BadPopArtMagic;
        if (std.mem.readInt(u32, bytes[8..12], .little) != popart_state_version) return error.UnsupportedPopArtVersion;
        if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongPopArtEndianness;
        const state = PopArtState{
            .updates = std.mem.readInt(u64, bytes[16..24], .little),
            .mu = @bitCast(std.mem.readInt(u64, bytes[24..32], .little)),
            .nu = @bitCast(std.mem.readInt(u64, bytes[32..40], .little)),
            .beta = @bitCast(std.mem.readInt(u64, bytes[40..48], .little)),
            .value_updates = std.mem.readInt(u64, bytes[48..56], .little),
            .value_mu = @bitCast(std.mem.readInt(u64, bytes[56..64], .little)),
            .value_nu = @bitCast(std.mem.readInt(u64, bytes[64..72], .little)),
        };
        if (!std.math.isFinite(state.mu) or !std.math.isFinite(state.nu) or
            !std.math.isFinite(state.value_mu) or !std.math.isFinite(state.value_nu) or
            !(state.nu >= state.mu * state.mu) or !(state.value_nu >= state.value_mu * state.value_mu) or
            state.updates != state.value_updates or state.beta != popart_beta) return error.InvalidPopArtState;
        return state;
    }

    pub fn save(self: *const PopArtState, io: std.Io, path: []const u8) !void {
        const bytes = self.encode();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = &bytes });
    }

    pub fn load(self: *PopArtState, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(popart_state_file_len + 1));
        defer gpa.free(bytes);
        // Qualified: `nn.parse` (the Net file) is also in scope here, so a bare `parse` is ambiguous.
        self.* = try PopArtState.parse(bytes);
    }
};

/// Adam moment estimates, one per weight. The optimizer state lives outside the weight file, so the
/// format, the shipped net and every existing variant stay untouched.
pub const Adam = struct {
    m: Net = std.mem.zeroes(Net),
    v: Net = std.mem.zeroes(Net),
    t: u64 = 0,
};

/// One Adam step on `net` from the raw gradient `g` (= d loss / d w, the gradient of `train`'s
/// 0.5 * (out - target)^2). `lr` is Adam's own step size: every parameter moves about `lr` per step
/// regardless of how large `g` is, so the step does not couple to the number of updates per move.
///
/// `par` feeds it one gradient per batch (the per-game SGD deltas divided by `lr`, i.e. the summed
/// gradient of the games in the batch) and steps once, which is the batch update Adam wants.
pub fn applyAdam(net: *Net, g: *const Net, state: *Adam, lr: f64) void {
    const beta1: f64 = 0.9;
    const beta2: f64 = 0.999;
    const eps: f64 = 1e-8;
    state.t += 1;
    const t: f64 = @floatFromInt(state.t);
    const c1 = 1 - std.math.pow(f64, beta1, t);
    const c2 = 1 - std.math.pow(f64, beta2, t);
    for (flat(net), flatConst(g), flat(&state.m), flat(&state.v)) |*w, x, *m, *v| {
        m.* = beta1 * m.* + (1 - beta1) * x;
        v.* = beta2 * v.* + (1 - beta2) * x * x;
        w.* -= lr * (m.* / c1) / (@sqrt(v.* / c2) + eps);
    }
}

/// Multi-stage TD value function: one value net per afterstate stage, staged by the number of empty cells.
/// Boundaries: 0..2, 3..4, 5..6, 7..8, 9..10, 11..12, 13..16 empties.
pub const stage_count: usize = 7;

pub fn stageOfEmpties(empties: usize) usize {
    return switch (empties) {
        0...2 => 0,
        3...4 => 1,
        5...6 => 2,
        7...8 => 3,
        9...10 => 4,
        11...12 => 5,
        else => 6,
    };
}

/// Value function used by the search: one net for every board, or one net per empty-cell stage.
/// `.quantized` is the opt-in int16 path: same policy code, weights from the quantized file. `.f32`
/// (#348) is the same net stored and evaluated as f32: half the bytes to read per leaf and an
/// f32x4 first layer under `-Dnn-simd`. Nothing selects either on its own; the f64 net stays the
/// default.
pub const Value = union(enum) { single: *const Net, staged: *const Staged, quantized: *const Quantized, f32: *const Net32 };

/// `forward` for a `Value`; `cache` belongs to the net being evaluated and is unused for staged,
/// quantized and f32 values.
pub fn valueAt(value: Value, board: Board, cache: *Cache) f64 {
    return switch (value) {
        .single => |net| forward(net, board, cache),
        .staged => |staged| staged.forward(board),
        .quantized => |q| forwardQuantized(q, board),
        .f32 => |net| forwardF32(net, board),
    };
}

// Container-level aliases: the `Staged` methods below shadow `forward` / `train` by name.
const netForward = forward;
const netTrain = train;

/// Multi-stage TD value function: `stage_count` nets of the `Net` architecture, chosen by the afterstate's
/// empty-cell count. Search and training reach it through `Value`.
pub const Staged = struct {
    nets: [stage_count]Net,

    pub fn init(rng: std.Random) Staged {
        var self: Staged = undefined;
        for (&self.nets) |*net| net.* = Net.init(rng);
        return self;
    }

    /// Stage of a board; afterstates with fewer empty cells use the earlier nets.
    pub fn stageOf(board: Board) usize {
        return stageOfEmpties(countEmpty(board));
    }

    /// Value of `board` from the net of its own stage.
    pub fn forward(self: *const Staged, board: Board) f64 {
        var cache: Cache = .{};
        return netForward(&self.nets[stageOf(board)], board, &cache);
    }

    /// One TD(0) step (gamma = 1, like the rest of the learner): `after` is the afterstate the move
    /// reached, `next` the afterstate of the following step (null at game over). Only the net of
    /// `after`'s stage is updated. Returns the TD error `reward + V(next) - V(after)`.
    pub fn train(self: *Staged, after: Board, next: ?Board, reward: f64, lr: f64) f64 {
        const stage = stageOf(after);
        var cache: Cache = .{};
        const v = netForward(&self.nets[stage], after, &cache);
        const target = reward + if (next) |n| self.forward(n) else 0;
        netTrain(&self.nets[stage], &cache, target, lr);
        return target - v;
    }

    pub fn saveFile(self: *const Staged, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try gpa.alloc(u8, staged_file_len);
        defer gpa.free(bytes);
        var w = blob.Writer.init(bytes);
        blob.writeHeader(&w, staged_file_magic, file_version);
        for (0..stage_count) |k| writeWeights(&w, &self.nets[k]);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    pub fn loadFile(self: *Staged, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(staged_file_len + 1));
        defer gpa.free(bytes);
        try self.parseStaged(bytes);
    }

    fn parseStaged(self: *Staged, bytes: []const u8) !void {
        if (bytes.len != staged_file_len) return error.WrongWeightFileSize;
        var r = blob.Reader.init(bytes);
        try blob.readHeader(&r, staged_file_magic, file_version, .{
            .magic = error.BadWeightMagic,
            .version = error.UnsupportedWeightVersion,
            .endian = error.WrongWeightEndianness,
        });
        for (0..stage_count) |k| self.nets[k] = try readWeights(&r);
    }
};

/// Dihedral symmetry: 8 cell-index permutations, rotations 0..3 then their mirrors 4..7.
/// `symmetryMap(k, cell)` is the cell that `cell` moves to under transform `k` (k = 0 is identity).
pub const symmetry_count = 8;

pub fn symmetryMap(k: usize, cell: usize) usize {
    const n = core.BoardSize;
    var r = cell / n;
    var c = cell % n;
    if (k >= 4) c = n - 1 - c;
    for (0..k % 4) |_| {
        const t = r;
        r = c;
        c = n - 1 - t;
    }
    return r * n + c;
}

/// Board under the k-th dihedral transform (pure, deterministic).
pub fn transformBoard(k: usize, board: Board) Board {
    var out: Board = undefined;
    for (0..core.BoardCellCount) |cell| {
        const to = symmetryMap(k, cell);
        out[to / core.BoardSize][to % core.BoardSize] = board[cell / core.BoardSize][cell % core.BoardSize];
    }
    return out;
}

/// Afterstate chosen by the self-play policy, with its scaled reward and the value that policy assigned
/// to `after`: `V(after)` for the 1-ply policy, the 2-ply search value of the afterstate for `deep2`.
const Pick = struct {
    dir: Direction,
    after: Board,
    reward: f64,
    value: f64,
};

/// One legal move drawn uniformly: the exploration half of `Options.explore`. The pick's `value` is
/// `V(after)` so the TD step and the priority buffer see the same numbers as a greedy pick.
fn randomPick(vfn: Value, state: GameState, rng: std.Random, death: bool) ?Pick {
    var picks: [4]Pick = undefined;
    var n: usize = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        var cache: Cache = .{};
        const reward = if (death) -1 / reward_scale else @as(f64, @floatFromInt(moved.score_delta)) / reward_scale;
        picks[n] = .{ .dir = d, .after = next.board, .reward = reward, .value = valueAt(vfn, next.board, &cache) };
        n += 1;
    }
    if (n == 0) return null;
    return picks[rng.uintLessThan(usize, n)];
}

/// Best legal move by reward + V(afterstate); null when no move changes the board.
fn pickMove(net: *const Net, state: GameState, death: bool) ?Pick {
    return pickMoveValue(.{ .single = net }, state, death);
}

/// `pickMove` for a single or staged value function.
fn pickMoveValue(vfn: Value, state: GameState, death: bool) ?Pick {
    var best: ?Pick = null;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        var cache: Cache = .{};
        const reward = if (death) -1 / reward_scale else @as(f64, @floatFromInt(moved.score_delta)) / reward_scale;
        const pick: Pick = .{ .dir = d, .after = next.board, .reward = reward, .value = valueAt(vfn, next.board, &cache) };
        if (best == null or pick.reward + pick.value > best.?.reward + best.?.value) best = pick;
    }
    return best;
}

/// Value of a post-spawn board for the chance-node search: best reward + V over the legal moves (0 if none).
fn bestAfterValue(vfn: Value, board_state: GameState) f64 {
    const pick = pickMoveValue(vfn, board_state, false) orelse return 0;
    return pick.reward + pick.value;
}

/// Expectimax with one spawn of lookahead: for each move, average over spawns of the best NN afterstate value.
fn chanceMove(vfn: Value, state: GameState) ?Direction {
    var best: ?Direction = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var moved = state;
        const result = moved.slide(d);
        if (!result.changed) continue;
        var total: f64 = 0;
        var empties: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (moved.board[r][c] != 0) continue;
                empties += 1;
                for ([_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } }) |spawn| {
                    var child = moved;
                    child.board[r][c] = spawn.value;
                    total += spawn.p * bestAfterValue(vfn, child);
                }
            }
        }
        const chance = if (empties == 0) bestAfterValue(vfn, moved) else total / empties;
        const value = @as(f64, @floatFromInt(result.score_delta)) / reward_scale + chance;
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

/// Search statistics for one move: afterstate evaluations and value-cache hits, plus the interior-node
/// lookups of the transposition table (both counters stay 0 when it is off).
pub const Stats = struct { leaves: u64 = 0, hits: u64 = 0, node_probes: u64 = 0, node_hits: u64 = 0 };

/// Default seed of the per-chance-node spawn sampler. Any fixed value does; it only has to make a sampled
/// run reproducible, so an A/B on the same spawn tape sees only the policy difference.
pub const default_sample_seed: u64 = 0x9e3779b97f4a7c15;

/// Depth one search runs at: `ply` own moves deep, dropping spawn branches under path probability
/// `cutoff` (`0` keeps every branch) and dropping the lowest-probability branches up to a share
/// `mass_cutoff` of each chance node's mass (`0` keeps every branch). Part of the value-cache key,
/// because a phase game asks for the same board at two depths.
pub const Depth = struct {
    ply: u8 = 0,
    cutoff: f64 = 0,
    mass_cutoff: f64 = 0,
    risk_weight: f64 = 0,
    omit_leaf_reward: bool = false,
    recover_cut_branches: bool = false,
    /// At most this many spawn branches per chance node, drawn by weighted sampling without replacement.
    /// `0` (the default) keeps every branch.
    sample: usize = 0,
    /// The seed of that sampler. Same board, depth and seed give the same spawns, so a re-search (or the
    /// interior-node table) is consistent with a fresh one.
    sample_seed: u64 = default_sample_seed,
    /// The shallowest chance node the sample applies to. `1` samples every spawn layer; `2` keeps the layer
    /// directly above the leaves at full width, so the spawn that decides whether the last move survives is
    /// never thinned.
    sample_min_depth: u8 = 1,
};

/// A completed search node: its value and whether that value is a direct leaf. The node table
/// stores both, because the parent of a direct leaf omits that move's reward.
const SearchResult = struct { value: f64 = 0, direct_leaf: bool = false };

/// Key depth of the 1-ply lookups (`get` / `getValue`, e.g. the `nnmcts` rollouts), which never search.
pub const leaf_depth: Depth = .{};

/// Bounded transposition table for afterstate values, direct-mapped and deterministic.
/// shortcut: 2^13 slots (128 KiB) keyed by the board hash and the search depth; a collision drops the
/// entry (a miss, never a wrong value), so the table only changes speed, not the chosen move.
pub const ValueCache = struct {
    /// The cache lives on the stack: wasm gets a smaller one (the 8K-entry version overflowed it).
    pub const capacity = if (builtin.target.cpu.arch.isWasm()) 1 << 10 else 1 << 13;
    const empty_key = std.math.maxInt(u64);
    const Entry = struct { key: u64 = empty_key, value: f64 = 0 };
    entries: [capacity]Entry = @splat(.{}),

    /// Drops every entry. `getValue` keys on the board contents alone, so a table filled by an older
    /// weight snapshot silently answers for the current one: anything that prices a board *between*
    /// weight updates (every teacher label, / ) has to start from an empty table.
    pub fn clear(self: *ValueCache) void {
        self.entries = @splat(.{});
    }

    fn keyOf(board: Board) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (board) |row| for (row) |value| {
            h = (h ^ @as(u64, value)) *% 0x100000001b3;
        };
        return h;
    }

    /// `keyOf` mixed with the search depth. The stored value is the depth-free leaf value
    /// today, but a phase game asks for one board at 2-ply and at 3-ply: without the depth in the key
    /// one entry would stand in for the other, and any depth-dependent leaf or backup would then be
    /// silently wrong. Both fields are mixed here, so the table keeps one key space.
    fn keyOfDepth(board: Board, depth: Depth) u64 {
        var h = keyOf(board);
        h = (h ^ @as(u64, depth.ply)) *% 0x100000001b3;
        h = (h ^ @as(u64, @bitCast(depth.cutoff))) *% 0x100000001b3;
        // the mass cut changes the value too, so it is part of the key as well; a zero mass
        // cut does not mix, which keeps every pre-#182 key (and the recorded golden values) as is.
        // The spawn sample stays out on purpose: a leaf is the net's exact V whatever the sample
        // size, and this table holds leaves only. Sampled interior values live in the per-search node
        // table, which is cleared before every top-level search, so its config cannot leak across runs.
        if (depth.mass_cutoff == 0) return h;
        return (h ^ @as(u64, @bitCast(depth.mass_cutoff))) *% 0x100000001b3;
    }

    /// `keyOfDepth` mixed with the leaf flag. The false case keeps the old key bit for bit, so a
    /// cache behaves exactly as before; the true case is a bijection of the key, which adds no new
    /// collision risk (a collision would be a miss, never a wrong value).
    fn keyOfDepthSym(board: Board, depth: Depth, sym_leaf: bool) u64 {
        const key = keyOfDepth(board, depth);
        if (key == empty_key or !sym_leaf) return key;
        return (key ^ 1) *% 0x100000001b3;
    }

    /// `forward(net, board)` with memoization; `stats` counts leaf evaluations and hits.
    pub fn get(self: *ValueCache, net: *const Net, board: Board, stats: *Stats) f64 {
        return self.getValue(.{ .single = net }, board, stats);
    }

    /// `get` for a single or staged value function.
    pub fn getValue(self: *ValueCache, vfn: Value, board: Board, stats: *Stats) f64 {
        return self.getValueAtDepth(vfn, board, leaf_depth, stats);
    }

    /// `getValue` for one search depth: the same board at another (ply, cutoff) is another entry.
    pub fn getValueAtDepth(self: *ValueCache, vfn: Value, board: Board, depth: Depth, stats: *Stats) f64 {
        return self.getValueAtDepthSym(vfn, board, depth, false, stats);
    }

    /// `getValueAtDepth` with the leaf flag: a symmetrically averaged leaf is another value, so
    /// it gets its own key instead of sharing the entry of the single-forward one.
    pub fn getValueAtDepthSym(self: *ValueCache, vfn: Value, board: Board, depth: Depth, sym_leaf: bool, stats: *Stats) f64 {
        const key = keyOfDepthSym(board, depth, sym_leaf);
        if (key == empty_key) return leafValueSym(vfn, board, sym_leaf, stats);
        const slot: usize = @intCast(key & (capacity - 1));
        if (self.entries[slot].key == key) {
            stats.hits += 1;
            return self.entries[slot].value;
        }
        const value_ = leafValueSym(vfn, board, sym_leaf, stats);
        self.entries[slot] = .{ .key = key, .value = value_ };
        return value_;
    }
};

/// The leaf: `sym_leaf = false` is the single forward pass as before; `true` averages the value
/// over the 8 dihedral transforms (`symmetryMap`), so one leaf costs `symmetry_count` forward passes
/// instead of one.
fn leafValueSym(vfn: Value, board: Board, sym_leaf: bool, stats: *Stats) f64 {
    var cache: Cache = .{};
    stats.leaves += 1;
    if (!sym_leaf) return valueAt(vfn, board, &cache);
    var total: f64 = 0;
    for (0..symmetry_count) |k| total += valueAt(vfn, transformBoard(k, board), &cache);
    return total / @as(f64, @floatFromInt(symmetry_count));
}

/// Per-thread cache reused across moves. A value is a pure function of the board and
/// the net, so a warm cache can only change speed, never the chosen move; there is no shared
/// mutable state because each thread keeps its own workspace.
threadlocal var move_cache: ValueCache = .{};

/// The calling thread's reusable cache, e.g. for policies that keep their own search loops.
pub fn sharedValueCache() *ValueCache {
    return &move_cache;
}

/// Bounded transposition table for the completed interior nodes of one search, direct-mapped
/// and deterministic like `ValueCache`. A node is identified by its exact board fingerprint, whether
/// it is a `value` or a `chance` node, its remaining depth, and the path probability where that
/// probability can still change the result. The full fingerprint is compared on a probe, so a slot
/// collision is a miss and never returns another position's value.
///
/// It is cleared at the start of every top-level search: the key does not carry the net, the cutoffs
/// or `risk_weight`, and a table surviving a weight update would price boards with the old net. Within
/// one search those settings are fixed, and the key's remaining depth keeps a shallower backup from
/// answering a deeper node (iterations may share nodes that really are at the same remaining depth).
pub const NodeCache = struct {
    /// The table lives in thread-local storage like `ValueCache`; wasm gets the smaller one because
    /// the 8K-entry leaf cache already overflowed that stack.
    pub const capacity = if (builtin.target.cpu.arch.isWasm()) 1 << 9 else 1 << 14;
    /// `0xff` marks an unused slot; a real cell stores the tile's exponent plus one (1..32, 0 empty),
    /// so no board the game can produce collides with the sentinel.
    const unused_cell = 0xff;
    const NodeKind = enum(u8) { value, chance };
    const Key = struct {
        cells: [core.BoardCellCount]u8 = @splat(unused_cell),
        /// The exact `cprob` bits when `cprobRelevant`, otherwise 0 (all those nodes share one key).
        cprob: f64 = 0,
        depth: u8 = 0,
        kind: NodeKind = .value,
    };
    /// 32 B key + 16 B result (value + direct_leaf) = 48 B per slot.
    const Entry = struct { key: Key = .{}, result: SearchResult = .{} };
    entries: [capacity]Entry = @splat(.{}),

    /// Drops every entry; a search must start from an empty table because the key does not carry the
    /// net, the cutoffs or `risk_weight`.
    pub fn clear(self: *NodeCache) void {
        for (&self.entries) |*entry| entry.* = .{};
    }

    /// Exact board fingerprint, one byte per cell: the tile's log2 plus one (0 for empty). Game tiles
    /// are powers of two up to `2^31`, so this is lossless; comparing these bytes instead of a hash
    /// makes a collision a miss.
    fn cellsOf(board: Board) [core.BoardCellCount]u8 {
        var cells: [core.BoardCellCount]u8 = undefined;
        var i: usize = 0;
        for (board) |row| for (row) |tile| {
            cells[i] = if (tile == 0) 0 else @as(u8, @intCast(@ctz(tile))) + 1;
            i += 1;
        };
        return cells;
    }

    /// Whether `cprob` can change this node's value. The fixed cut is only tested at `chance` depth
    /// > 1, so `value` depth <= 2 and `chance` depth <= 1 never read it; their cutoffs and the mass
    /// cut compare relative probabilities, and `riskScore` normalizes by the node's own mass, so the
    /// scale of `cprob` cancels. Deeper nodes keep the exact bits because the cut depends on them.
    fn cprobRelevant(kind: NodeKind, depth: u8) bool {
        return switch (kind) {
            .value => depth > 2,
            .chance => depth > 1,
        };
    }

    fn keyFor(board: Board, kind: NodeKind, depth: u8, cprob: f64) Key {
        return .{
            .cells = cellsOf(board),
            .cprob = if (cprobRelevant(kind, depth)) cprob else 0,
            .depth = depth,
            .kind = kind,
        };
    }

    fn slotOf(key: Key) usize {
        var h: u64 = 0xcbf29ce484222325;
        for (key.cells) |cell| h = (h ^ cell) *% 0x100000001b3;
        h = (h ^ @as(u64, @backingInt(key.kind))) *% 0x100000001b3;
        h = (h ^ key.depth) *% 0x100000001b3;
        h = (h ^ @as(u64, @bitCast(key.cprob))) *% 0x100000001b3;
        return @intCast(h & (capacity - 1));
    }

    /// The completed value of this exact node; null on an empty or mismatched slot. `stats` counts
    /// probes and hits so a measurement can report the table's hit rate.
    fn get(self: *const NodeCache, key: Key, stats: *Stats) ?SearchResult {
        stats.node_probes += 1;
        const entry = self.entries[slotOf(key)];
        if (entry.key.depth != key.depth or entry.key.kind != key.kind or
            entry.key.cprob != key.cprob or !std.mem.eql(u8, &entry.key.cells, &key.cells)) return null;
        stats.node_hits += 1;
        return entry.result;
    }

    /// Store a completed node; `Search.nodePut` withholds incomplete (budget-exhausted) values.
    fn put(self: *NodeCache, key: Key, result: SearchResult) void {
        self.entries[slotOf(key)] = .{ .key = key, .result = result };
    }
};

/// Per-thread interior-node table reused by the opt-in search. The search clears it, so
/// a warm table changes speed only and can never answer across a weight update.
threadlocal var node_cache: NodeCache = .{};

/// The calling thread's reusable interior-node table.
pub fn sharedNodeCache() *NodeCache {
    return &node_cache;
}

/// shortcut: branch probability cut for `deep3`, the same idea as `solver.zig`'s
/// `prob_cutoff` at a coarser scale. A 4 under a 4 is already ~1e-3 of the path at the second spawn
/// layer, so 1e-2 drops those subtrees (about 10% of a node's mass, renormalized away) while the
/// last spawn layer next to the leaves stays full width. Raise it for speed, lower it for accuracy,
/// 0 disables the cut (that is `.deep`).
const deep3_cutoff: f64 = 1e-2;

/// The two spawns of one empty cell, with their probabilities.
const spawn_tiles = [_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } };

/// One spawn branch of a chance node: where the tile lands, what it is, and its probability
/// `cprob * spawn.p`. Collected in board order, which is also the tie-break of the mass cut.
const ChanceBranch = struct { r: usize, c: usize, tile: core.Tile, p: f64 };

fn spawnProbability(tile: core.Tile) f64 {
    for (spawn_tiles) |spawn| if (spawn.value == tile) return spawn.p;
    unreachable;
}

/// The cutoff fallback keeps a cut branch's board and drops exactly one own-move layer.
fn spawnBranchDepth(depth: u8, cut: bool, recover_cut_branches: bool) u8 {
    return if (cut and recover_cut_branches and depth > 1) depth - 1 else depth;
}

/// Expected value over a complete chance node, with each branch's local probability as its weight.
fn weightedChanceValue(values: []const f64, probabilities: []const f64) f64 {
    std.debug.assert(values.len == probabilities.len);
    var total: f64 = 0;
    for (values, probabilities) |value, probability| total += value * probability;
    return total;
}

/// A full board's worth of branches: at most one spawn pair per cell.
const max_chance_branches = core.BoardSize * core.BoardSize * spawn_tiles.len;

/// The tail of the downside-risk mix: a chance node's CVaR is the probability-weighted mean
/// of its branches below this lower quantile of the node's mass.
const risk_alpha: f64 = 0.25;

/// The CVaR of one chance node: the probability-weighted mean of the branches below the node's
/// lower `alpha` quantile, so a node that is mostly good but sometimes ruinous scores worse than one
/// that is always equally good. `values` and `probs` are the kept branches of the node in board order;
/// ties keep that order, so the result is deterministic and the search stays tie-break stable.
/// The smallest branch is always in the tail, even when it alone carries more than `alpha` of the mass,
/// so the tail is never empty and `alpha <= 1` always describes a whole branch.
fn downsideRisk(values: []const f64, probs: []const f64, alpha: f64) f64 {
    std.debug.assert(values.len == probs.len and values.len > 0);
    // insertion sort of the indices by value, stable: the board-order tie-break of the mass cut
    var order: [max_chance_branches]usize = undefined;
    for (0..values.len) |i| {
        var j = i;
        while (j > 0 and values[order[j - 1]] > values[i]) : (j -= 1) order[j] = order[j - 1];
        order[j] = i;
    }
    var all_p: f64 = 0;
    for (probs) |p| all_p += p;
    const tail_mass = alpha * all_p;
    var tail_p: f64 = 0;
    var tail_v: f64 = 0;
    for (order[0..values.len]) |i| {
        // the first branch is always taken; after that the tail stops before the branch that would
        // cross the quantile, so the mean is over branches that are wholly below it
        if (tail_p > 0 and tail_p + probs[i] > tail_mass) break;
        tail_v += values[i] * probs[i];
        tail_p += probs[i];
    }
    return tail_v / tail_p;
}

/// The score of a chance node: its probability-weighted `mean` moved toward the node's CVaR by
/// `risk_weight` (`0` = the plain mean, the pre-#187 value, with no tail work at all; `1` = CVaR
/// alone). Because the tail mean of the lowest branches never exceeds the mean over all of them,
/// `risk_weight` in `0..1` only ever lowers a node's score and never reorders it above its own mean.
fn riskScore(mean: f64, values: []const f64, probs: []const f64, risk_weight: f64) f64 {
    if (risk_weight == 0) return mean;
    return (1 - risk_weight) * mean + risk_weight * downsideRisk(values, probs, risk_alpha);
}

/// Marks the branches `mass_cutoff` drops and returns the share of the node's mass they carry:
/// the lowest `p` first, dropped while their cumulative share stays within `mass_cutoff`; a branch that
/// would push it over stays. Equal probabilities keep `branches` order, and the heaviest branch is never
/// dropped, so `mass_cutoff` below 1 always keeps at least one branch. `0` (and the full-width
/// spelling 1) drops nothing.
fn massCutMask(branches: []const ChanceBranch, mass_cutoff: f64, dropped: *[max_chance_branches]bool) f64 {
    if (mass_cutoff <= 0 or mass_cutoff >= full_width_cutoff or branches.len < 2) return 0;
    var all_p: f64 = 0;
    for (branches) |b| all_p += b.p;
    const limit = mass_cutoff * all_p;
    // insertion sort of the indices by `p`, stable, so equal probabilities stay in board order
    var order: [max_chance_branches]usize = undefined;
    for (0..branches.len) |i| {
        var j = i;
        while (j > 0 and branches[order[j - 1]].p > branches[i].p) : (j -= 1) order[j] = order[j - 1];
        order[j] = i;
    }
    var dropped_p: f64 = 0;
    for (order[0..branches.len]) |i| {
        if (dropped_p + branches[i].p > limit) break;
        dropped[i] = true;
        dropped_p += branches[i].p;
    }
    return dropped_p / all_p;
}

/// a deterministic uniform in (0, 1) for one branch of one chance node. The node's board hash,
/// depth and the branch's index in board order are hashed with the run seed, so the same node under
/// the same seed always proposes the same spawns. `cprob` is deliberately not an input: it scales
/// every branch weight of a node equally, which leaves a weighted sample unchanged.
fn sampleUniform(seed: u64, board_hash: u64, depth: u8, index: usize) f64 {
    var h = seed ^ board_hash;
    h ^= @as(u64, depth) *% 0x9e3779b97f4a7c15;
    h +%= @as(u64, @intCast(index)) *% 0x165667b19e3779f9;
    // splitmix64 finalizer
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    // 53 random bits, shifted off both ends so the result is never exactly 0 (log would be -inf) or 1
    return (@as(f64, @floatFromInt(h >> 11)) + 0.5) / 9007199254740992.0;
}

/// marks all but `keep` branches of one chance node as dropped, choosing them by weighted
/// sampling without replacement (Efraimidis-Spirakis: keep the largest `u^(1/p)`, equivalently the
/// smallest `-ln(u)/p`). Probabilities are the same `cprob * spawn.p` the full enumeration averages
/// with, so a kept set still renormalizes over its own mass below and the 2/4 ratio is preserved.
/// Every key depends only on the node identity and the branch index, so the choice is deterministic
/// for a seed. `keep` at or above the branch count drops nothing, and branches already dropped by the
/// mass cut are never brought back.
fn sampleMask(branches: []const ChanceBranch, keep: usize, seed: u64, board_hash: u64, depth: u8, dropped: *[max_chance_branches]bool) void {
    if (keep == 0) return;
    var available: usize = 0;
    for (branches, 0..) |_, i| {
        if (!dropped[i]) available += 1;
    }
    if (keep >= available) return;
    var keys: [max_chance_branches]f64 = undefined;
    var taken: [max_chance_branches]bool = @splat(false);
    for (branches, 0..) |branch, i| {
        keys[i] = if (dropped[i]) std.math.inf(f64) else -@log(sampleUniform(seed, board_hash, depth, i)) / branch.p;
    }
    for (0..keep) |_| {
        var best: ?usize = null;
        for (branches, 0..) |_, i| {
            if (taken[i] or dropped[i]) continue;
            if (best == null or keys[i] < keys[best.?]) best = i;
        }
        taken[best.?] = true;
    }
    for (branches, 0..) |_, i| {
        if (!taken[i]) dropped[i] = true;
    }
}

/// Depth-limited expectimax: our moves and every spawn (2 at 0.9, 4 at 0.1), no RNG.
/// The caller owns the value cache so a game can reuse one workspace for all of its moves.
const Search = struct {
    vfn: Value,
    cache: *ValueCache,
    /// Optional interior-node table; null (the default) searches exactly as before.
    nodes: ?*NodeCache = null,
    stats: Stats = .{},
    /// Spawn branches under this path probability are dropped; 0 searches every branch (`.deep`).
    cutoff: f64 = 0,
    /// Chance nodes may drop their lowest-probability branches while the dropped share of the node's
    /// mass stays within this value; 0 searches every branch, as `cutoff = 0` did.
    mass_cutoff: f64 = 0,
    /// : how far a chance node's score moves from its mean toward its CVaR. `0` (the default)
    /// returns `total / mass` exactly as before, bit for bit; `1` scores the node by its CVaR alone.
    /// Only the aggregation at chance nodes changes, never a leaf, so it is not part of the cache key.
    risk_weight: f64 = 0,
    /// Own-move plies this whole search runs at; with `cutoff` it keys every leaf lookup.
    ply: u8 = 0,
    /// : omit the reward of a move whose following chance node resolves directly to a leaf.
    omit_leaf_reward: bool = false,
    /// : evaluate cutoff spawn children one own-move layer shallower instead of dropping their mass.
    recover_cut_branches: bool = false,
    /// : average every leaf over the 8 dihedral transforms instead of one forward pass.
    sym_leaf: bool = false,
    /// : at most this many spawn branches per chance node (0 keeps every branch). The draw is a
    /// pure function of the node's board, its depth and `sample_seed`, so the search still never
    /// touches the game's spawn RNG.
    sample: usize = 0,
    sample_seed: u64 = default_sample_seed,
    /// : chance nodes shallower than this keep every branch (see `Depth.sample_min_depth`).
    sample_min_depth: u8 = 1,
    /// : `leaf` calls this search may still make (cache hits count too). Unbounded by default, so
    /// every pre-#183 tree is searched exactly as before; the iterative deepening sets it per iteration.
    nodes_left: usize = std.math.maxInt(usize),
    /// : a `leaf` found `nodes_left == 0`. The tree above it is incomplete, so its value is
    /// meaningless and whoever set the budget throws the whole iteration away.
    out_of_budget: bool = false,

    /// `cache.getValue` at this search's depth: one entry per (board, ply, cutoff, sym_leaf).
    fn leaf(self: *Search, board: Board) f64 {
        if (self.nodes_left == 0) {
            self.out_of_budget = true;
            return 0;
        }
        self.nodes_left -= 1;
        return self.cache.getValueAtDepthSym(self.vfn, board, .{
            .ply = self.ply,
            .cutoff = self.cutoff,
            .mass_cutoff = self.mass_cutoff,
        }, self.sym_leaf, &self.stats);
    }

    /// Completed-node lookup in the table; null when this search has no table or the slot holds
    /// another node. The counters stay at 0 without a table, so every pre-#263 search is unchanged.
    fn nodeGet(self: *Search, board: Board, kind: NodeCache.NodeKind, depth: u8, cprob: f64) ?SearchResult {
        const nodes = self.nodes orelse return null;
        return nodes.get(NodeCache.keyFor(board, kind, depth, cprob), &self.stats);
    }

    /// Store a completed node. A value computed after the leaf budget ran out is meaningless (its
    /// caller discards the iteration), so it must never be reused as if it were exact.
    fn nodePut(self: *Search, board: Board, kind: NodeCache.NodeKind, depth: u8, cprob: f64, result: SearchResult) void {
        const nodes = self.nodes orelse return;
        if (self.out_of_budget) return;
        nodes.put(NodeCache.keyFor(board, kind, depth, cprob), result);
    }

    /// `value` with the leaf one own move further out and no reward at it: `bootValue(S, 0)` is the
    /// best afterstate value one move from `S`, and every level above adds one reward plus the spawn
    /// average below it, `bootValue(S, r) = max_d (score_delta / reward_scale +
    /// E_spawn[bootValue(S', r - 1)])`. So `bootValue(S, r)` is `r` rewards and the value of the
    /// afterstate `r + 1` moves ahead: exactly one move past the last reward it summed. A leaf taken
    /// at the last reward's afterstate (what `value` does) would count that reward twice.
    fn bootValue(self: *Search, state: GameState, rewards: u8, cprob: f64) f64 {
        if (self.out_of_budget) return 0; // #183: the caller discards this iteration
        var best: f64 = -std.math.inf(f64);
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            var next = state;
            const moved = next.slide(d);
            if (!moved.changed) continue;
            const v = if (rewards == 0)
                self.leaf(next.board)
            else
                @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + self.bootChance(next, rewards - 1, cprob);
            best = @max(best, v);
        }
        return if (best == -std.math.inf(f64)) 0 else best;
    }

    /// Chance node of `bootValue`: `chance`'s spawn width, branch cut and mass renormalization, one
    /// reward count shallower. With nothing to spawn the search keeps moving at the same count; a node
    /// whose every branch is cut does the same, because the afterstate leaf here would be the one the
    /// last reward already reached.
    fn bootChance(self: *Search, state: GameState, rewards: u8, cprob: f64) f64 {
        if (self.out_of_budget) return 0; // #183
        var empties: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] == 0) empties += 1;
            }
        }
        if (empties == 0) return self.bootValue(state, rewards, cprob);
        var total: f64 = 0;
        var mass: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] != 0) continue;
                for ([_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } }) |spawn| {
                    const p = cprob * spawn.p;
                    // the spawn layer right above the leaves stays full width, as in `chance`
                    if (rewards > 0 and p / empties < self.cutoff) continue;
                    var child = state;
                    child.board[r][c] = spawn.value;
                    total += p * self.bootValue(child, rewards, p);
                    mass += p;
                }
            }
        }
        return if (mass == 0) self.bootValue(state, rewards, cprob) else total / mass;
    }

    /// Max over legal moves of reward + expected value after the spawn; 0 when no move changes the board.
    fn valueResult(self: *Search, state: GameState, depth: u8, cprob: f64) SearchResult {
        if (self.out_of_budget) return .{ .value = 0 }; // #183: the caller discards this iteration
        if (self.nodeGet(state.board, .value, depth, cprob)) |cached| return cached;
        var best: f64 = -std.math.inf(f64);
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            var next = state;
            const moved = next.slide(d);
            if (!moved.changed) continue;
            const continuation = self.chanceResult(next, depth - 1, cprob);
            const reward = if (self.omit_leaf_reward and continuation.direct_leaf)
                0
            else
                @as(f64, @floatFromInt(moved.score_delta)) / reward_scale;
            best = @max(best, reward + continuation.value);
        }
        const result: SearchResult = .{ .value = if (best == -std.math.inf(f64)) 0 else best };
        self.nodePut(state.board, .value, depth, cprob, result);
        return result;
    }

    fn value(self: *Search, state: GameState, depth: u8, cprob: f64) f64 {
        return self.valueResult(state, depth, cprob).value;
    }

    /// Expected value of a post-move board after spawning; depth 0 is the value leaf (the afterstate itself).
    /// `cprob` is the probability of reaching this node. Kept branches are averaged over their own
    /// mass, so a cut node keeps the scale of a full average instead of shrinking with the dropped ones.
    /// Two passes: the first collects every branch's probability, the second evaluates the ones
    /// the fixed `cutoff` and the mass `mass_cutoff` both keep.
    /// Direct leaves include both the depth boundary and all-pruned chance nodes; a full board instead
    /// keeps moving at the same depth, and a terminal board's value remains 0 (no legal move).
    fn chanceResult(self: *Search, state: GameState, depth: u8, cprob: f64) SearchResult {
        if (depth == 0) return .{ .value = self.leaf(state.board), .direct_leaf = true };
        if (self.out_of_budget) return .{ .value = 0 }; // #183: the caller discards this iteration
        if (self.nodeGet(state.board, .chance, depth, cprob)) |cached| return cached;
        var empties: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] == 0) empties += 1;
            }
        }
        if (empties == 0) return self.valueResult(state, depth, cprob);
        // pass 1: every branch of this node, in board order (the tie-break of the mass cut)
        var branches: [max_chance_branches]ChanceBranch = undefined;
        var count: usize = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] != 0) continue;
                for (spawn_tiles) |spawn| {
                    branches[count] = .{ .r = r, .c = c, .tile = spawn.value, .p = cprob * spawn.p };
                    count += 1;
                }
            }
        }
        std.debug.assert(count <= max_chance_branches);
        // the mass cut drops the smallest branches first, the fixed cut its usual depth > 1 way
        var dropped: [max_chance_branches]bool = @splat(false);
        _ = massCutMask(branches[0..count], self.mass_cutoff, &dropped);
        // : keep only a finite weighted sample of this node's spawns. The dropped branches follow
        // the cut path below, so the kept ones renormalize over their own mass exactly as a cut node
        // does and the value keeps the scale of a full average. Not combined with the recovery:
        // that row evaluates every branch one layer shallower, which sampling would silently undo.
        if (self.sample > 0 and depth >= self.sample_min_depth and !self.recover_cut_branches) {
            sampleMask(branches[0..count], self.sample, self.sample_seed, ValueCache.keyOf(state.board), depth, &dropped);
        }
        // : if the fixed cutoff removes any branch, evaluate every spawn from its real child
        // board. Kept branches retain this depth; cut branches lose one own-move/reward layer. Their
        // local probabilities sum to 1, so no mass is discarded or renormalized.
        var has_cut_branch = false;
        if (self.recover_cut_branches) {
            for (branches[0..count], 0..) |branch, i| {
                if (!dropped[i] and depth > 1 and branch.p / empties < self.cutoff) {
                    has_cut_branch = true;
                    break;
                }
            }
        }
        if (has_cut_branch) {
            var values: [max_chance_branches]f64 = undefined;
            var probabilities: [max_chance_branches]f64 = undefined;
            var evaluated: usize = 0;
            for (branches[0..count], 0..) |branch, i| {
                if (dropped[i]) continue;
                const cut = depth > 1 and branch.p / empties < self.cutoff;
                var child = state;
                child.board[branch.r][branch.c] = branch.tile;
                const result = self.valueResult(child, spawnBranchDepth(depth, cut, true), branch.p);
                values[evaluated] = result.value;
                probabilities[evaluated] = spawnProbability(branch.tile) / empties;
                evaluated += 1;
            }
            return .{ .value = riskScore(weightedChanceValue(values[0..evaluated], probabilities[0..evaluated]), values[0..evaluated], probabilities[0..evaluated], self.risk_weight) };
        }

        // Legacy full/cut path: unchanged when there are no fallback branches. Retained branches
        // are averaged over their own mass, preserving the existing deep3 behavior bit for bit.
        var total: f64 = 0;
        var mass: f64 = 0;
        var kept_values: [max_chance_branches]f64 = undefined;
        var kept_probs: [max_chance_branches]f64 = undefined;
        var kept: usize = 0;
        for (branches[0..count], 0..) |branch, i| {
            if (dropped[i] or (depth > 1 and branch.p / empties < self.cutoff)) continue;
            var child = state;
            child.board[branch.r][branch.c] = branch.tile;
            const result = self.valueResult(child, depth, branch.p);
            total += branch.p * result.value;
            mass += branch.p;
            kept_values[kept] = result.value;
            kept_probs[kept] = branch.p;
            kept += 1;
        }
        // every branch cut: the node is already that unlikely, so the afterstate leaf is the answer
        const result: SearchResult = if (mass == 0)
            .{ .value = self.leaf(state.board), .direct_leaf = true }
        else
            .{ .value = riskScore(total / mass, kept_values[0..kept], kept_probs[0..kept], self.risk_weight) };
        self.nodePut(state.board, .chance, depth, cprob, result);
        return result;
    }

    fn chance(self: *Search, state: GameState, depth: u8, cprob: f64) f64 {
        return self.chanceResult(state, depth, cprob).value;
    }
};

/// The fixed root move order of every pre-#183 search: the tie-break of `deepMove` is "first move in
/// this order with a strictly larger value".
const dir_order = [4]Direction{ .up, .left, .down, .right };

/// One root's best move and its score (reward + `Search.chance`), `dir = null` on a dead board.
const RootBest = struct {
    dir: ?Direction = null,
    value: f64 = -std.math.inf(f64),
    /// The runner-up, which only `Rerank` and the tie-margin deepening read.
    second_dir: ?Direction = null,
    second_value: f64 = -std.math.inf(f64),
};

fn rootMoveScore(search: *Search, state: GameState, dir: Direction) ?f64 {
    var next = state;
    const moved = next.slide(dir);
    if (!moved.changed) return null;
    const continuation = search.chanceResult(next, search.ply, 1);
    const reward = if (search.omit_leaf_reward and continuation.direct_leaf)
        0
    else
        @as(f64, @floatFromInt(moved.score_delta)) / reward_scale;
    return reward + continuation.value;
}

/// The root layer of `deepMove`: max over `order` of reward + expected value after the spawn. `order`
/// is the fixed one except in iterative deepening, where the previous iteration's best move
/// goes first; moving a later direction earlier only breaks ties differently if the two scores are
/// bit-identical, which can only happen on a symmetric board.
fn rootMove(search: *Search, state: GameState, order: [4]Direction, rerank: ?Rerank) RootBest {
    var best: RootBest = .{};
    var second: RootBest = .{};
    for (order) |d| {
        const v = rootMoveScore(search, state, d) orelse continue;
        if (best.dir == null or v > best.value) {
            second = best;
            best = .{ .dir = d, .value = v };
        } else if (second.dir == null or v > second.value) {
            second = .{ .dir = d, .value = v };
        }
        // : out of budget, so this root's score is garbage — stop before spending another branch
        if (search.out_of_budget) return best;
    }
    best.second_dir = second.dir;
    best.second_value = second.value;
    // : only a near tie is worth a second opinion, and only a nonzero weight can change the pick.
    if (rerank) |r| {
        if (best.dir != null and second.dir != null and r.weight != 0 and best.value - second.value <= r.margin) {
            const best_other = rerankScore(r, state, best.dir.?);
            const second_other = rerankScore(r, state, second.dir.?);
            if (second.value + r.weight * second_other > best.value + r.weight * best_other) return second;
        }
    }
    return best;
}

/// Opt-in heterogeneous rerank of the top two moves. The 2-ply search still scores every
/// legal move with the net; if the gap between its best and its second move is within `margin`, only
/// those two are re-scored with a cheap foreign evaluator and the move with the larger
/// `nn + weight * other` wins. `null` (the default) or `weight = 0` keeps the plain argmax.
pub const Rerank = struct {
    /// Largest NN-value gap (best - second) that still triggers the rerank. NN values are score
    /// deltas over `reward_scale`, so 0.05 is 50 points of one-move score.
    margin: f64,
    /// Weight of the foreign score in the combined value `nn + weight * other`. 0 = never changes
    /// the pick, whatever the margin.
    weight: f64,
    /// Foreign evaluator of an afterstate board, already on the NN value scale (a
    /// `solver.evaluate` score divided by `reward_scale`).
    score: *const fn (Board) f64,
};

/// `Rerank.score` of a move: the foreign evaluator on the afterstate the move leads to.
fn rerankScore(r: Rerank, state: GameState, d: Direction) f64 {
    var next = state;
    _ = next.slide(d);
    return r.score(next.board);
}

/// `depth.ply` own moves deep: my move -> spawn -> ... -> afterstate value + score_delta/reward_scale.
/// `ply` 2 is `deep` (full width); `ply` 3 is `deep3`, whose cost only fits with a spawn-probability cut;
/// `ply` 4 is `deep4` or `phase_h4`, the former everywhere at that cut and the latter
/// only where the board is nearly full.
fn deepMove(vfn: Value, state: GameState, stats: *Stats, cache: *ValueCache, nodes: ?*NodeCache, depth: Depth, sym_leaf: bool, rerank: ?Rerank, risk_weight: f64) ?Direction {
    var search = newSearch(vfn, cache, nodes, depth, sym_leaf, risk_weight);
    defer stats.* = search.stats;
    return rootMove(&search, state, dir_order, rerank).dir;
}

/// A `Search` over the depth row `depth` (#183 factored this out of `deepMove` so the iterative
/// deepening can rebuild the same search at another ply).
fn newSearch(vfn: Value, cache: *ValueCache, nodes: ?*NodeCache, depth: Depth, sym_leaf: bool, risk_weight: f64) Search {
    return .{
        .vfn = vfn,
        .cache = cache,
        .nodes = nodes,
        .cutoff = depth.cutoff,
        .mass_cutoff = depth.mass_cutoff,
        .ply = depth.ply,
        .omit_leaf_reward = depth.omit_leaf_reward,
        .recover_cut_branches = depth.recover_cut_branches,
        .sym_leaf = sym_leaf,
        .sample = depth.sample,
        .sample_seed = depth.sample_seed,
        .sample_min_depth = depth.sample_min_depth,
        // a phase row may carry its own weight; the caller's option is the one every other row uses
        .risk_weight = if (depth.risk_weight != 0) depth.risk_weight else risk_weight,
    };
}

/// Highest ply an iterative deepening iteration may reach. The budget normally stops it long before;
/// the cap only keeps a huge budget from running away into a tree no cache can hold.
const max_id_ply: u8 = 6;

/// Result of `deepMoveId`.
pub const IdMove = struct {
    /// The move of the deepest *completed* iteration; `null` only when no iteration completed (a budget
    /// too small for one full 2-ply tree, e.g. 0), in which case the caller falls back to a legal move.
    dir: ?Direction,
    /// Ply of the deepest completed iteration, 0 when none completed.
    ply: u8,
    /// Leaf evaluations spent over every iteration, the discarded partial one included.
    leaves: u64,
};

/// Iterative deepening over the ply: search the start depth, then one ply deeper, and keep the
/// best move of the last iteration that finished, which is what makes a truncated iteration harmless
/// (a partial tree is never adopted). The budget is leaf evaluations, never wall clock, so the same
/// board and budget always give the same move.
///
/// The start row's `cutoff` / `mass_cutoff` apply to every iteration: the deeper iterations are the
/// same tree one spawn layer out, not a different width. The previous iteration's root move goes first
/// in the next iteration's move order, so with a budget that runs out mid-iteration the deeper tree is
/// explored from its most promising branch outward.
fn deepMoveId(vfn: Value, state: GameState, stats: *Stats, cache: *ValueCache, nodes: ?*NodeCache, start: Depth, budget: usize, sym_leaf: bool, rerank: ?Rerank, risk_weight: f64, margin: ?f64) IdMove {
    var order = dir_order;
    var spent: u64 = 0;
    var hits: u64 = 0;
    var node_probes: u64 = 0;
    var node_hits: u64 = 0;
    var best: ?Direction = null;
    var best_ply: u8 = 0;
    var ply: u8 = @max(start.ply, 2);
    while (ply <= max_id_ply) : (ply += 1) {
        var search = newSearch(vfn, cache, nodes, .{ .ply = ply, .cutoff = start.cutoff, .mass_cutoff = start.mass_cutoff, .omit_leaf_reward = start.omit_leaf_reward, .recover_cut_branches = start.recover_cut_branches, .sample = start.sample, .sample_seed = start.sample_seed, .sample_min_depth = start.sample_min_depth }, sym_leaf, risk_weight);
        // what this iteration may still spend: the budget minus every iteration before it
        search.nodes_left = budget -| @as(usize, @intCast(spent));
        const root = rootMove(&search, state, order, rerank);
        spent += search.stats.leaves;
        hits += search.stats.hits;
        node_probes += search.stats.node_probes;
        node_hits += search.stats.node_hits;
        // a partial iteration (or a dead board) keeps the previous iteration's move
        if (search.out_of_budget or root.dir == null) break;
        // : with a tie margin, only the first iteration's near tie justifies a deeper tree. A root
        // move that is already clearly better than its runner-up gains nothing from the extra ply, so
        // the search stops here and returns it (its value does not enter the comparison, only the gap).
        const first_iteration = best == null;
        best = root.dir;
        best_ply = ply;
        if (first_iteration) {
            if (margin) |m| {
                if (root.second_dir == null or root.value - root.second_value > m) break;
            }
        }
        // move ordering: the completed iteration's best move first, the rest of the order as it was
        const first = root.dir.?;
        var next_order = order;
        next_order[0] = first;
        var at: usize = 1;
        for (order) |d| {
            if (d == first) continue;
            next_order[at] = d;
            at += 1;
        }
        order = next_order;
    }
    stats.leaves += spent;
    stats.hits += hits;
    stats.node_probes += node_probes;
    stats.node_hits += node_hits;
    return .{ .dir = best, .ply = best_ply, .leaves = spent };
}

/// the last-resort move when no iteration completed (budget 0 included): the first legal
/// direction in the fixed order, which needs no leaf evaluation at all.
fn firstLegalMove(state: GameState) ?Direction {
    for (dir_order) |d| {
        var next = state;
        if (next.slide(d).changed) return d;
    }
    return null;
}

/// Lookahead of the 2-ply training variants: `search2_ply` own moves below the board being
/// valued, with a spawn layer before each of them — the tree `Mode.deep` visits at this ply.
const search2_ply: u8 = 2;

/// Bootstrap of the `search2` TD target of a post-spawn board: the `.deep` tree's shape with
/// its last reward moved into the leaf, `search2_ply - 1` searched rewards and then `V` of the
/// afterstate one own move past the last of them.
///
/// The per-move target is `r_t + bootValue(S_{t+1}, search2_ply - 1)` = `r_t + r_{t+1} + V(A_{t+2})`:
/// the reward of the step being trained plus the searched reward `r_{t+1}` and the value of the
/// afterstate both moves led to, whose own reward `r_{t+2}` is inside `V` and nowhere else.
/// `Search.value` at the same count instead returns `search2_ply` rewards plus `V` of the afterstate
/// the *last* reward reached, so that last reward lands in the target twice and the fixed point of
/// the `search2` target sits ~1.5x above the true value on a 1-reward-per-move chain.
fn search2Value(vfn: Value, state: GameState, cache: *ValueCache, sym_leaf: bool) f64 {
    // a target is priced between weight updates, so a table filled by an older snapshot would
    // answer for the current weights; start empty, as `teacherValue` does. The search inside
    // this call still reuses its own entries, which all come from the current weights.
    cache.clear();
    // The leaf sits one own move deeper than `value` at the same reward count: key it as such.
    var search: Search = .{ .vfn = vfn, .cache = cache, .ply = search2_ply + 1, .sym_leaf = sym_leaf };
    return search.bootValue(state, search2_ply - 1, 1);
}

/// Self-play move under `Options.teacher`: the mode's own move ranking, `reward +
/// `Search.chance(afterstate)` at the mode's phase row — the moves `Mode.deep` / `Mode.deep3` pick —
/// so `Pick.value` is that chance value (not `V`) and `reward + value` is the teacher's q of the move.
/// `teacherValue` of the board this move is made on is the tree one own move deeper, exactly as
/// `deepMove` scores the same move. Spawns are enumerated, so this draws nothing from the rng.
fn pickTeacherValue(vfn: Value, state: GameState, mode: Mode, cache: *ValueCache, sym_leaf: bool) ?Pick {
    // same as `search2Value`: this pick is priced after the previous move's weight update, so
    // it must not read leaves cached with the older weights.
    cache.clear();
    const depth = phaseDepth(mode, state.board);
    var search: Search = .{
        .vfn = vfn,
        .cache = cache,
        .cutoff = depth.cutoff,
        .mass_cutoff = depth.mass_cutoff,
        .ply = depth.ply,
        .sym_leaf = sym_leaf,
    };
    var best: ?Pick = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        const pick: Pick = .{
            .dir = d,
            .after = next.board,
            .reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale,
            .value = search.chance(next, depth.ply, 1),
        };
        const value = pick.reward + pick.value;
        if (best == null or value > best_value) {
            best = pick;
            best_value = value;
        }
    }
    return best;
}

/// `pickMoveValue` for the 2-ply self-play policy: candidates are ranked by the eval's `.deep`
/// value, `reward + Search.chance(afterstate)` — the `1 + search2_ply` own moves deep that `deepMove`
/// at `ply = search2_ply` scores — so `Pick.value` is that chance value (not `V`), covering
/// `search2_ply` moves and a reward-bearing leaf below the afterstate. `search2Value` bootstraps the
/// same tree with its last reward moved into the leaf, so a `search2` target spans `search2_ply`
/// rewards and V one move past the last of them. Spawns are enumerated, so this draws nothing from
/// the rng and the spawn sequence of the game is the same as with `greedy1`.
/// The `.deep` phase row is `(ply = search2_ply, full width)`, so this is `pickTeacherValue(.., .deep)`:
/// the `d2` cell of plays exactly this move.
fn pickSearch2Value(vfn: Value, state: GameState, cache: *ValueCache, sym_leaf: bool) ?Pick {
    return pickTeacherValue(vfn, state, .deep, cache, sym_leaf);
}

pub const Mode = enum {
    /// 1-ply over afterstates: argmax of reward + V(afterstate).
    greedy,
    /// greedy plus one spawn of chance-node lookahead (see `chanceMove`).
    chance,
    /// 2-ply: one more own move and spawn layer below `chance` (see `deepMove`).
    deep,
    /// 3-ply: `deep` plus one more own move and spawn layer, with the probability cut.
    deep3,
    /// 4-ply: `deep3` plus one more own move and spawn layer at the same 1e-2 cut.
    deep4,
    /// : `deep3` with cutoff spawn children evaluated one own-move layer shallower.
    deep3_recover,
    /// 2-ply with the cumulative-mass cut of instead of the fixed probability cut: the same tree as
    /// `deep`, minus the lowest-probability spawn branches up to `mass2_ply_cutoff` of each node's mass.
    deep_mass,
    /// 2-ply while the board has 8+ empty cells, then 3-ply with the 1e-2 cut.
    phase_h1,
    /// Same start, 3-ply from 5..7 empty cells, and the 1e-4 cut once 4 or fewer are left.
    phase_h2,
    /// 2-ply only down to 10 empty cells, then 3-ply from 6 and the 1e-4 cut near the end.
    phase_h3,
    /// 3-ply everywhere, then 4-ply once 2 or fewer empty cells are left: the endgame is where
    /// the branching factor collapses, so the extra ply is cheap there and it is the ply that decides
    /// 4096 -> 8192.
    phase_h4,
};

/// One row of a phase schedule: a board with at least `min_empty` empty cells searches `ply`
/// own moves deep, dropping spawn branches under path probability `cutoff` (`full_width_cutoff` keeps
/// every branch) and dropping the lowest-probability branches up to `mass_cutoff` of a chance node's
/// mass.
pub const Phase = struct { min_empty: u8, ply: u8, cutoff: f64, mass_cutoff: f64 = 0, risk_weight: f64 = 0, sample: usize = 0, sample_min_depth: u8 = 1 };

/// A row cutoff that keeps every spawn branch: no branch probability ever reaches 1.
const full_width_cutoff: f64 = 1;

/// Tighter cut for the last rows of `phase_h2` / `phase_h3`: the board is nearly full there, so
/// the 3-ply tree stays small even with more spawn branches kept.
const phase_tight_cutoff: f64 = 1e-4;

/// The mass cut of the `deep_mass` row: 5% of every chance node's mass may be dropped, which is
/// the low-probability tail (the 4s under a wide board) rather than a fixed probability floor.
const mass2_ply_cutoff: f64 = 0.05;

/// The search depth of every `Mode`, one table: rows are in descending `min_empty`, so the
/// first row the board's empty cells reach is its phase. `deep` and `deep3` are one row each, which is
/// what keeps their behavior exactly the pre-#157 search.
const phase_schedules = struct {
    const deep = [_]Phase{.{ .min_empty = 0, .ply = 2, .cutoff = full_width_cutoff }};
    const deep3 = [_]Phase{.{ .min_empty = 0, .ply = 3, .cutoff = deep3_cutoff }};
    /// : one row, `deep3` plus one more own move and spawn layer at the same cut, so the only
    /// difference from `deep3` is the extra ply.
    const deep4 = [_]Phase{.{ .min_empty = 0, .ply = 4, .cutoff = deep3_cutoff }};
    /// : the `deep3` tree with the cut spawn children evaluated one own-move layer shallower.
    const deep3_recover = [_]Phase{.{ .min_empty = 0, .ply = 3, .cutoff = deep3_cutoff }};
    /// : one row, the `deep` tree with the cumulative-mass cut
    const deep_mass = [_]Phase{.{ .min_empty = 0, .ply = 2, .cutoff = full_width_cutoff, .mass_cutoff = mass2_ply_cutoff }};
    const phase_h1 = [_]Phase{
        .{ .min_empty = 8, .ply = 2, .cutoff = full_width_cutoff },
        .{ .min_empty = 0, .ply = 3, .cutoff = deep3_cutoff },
    };
    const phase_h2 = [_]Phase{
        .{ .min_empty = 8, .ply = 2, .cutoff = full_width_cutoff },
        .{ .min_empty = 5, .ply = 3, .cutoff = deep3_cutoff },
        .{ .min_empty = 0, .ply = 3, .cutoff = phase_tight_cutoff },
    };
    const phase_h3 = [_]Phase{
        .{ .min_empty = 10, .ply = 2, .cutoff = full_width_cutoff },
        .{ .min_empty = 6, .ply = 3, .cutoff = deep3_cutoff },
        .{ .min_empty = 0, .ply = 3, .cutoff = phase_tight_cutoff },
    };
    /// : `deep3` is the baseline and the only row that changes is the very end, so the rows above
    /// the 3-empty boundary are one row. 4-ply keeps the `deep3` cut because the tree it searches is
    /// already tiny — at 2 empties it is a handful of spawns, not a midgame branching factor.
    const phase_h4 = [_]Phase{
        .{ .min_empty = 3, .ply = 3, .cutoff = deep3_cutoff },
        .{ .min_empty = 0, .ply = 4, .cutoff = deep3_cutoff },
    };
};

/// Rows `mode` searches with, in the order they are matched; empty for the 1-ply modes.
fn schedule(mode: Mode) []const Phase {
    return switch (mode) {
        .greedy, .chance => &.{},
        .deep => &phase_schedules.deep,
        .deep3 => &phase_schedules.deep3,
        .deep4 => &phase_schedules.deep4,
        .deep3_recover => &phase_schedules.deep3_recover,
        .deep_mass => &phase_schedules.deep_mass,
        .phase_h1 => &phase_schedules.phase_h1,
        .phase_h2 => &phase_schedules.phase_h2,
        .phase_h3 => &phase_schedules.phase_h3,
        .phase_h4 => &phase_schedules.phase_h4,
    };
}

/// `Search` runs full width at `cutoff = 0`; the table spells that as `full_width_cutoff` because no
/// branch probability reaches it. Converted once, where a row becomes a search.
fn searchCutoff(cutoff: f64) f64 {
    return if (cutoff >= full_width_cutoff) 0 else cutoff;
}

/// `searchCutoff` for the mass cut: `0` and its full-width spelling 1 both mean keep everything.
fn searchMassCutoff(mass_cutoff: f64) f64 {
    return if (mass_cutoff >= full_width_cutoff) 0 else mass_cutoff;
}

/// The row that applies to `board` under the search `mode`, in `Search`'s own units. Rows are in
/// descending `min_empty` and every schedule ends at 0, so one row always matches.
fn phaseDepth(mode: Mode, board: Board) Depth {
    const rows = schedule(mode);
    std.debug.assert(rows.len != 0);
    const empties = countEmpty(board);
    for (rows) |row| {
        if (empties >= row.min_empty) return .{
            .ply = row.ply,
            .cutoff = searchCutoff(row.cutoff),
            .mass_cutoff = searchMassCutoff(row.mass_cutoff),
            .risk_weight = row.risk_weight,
            .recover_cut_branches = mode == .deep3_recover,
            .sample = row.sample,
            .sample_min_depth = row.sample_min_depth,
        };
    }
    unreachable; // every schedule ends with min_empty = 0
}

fn searchDepth(mode: Mode, board: Board, omit_leaf_reward: bool) Depth {
    var depth = phaseDepth(mode, board);
    depth.omit_leaf_reward = omit_leaf_reward;
    return depth;
}

/// Score one root branch using the same fixed horizon and cutoff as the serial deep search.
pub fn scoreRootMove(net: *const Net, state: GameState, mode: Mode, dir: Direction, cache: *ValueCache) ?f64 {
    return scoreRootMoveValue(.{ .single = net }, state, mode, dir, cache);
}

/// `scoreRootMove` for any `Value`, so the f32 build scores root branches with the f32 net (#348).
pub fn scoreRootMoveValue(vfn: Value, state: GameState, mode: Mode, dir: Direction, cache: *ValueCache) ?f64 {
    if (mode != .deep and mode != .deep3) return null;
    const depth = searchDepth(mode, state.board, false);
    const nodes: ?*NodeCache = if (depth.ply >= 3) blk: {
        const table = sharedNodeCache();
        table.clear();
        break :blk table;
    } else null;
    var search = newSearch(vfn, cache, nodes, depth, false, 0);
    return rootMoveScore(&search, state, dir);
}

pub fn bestMove(net: *const Net, state: GameState, mode: Mode) ?Direction {
    return bestMoveValue(.{ .single = net }, state, mode);
}

/// `bestMove` for a single or staged value function.
pub fn bestMoveValue(vfn: Value, state: GameState, mode: Mode) ?Direction {
    var stats: Stats = .{};
    return bestMoveWithCacheValue(vfn, state, mode, &stats, sharedValueCache());
}

/// `bestMove` plus search counters (leaves = `forward` evaluations, hits = value-cache reuse).
/// Uses a fresh cache per call, so the counters describe exactly this search.
pub fn bestMoveWithStats(net: *const Net, state: GameState, mode: Mode, stats: *Stats) ?Direction {
    var cache: ValueCache = .{};
    return bestMoveWithCache(net, state, mode, stats, &cache);
}

/// `bestMoveWithStats` with a caller-owned cache: reuse it across moves for speed.
/// A warm cache returns the same moves as a fresh one — only faster.
pub fn bestMoveWithCache(net: *const Net, state: GameState, mode: Mode, stats: *Stats, cache: *ValueCache) ?Direction {
    return bestMoveWithCacheValue(.{ .single = net }, state, mode, stats, cache);
}

pub fn bestMoveWithCacheValue(vfn: Value, state: GameState, mode: Mode, stats: *Stats, cache: *ValueCache) ?Direction {
    return bestMoveWithCacheValueOpts(vfn, state, mode, stats, cache, .{});
}

/// `bestMoveWithCacheValue` with the search options: `sym_leaf` averages every leaf, and
/// `id_budget` swaps the fixed-depth search for a node-budgeted iterative deepening.
/// `rerank` re-scores the top two moves of a near tie with a foreign evaluator.
pub fn bestMoveWithCacheValueOpts(vfn: Value, state: GameState, mode: Mode, stats: *Stats, cache: *ValueCache, opts: Options) ?Direction {
    // `.greedy` / `.chance` have no phase schedule (and no search), so only search modes resolve a depth.
    var depth: Depth = undefined;
    var searching = false;
    switch (mode) {
        .greedy, .chance => {},
        else => {
            depth = searchDepth(mode, state.board, opts.omit_leaf_reward);
            if (opts.cutoff) |cutoff| depth.cutoff = cutoff;
            if (opts.sample) |sample| depth.sample = sample;
            depth.sample_seed = opts.sample_seed;
            depth.sample_min_depth = opts.sample_min_depth;
            searching = true;
        },
    }
    // : one interior-node table per top-level search, cleared here because settings and net are
    // not part of its key (a table surviving a weight update would price boards with the old net).
    // : 3-ply and deeper default to the table (same moves, ~40% cheaper); 2-ply keeps no table.
    const nodes: ?*NodeCache = if (opts.node_tt orelse (searching and depth.ply >= 3)) blk: {
        const table = sharedNodeCache();
        table.clear();
        break :blk table;
    } else null;
    // : a node budget means iterative deepening. `mode` only sets the first iteration's depth (its
    // phase row, so `.deep` starts at ply 2 and `.deep3` at ply 3) and the spawn cut every ply keeps.
    if (opts.id_budget) |budget| {
        const dir = switch (mode) {
            .greedy, .chance => if (pickMoveValue(vfn, state, false)) |p| p.dir else null,
            .deep, .deep3, .deep4, .deep3_recover, .deep_mass, .phase_h1, .phase_h2, .phase_h3, .phase_h4 => deepMoveId(vfn, state, stats, cache, nodes, depth, budget, opts.sym_leaf, opts.rerank, opts.risk_weight, opts.id_margin).dir,
        };
        // a budget too small for one full iteration still has to answer with a legal move
        return dir orelse firstLegalMove(state);
    }
    return switch (mode) {
        .greedy => if (pickMoveValue(vfn, state, false)) |p| p.dir else null,
        .chance => chanceMove(vfn, state),
        // every search mode takes its (ply, cutoff, mass cutoff) for this board from the phase table
        .deep, .deep3, .deep4, .deep3_recover, .deep_mass, .phase_h1, .phase_h2, .phase_h3, .phase_h4 => deepMove(vfn, state, stats, cache, nodes, depth, opts.sym_leaf, opts.rerank, opts.risk_weight),
    };
}

pub const PlayResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
    /// `forward` leaf evaluations summed over the game (0 for modes that do not search).
    leaves: u64 = 0,
    /// Interior-node lookups and hits summed over the game (#263; both 0 for a 2-ply search or with
    /// `Options.node_tt = false`).
    node_probes: u64 = 0,
    node_hits: u64 = 0,
    /// The board the game ended on (the tape replay test compares it). Zero for callers that
    /// build a `PlayResult` without playing (the expectimax row of `nn_main`).
    board: Board = std.mem.zeroes(Board),
    /// (#336) Stopped at the caller's time budget. Partial score/tile/moves are diagnostics only.
    timed_out: bool = false,
};

/// Play one game to game over with the network policy.
pub fn play(net: *const Net, rng: std.Random, mode: Mode) PlayResult {
    return playValue(.{ .single = net }, rng, mode);
}

/// `play` for a single or staged value function.
pub fn playValue(vfn: Value, rng: std.Random, mode: Mode) PlayResult {
    return playValueOpts(vfn, rng, mode, .{});
}

/// `playValue` with the search options: `sym_leaf` averages every search leaf over the 8
/// dihedral transforms. Modes without a search (`.greedy`, `.chance`) are unchanged.
pub fn playValueOpts(vfn: Value, rng: std.Random, mode: Mode, opts: Options) PlayResult {
    var stats: Stats = .{};
    return playWithOptsValue(vfn, rng, mode, &stats, opts);
}

pub fn playWithStats(net: *const Net, rng: std.Random, mode: Mode, stats: *Stats) PlayResult {
    return playWithStatsValue(.{ .single = net }, rng, mode, stats);
}

pub fn playWithStatsValue(vfn: Value, rng: std.Random, mode: Mode, stats: *Stats) PlayResult {
    return playWithOptsValue(vfn, rng, mode, stats, .{});
}

/// `playWithStatsValue` with the search options.
pub fn playWithOptsValue(vfn: Value, rng: std.Random, mode: Mode, stats: *Stats, opts: Options) PlayResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    var timed_out = false;
    var move_stats: Stats = .{};
    var cache: ValueCache = .{}; // one workspace for the whole game (issue #88)
    while (true) {
        if (gameTimeLimitReached(start, opts.game_time_limit_ns)) {
            timed_out = true;
            break;
        }
        move_stats = .{};
        const dir = bestMoveWithCacheValueOpts(vfn, state, mode, &move_stats, &cache, opts) orelse break;
        stats.leaves += move_stats.leaves;
        stats.hits += move_stats.hits;
        stats.node_probes += move_stats.node_probes;
        stats.node_hits += move_stats.node_hits;
        _ = state.move(dir, rng);
        moves += 1;
    }
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .leaves = stats.leaves, .node_probes = stats.node_probes, .node_hits = stats.node_hits, .board = state.board, .timed_out = timed_out };
}

/// One spawn of a game: a board-independent random pick and which of the two values the tile had.
/// The pick gives each board an almost-uniform hole choice, even when the policies diverge.
pub const Spawn = struct {
    /// A common random number, uniform over all `u32` values. Both boards map it onto their own
    /// empty cells with `slotFor`. Each hole owns either floor(2^32 / n) or ceil(2^32 / n) picks, so
    /// the distribution is exact when n divides 2^32 and otherwise differs by at most one pick.
    ///
    /// Recording an index and clamping it to the other board's hole count was biased: with 4 holes
    /// on A and 2 on B, A's index 2 and 3 both became B's hole 1, which then took 75% of B's spawns.
    /// Mapping one common pick into nearly equal-width buckets on both boards keeps each marginal
    /// uniform to 32-bit precision — the textbook common-random-numbers construction.
    pick: u32,
    /// 1 for the 1-in-10 draw that spawns a 4, 0 for a 2.
    four: u1,

    pub fn tile(self: Spawn) core.Tile {
        return if (self.four == 1) 4 else 2;
    }

    /// The hole `pick` selects out of `empty_count`, in `spawnRandomTile`'s row-major order. If
    /// `empty_count` does not divide 2^32, bucket sizes differ by one value out of 4,294,967,296.
    pub fn slotFor(pick: u32, empty_count: usize) usize {
        std.debug.assert(empty_count > 0);
        return @intCast((@as(u64, pick) * @as(u64, @intCast(empty_count))) >> 32);
    }

    /// The `pick` that lands in `slot` out of `empty_count` holes (the middle of the bucket).
    pub fn pickFor(slot: usize, empty_count: usize) u32 {
        std.debug.assert(slot < empty_count);
        const lo = (@as(u64, @intCast(slot)) << 32) / @as(u64, @intCast(empty_count));
        const hi = (@as(u64, @intCast(slot + 1)) << 32) / @as(u64, @intCast(empty_count));
        return @intCast((lo + hi) / 2);
    }

    pub fn at(pick: u32, value: core.Tile) Spawn {
        std.debug.assert(value == 2 or value == 4);
        return .{ .pick = pick, .four = @intFromBool(value == 4) };
    }
};

/// Spawn tape for common random numbers: one game's spawn sequence, recorded while policy A
/// plays and replayed while policy B plays the same game. Spawn luck alone is worth ±5,000 points
/// between two seeds (`docs/solver-history.md`), so two policies compared on the same tapes
/// leave only their own difference in the paired statistics.
pub const Tape = struct {
    pub const Recording = enum { record, replay };

    mode: Recording,
    /// Record: room for the game's spawns, `len` entries are written. Replay: the recorded sequence,
    /// `len` = `spawns.len`.
    spawns: []Spawn,
    len: usize = 0,
    cursor: usize = 0,

    pub fn initRecord(spawns: []Spawn) Tape {
        return .{ .mode = .record, .spawns = spawns };
    }

    pub fn initReplay(spawns: []Spawn) Tape {
        return .{ .mode = .replay, .spawns = spawns, .len = spawns.len };
    }

    /// Spawn one tile through the tape. Record mode draws a full-range `u32` pick and then the tile
    /// value, mapping the pick into this board's holes. This intentionally does not promise the same
    /// game as `playValue` with the same seed: `spawnRandomTile` draws a bounded index instead. Replay
    /// mode maps the recorded pick into the replay board and draws nothing while tape entries remain.
    /// Returns false when the board is full, like `spawnRandomTile`.
    fn spawn(self: *Tape, state: *GameState, rng: std.Random) bool {
        switch (self.mode) {
            .record => {
                var empties: [core.BoardCellCount]core.Position = undefined;
                var empty_count: usize = 0;
                for (0..core.BoardSize) |row| for (0..core.BoardSize) |col| {
                    if (state.board[row][col] == 0) {
                        empties[empty_count] = .{ .row = row, .col = col };
                        empty_count += 1;
                    }
                };
                if (empty_count == 0) return false;
                const pick = rng.int(u32);
                const slot = Spawn.slotFor(pick, empty_count);
                const pos = empties[slot];
                const value: core.Tile = if (rng.uintLessThan(u8, 10) == 0) 4 else 2;
                // a full buffer only stops the recording; the game keeps consuming the same rng draws
                if (self.len < self.spawns.len) {
                    self.spawns[self.len] = Spawn.at(pick, value);
                    self.len += 1;
                }
                state.board[pos.row][pos.col] = value;
                return true;
            },
            .replay => {
                // ran out of tape (the replayed game is longer than the recorded one): draw as usual
                // so the game still ends legally; the pair was on the same stream up to here
                if (self.cursor >= self.len) return state.spawnRandomTile(rng);
                const recorded = self.spawns[self.cursor];
                self.cursor += 1;
                // Use the same row-major empty-cell order as `spawnRandomTile`; map the common pick
                // independently for this board so its own holes retain an almost-uniform distribution.
                var empties: [core.BoardCellCount]core.Position = undefined;
                var empty_count: usize = 0;
                for (0..core.BoardSize) |row| for (0..core.BoardSize) |col| {
                    if (state.board[row][col] == 0) {
                        empties[empty_count] = .{ .row = row, .col = col };
                        empty_count += 1;
                    }
                };
                if (empty_count == 0) return false;
                // the same uniform number, mapped onto this board's holes: B draws from its own uniform
                // distribution over its own holes instead of inheriting a clamped index
                const slot = Spawn.slotFor(recorded.pick, empty_count);
                const pos = empties[slot];
                state.board[pos.row][pos.col] = recorded.tile();
                return true;
            },
        }
    }
};

/// Play with a spawn tape. Record mode generates a tape-specific sequence of common random picks;
/// replay mode maps those picks onto this policy's board so two policies can be compared on shared randomness.
pub fn playTapedValue(vfn: Value, rng: std.Random, mode: Mode, tape: *Tape) PlayResult {
    return playTapedValueOpts(vfn, rng, mode, tape, .{});
}

pub fn playTapedValueOpts(vfn: Value, rng: std.Random, mode: Mode, tape: *Tape, opts: Options) PlayResult {
    var stats: Stats = .{};
    return playTapedWithStatsValueOpts(vfn, rng, mode, tape, &stats, opts);
}

/// `playTapedValue` plus search counters. The tape owns the spawns of the whole game, the two start
/// tiles included, and `GameState.move` is spelled out (slide, then a taped spawn) so record mode draws
/// randomness in the same order (cell pick, then tile value) as `move`'s spawn path.
pub fn playTapedWithStatsValue(vfn: Value, rng: std.Random, mode: Mode, tape: *Tape, stats: *Stats) PlayResult {
    return playTapedWithStatsValueOpts(vfn, rng, mode, tape, stats, .{});
}

/// `playTapedWithStatsValue` with the search options, so a tape recorded by one policy can be
/// replayed with a search option on exactly the same spawns.
pub fn playTapedWithStatsValueOpts(vfn: Value, rng: std.Random, mode: Mode, tape: *Tape, stats: *Stats, opts: Options) PlayResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    var state = GameState.initEmpty();
    _ = tape.spawn(&state, rng);
    _ = tape.spawn(&state, rng);
    var moves: u32 = 0;
    var timed_out = false;
    var move_stats: Stats = .{};
    var cache: ValueCache = .{};
    while (true) {
        if (gameTimeLimitReached(start, opts.game_time_limit_ns)) {
            timed_out = true;
            break;
        }
        move_stats = .{};
        const dir = bestMoveWithCacheValueOpts(vfn, state, mode, &move_stats, &cache, opts) orelse break;
        stats.leaves += move_stats.leaves;
        stats.hits += move_stats.hits;
        stats.node_probes += move_stats.node_probes;
        stats.node_hits += move_stats.node_hits;
        const result = state.slide(dir);
        if (result.changed) _ = tape.spawn(&state, rng);
        moves += 1;
    }
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .leaves = stats.leaves, .node_probes = stats.node_probes, .node_hits = stats.node_hits, .board = state.board, .timed_out = timed_out };
}

/// Paired comparison of two policies on the same spawn tape: `diffs[i]` is `score_b - score_a`
/// of game `i`, summarized as mean ± standard error over the games themselves (not over seed ranges,
/// which would make a single range's stderr 0 and hide the pairing).
pub const PairSummary = struct {
    games: usize = 0,
    mean: f64 = 0,
    stderr: f64 = 0,
    /// Games where B scored above A, and games where it scored below A (ties in neither).
    better: usize = 0,
    worse: usize = 0,
};

/// Aggregate the per-game score differences of a paired run; the mean is the paired effect and the
/// standard error is its uncertainty (the paired design shrinks it, which is the point of ).
pub fn pairStats(diffs: []const f64) PairSummary {
    var out: PairSummary = .{ .games = diffs.len };
    if (diffs.len == 0) return out;
    const st = meanStderr(diffs);
    out.mean = st.mean;
    out.stderr = st.stderr;
    for (diffs) |d| {
        if (d > 0) out.better += 1 else if (d < 0) out.worse += 1;
    }
    return out;
}

/// Buckets for the training diagnostic: empty cells 0..3, 4..7, 8+; moves left 0..9, 10..29, 30+.
pub fn emptyBucket(empties: usize) usize {
    return if (empties <= 3) 0 else if (empties <= 7) 1 else 2;
}

pub fn distBucket(moves_left: usize) usize {
    return if (moves_left < 10) 0 else if (moves_left < 30) 1 else 2;
}

/// Max-tile buckets of the deep-position diagnostic: <=256, 512, 1024, 2048, 4096, 8192+.
/// The tail beyond 4096 is where the value error is concentrated, so the table has to show
/// how many training steps ever reach it.
pub const tile_buckets = 6;

pub fn tileBucket(max_tile: core.Tile) usize {
    if (max_tile <= 256) return 0;
    if (max_tile <= 512) return 1;
    if (max_tile <= 1024) return 2;
    if (max_tile <= 2048) return 3;
    if (max_tile <= 4096) return 4;
    return 5;
}

/// Per-bucket sums over the training steps: TD error |target - V(s)| and value change V(s_{t+1}) - V(s_t).
pub const Diag = struct {
    pub const Stat = struct { count: u64 = 0, abs_err: f64 = 0, drop: f64 = 0 };
    /// Steps still inside the last-30 window; their distance to game over is not known until the game ends.
    const Pending = struct { empties: usize, err: f64, drop: f64 };
    const window = 30;

    stats: [3][3]Stat = @splat(@splat(.{})),
    /// Steps and |TD error| by max tile. No window: the bucket of an afterstate is known at once.
    tiles: [tile_buckets]Stat = @splat(.{}),
    ring: [window]Pending = undefined,
    pushed: usize = 0,

    pub fn add(self: *Diag, empties: usize, moves_left: usize, err: f64, drop: f64) void {
        const s = &self.stats[emptyBucket(empties)][distBucket(moves_left)];
        s.count += 1;
        s.abs_err += err;
        s.drop += drop;
    }

    /// Record one training step of the current game (the afterstate it trained). Steps older than the
    /// window are at least 30 moves from the end.
    pub fn push(self: *Diag, after: Board, err: f64, drop: f64) void {
        const empties = countEmpty(after);
        const tile = &self.tiles[tileBucket(core.maxTile(after))];
        tile.count += 1;
        tile.abs_err += err;
        if (self.pushed >= window) {
            const old = self.ring[self.pushed % window];
            self.add(old.empties, window, old.err, old.drop);
        }
        self.ring[self.pushed % window] = .{ .empties = empties, .err = err, .drop = drop };
        self.pushed += 1;
    }

    /// Call at game end: flush the window with the known distances, then reset for the next game.
    pub fn finish(self: *Diag) void {
        const n = self.pushed;
        for (n -| window..n) |i| {
            const p = self.ring[i % window];
            self.add(p.empties, n - 1 - i, p.err, p.drop);
        }
        self.pushed = 0;
    }
};

/// Thin alias so the existing `nn.countEmpty` callers (training diagnostics, replay buckets) keep
/// compiling; the scan itself is the core board predicate.
pub fn countEmpty(board: Board) usize {
    return core.countEmpty(board);
}

/// TD target of the per-move trainer; see `Options.target`.
pub const Target = enum { td1, search2 };

/// Self-play policy of the per-move trainer; see `Options.selfplay`.
pub const SelfPlay = enum { greedy1, deep2 };

/// Default upper bound on the moves of one self-play game. A healthy game ends in a few
/// thousand moves, so this is far past anything normal; a game that reaches it is pathological.
pub const default_move_cap: u32 = 100_000;

const clockNow = game_limit.clockNow;
const gameStartTime = game_limit.start;

fn gameTimeLimitReached(start: std.Io.Timestamp, limit_ns: ?u64) bool {
    return game_limit.reached(start, limit_ns);
}

/// Opt-in training variants. The default `.{}` is the plain per-move TD(0) of `trainGame`; every
/// option whose default is 0/false/null leaves that path bit-identical.
pub const Options = struct {
    /// Upper bound on the moves of one self-play game, 0 = no cap. Reaching it ends the game with
    /// `GameResult.capped = true` and trains nothing more, and the CLI stops the run before the
    /// checkpoint write. Without it one pathological game can hang a batch for the whole run.
    move_cap: u32 = default_move_cap,
    /// Candidate : adaptive target normalization for the plain TD(0) path only.
    popart: ?*PopArtState = null,
    /// (#336) One game's wall-clock budget; null leaves library calls unlimited. The CLI sets it.
    game_time_limit_ns: ?u64 = null,
    /// Candidate (#321): TD(0) whose target is the expectation over every spawn outcome of the pending
    /// afterstate instead of the one tile the game actually drew. Each empty cell is equally likely and
    /// the tile is 2 with p = 0.9 and 4 with p = 0.1, so the weights sum to 1; every branch is priced
    /// with the same greedy 1-ply `reward + V` the sampled target bootstraps from, with the weights
    /// frozen. The enumeration reads the net only, so it draws nothing from the game's spawn rng and
    /// the spawns, and the moves every seed plays, stay those of the sampled run. Plain per-move TD(0)
    /// with the `greedy1` policy only, see `expectedSpawnIsPlainTd1`.
    expected_spawn: bool = false,
    /// Candidate A: no per-move update during the game; afterwards train every stored afterstate from last to first.
    backward: bool = false,
    /// After the game, re-train the last `replay_k` afterstates once more (0 = off), oldest first.
    replay_k: usize = 0,
    /// Learning rate of that replay pass, as a factor of the training lr.
    replay_lr_scale: f64 = 0.5,
    /// Frozen bootstrap net: the V of the next afterstate in every TD target (online TD, priority
    /// replay, backward/replay sweeps) comes from this net instead of the live one. Self-play still
    /// moves and ranks by the live net, so policy and spawn tape are unchanged. The caller owns the net
    /// and must not mutate it during a parallel batch.
    target_net: ?*const Net = null,
    /// Prioritized replay buffer; null = off. Normal per-move updates run as well.
    priority: ?*Priority = null,
    /// With `priority`, the `m` extra updates draw uniformly from entries with 4..7 empty cells
    /// (midgame) instead of by |TD error|.
    midgame: bool = false,
    /// With `priority`, keep the uniform share over the whole buffer and draw the rest by |TD error|
    /// from the 4..7 empty-cell bucket only.
    midmix: bool = false,
    /// Probability that a self-play move is a uniformly random legal move instead of the policy's
    /// pick. Widens the state distribution; such a move's TD target is the usual `reward + V(after)`.
    explore: f64 = 0,
    /// With `warm_open` > 0 the first `warm_open` moves of every game are random and train nothing;
    /// they never enter a TD target, so only the start distribution widens (unlike `explore`).
    warm_open: u32 = 0,
    /// Apply every per-move TD update to all 8 dihedral transforms of the afterstate, with the same
    /// target (8 samples per visited board, deterministic).
    symmetry: bool = false,
    /// Apply each per-move TD update to one uniformly random dihedral transform of the afterstate.
    /// Keeps one update per move, at `base`'s cost, but draws one rng value per update, so the self-play
    /// spawn stream differs from `base` after the first move.
    rsym: bool = false,
    /// Draw one random dihedral transform per per-move TD update and apply it to both the trained
    /// afterstate and the target's next board, so the update is that of the rotated transition. A
    /// terminal target is the reward alone and is not rotated. Draws one rng value per update.
    rtgt: bool = false,
    /// Potential-based reward shaping with `Phi(board) = shape_c * countEmpty(board) / 15` and
    /// gamma = 1: the target becomes `reward + Phi(next.after) - Phi(p.after) + V(next.after)`
    /// (`reward - Phi(p.after)` when terminal). `0` returns the raw reward through a branch, not
    /// `reward + 0.0`, so `base`/`par` stay bit-identical. Only the per-move `td1` + `greedy1` path
    /// implements it; `validateOptions` refuses combinations that would silently drop the shape.
    shape_c: f64 = 0,
    /// Reward every move with -1 and sweep the afterstates backward at game over, so the net learns
    /// V ≈ -(moves to game over) and the policy plays the fastest game over (a death net). With -1 per
    /// move the TD target is the same for every board, so this must keep the backward full return:
    /// plain TD(0) collapses the net to a constant.
    death: bool = false,
    /// TD(lambda) with accumulating eligibility traces over the parameters; 0 = TD(0).
    lambda: f64 = 0,
    /// TD(lambda) over the `replay_k` sweep `trainAfter` runs: one accumulating trace across the whole
    /// sweep instead of K independent one-step updates. `0` keeps the TD(0) sweep bit for bit;
    /// `replay_k = 1` makes it the gradient itself, matching the sweep up to float associativity.
    sweep_lambda: f64 = 0,
    /// Multiplier on the merged batch delta in `trainBatchOpts`. One batch applies a single SGD step
    /// built from `games` per-game deltas, so the mean step is `1/games` of the serial work for the same
    /// games; `batch_scale` recovers it (`games` = linear scaling rule, `sqrt(games)` = noise-only).
    batch_scale: f64 = 1.0,
    /// Take an Adam step on the merged batch gradient instead of the SGD step, normalising every
    /// parameter's step size. Only `par` implements it: the serial path would need per-sample state.
    adam: ?*Adam = null,
    /// Adam's own step size, independent of `lr` (which only sets how far the per-game nets walk while
    /// the batch gradient accumulates, and cancels out of the normalised step).
    adam_lr: f64 = 0.01,
    /// Weight each per-move sample by how few empty cells its board has, so the tail that decides the
    /// 8192 rate takes a larger step. 0 = off, bit-identical to the historical run: for plain SGD a
    /// per-sample loss weight *is* an `lr` multiplier, so this scales the step (`1 + alpha` at 0 empties,
    /// 1 at 15) instead of touching the loss.
    tail_alpha: f64 = 0,
    /// Ceiling on the TD target, in `reward_scale` units (0 = off). TD(0) with function approximation
    /// has no fixed point when the bootstrapped value runs away, and the 128/32 net did exactly that on
    /// training seed 2 (|w| 1e86, 6,334 NaN between 100,000 and 300,000 games). A score is bounded by
    /// the board (~131 for a 16k tile), so the ceiling keeps every sane update and cuts the
    /// V -> target -> V feedback that explodes.
    clamp: f64 = 0,
    /// L2-norm ceiling on the delta one merged `par` batch applies, 0 = off. The batch delta is
    /// `lr x the sum of the games' gradients`, so its norm grows with the batch while a single sample's
    /// cannot. Scales the whole delta down to the cap, never up — same style as `sym_batch_clip`.
    grad_clip: f64 = 0,
    /// Per-element ceiling on one step's `|dw|`, 0 = off. Applies to the serial per-move step and to
    /// the single merged `par` step; the per-game nets inside a batch only accumulate gradient.
    step_clip: f64 = 0,
    /// True batch symmetry augmentation: one averaged step per afterstate, all 8 transformed gradients
    /// from one net snapshot (unlike `symmetry`, which steps 8 times sequentially).
    sym_batch: bool = false,
    /// Fixed n-step TD on the afterstate chain; 0 = off. When set it owns the game loop and the other
    /// options are ignored. 1 is TD(0); larger n propagates rewards faster.
    nstep: usize = 0,
    /// TD target of the per-move path (`trainGamePerMove`, the default). `td1` is
    /// `reward + V(next afterstate)`. `search2` keeps the eval's `.deep` tree with `search2_ply - 1`
    /// searched rewards and then V of the afterstate one move past the last of them, so no reward is in
    /// the target twice (see `search2Value`). The `backward`/`death`/`nstep` paths keep the 1-ply policy.
    target: Target = .td1,
    /// Self-play move of the per-move path. `greedy1` is the 1-ply argmax; `deep2` plays the eval's
    /// 2-ply search (`.deep`: same tie-break, spawns enumerated, no rng use).
    selfplay: SelfPlay = .greedy1,
    /// Distillation. When set, the per-move target is the search value of the post-spawn board under
    /// this mode (`teacherValue`: the mode's phase row, `ply` own moves and the eval's `V` at the leaf)
    /// and the self-play move is that mode's move (`pickTeacherValue`), so policy and target look at one
    /// tree: `.deep` distils the 2-ply value, `.deep3` the 3-ply one a 2-ply eval could not afford at
    /// play time. `null` keeps `target`/`selfplay`; when set both are ignored. Modes with a phase
    /// schedule only (`.greedy`/`.chance` have none).
    teacher: ?Mode = null,
    /// Append every afterstate the plain per-move path visits whose max tile reached `dump_min_tile`
    /// to `dump` (bounded, overflow dropped). The updates are unchanged.
    dump_record: bool = false,
    /// Deep-position buffer: written by `dump_record`, read by the variants that train from it.
    dump: ?*Dump = null,
    /// One extra TD(0) update from a uniformly drawn deep afterstate of `dump` per self-play step, so
    /// half the update budget stays on the self-play trajectory and half lands on the deep tail.
    deep_mix: bool = false,
    /// Search depth of the drawn deep half's teacher label. `.deep3` matches `tch`; a deep search costs
    /// ~1.5 ms/sample, so `ft2` is slower than `base` per game by about that much.
    deep_mode: Mode = .deep3,
    /// One deep update every `deep_every` self-play steps. `1` is 50% deep and diverges; the stable
    /// offline mix is 1 in 4..8, which is what the CLI passes.
    deep_every: u32 = 1,
    /// Every game starts on a uniformly drawn deep afterstate of `dump` instead of an empty board.
    /// Requires `dump`.
    restart: bool = false,
    /// Average every search leaf (`Search.leaf`) over the 8 dihedral transforms of the board instead of
    /// one forward pass: less evaluation noise at 8x the leaf cost. The 1-ply modes (`.greedy`/`.chance`)
    /// have no `Search` and are unchanged.
    sym_leaf: bool = false,
    /// Omit only a move reward whose next chance node resolves directly to a leaf.
    omit_leaf_reward: bool = false,
    /// Reuse completed `Search.value` / `Search.chance` nodes inside one top-level search. The table is
    /// cleared per move, so the same board/depth/cprob gives the same value as without it. `null` turns
    /// it on for a search whose phase row is 3-ply or deeper and leaves 2-ply (`deep`) without it, so
    /// `deep` stays byte-identical and allocates nothing; `false` is the explicit off switch.
    node_tt: ?bool = null,
    /// Override the spawn-probability cut the mode's phase row would use (`null` = the row's cutoff).
    /// A measurement needs e.g. `.deep3` at 1e-3 instead of 1e-2 without a second mode.
    cutoff: ?f64 = null,
    /// At most this many spawn branches per chance node, drawn by weighted sampling without replacement
    /// from the node's 0.9/0.1 probabilities. `null` keeps every branch; with a finite budget a 4-ply
    /// tree then fits a 3-ply cost. The draw is deterministic in the node's board, its depth and
    /// `sample_seed`, and never consumes the game's spawn RNG or the value cache.
    sample: ?usize = null,
    /// Seed of that sampler, so the same board and seed sample the same spawns.
    sample_seed: u64 = default_sample_seed,
    /// Shallowest chance node to sample (`1` = every spawn layer). At `2` the spawn layer above the
    /// leaves keeps every branch, which keeps the last move's death risk exact.
    sample_min_depth: u8 = 1,
    /// Leaf-evaluation budget of an iterative deepening search, `null` = off. When set,
    /// `bestMoveWithCacheValueOpts` searches depth 2, 3, ... and keeps the best move of the last
    /// iteration that finished inside the budget (`mode` only sets the first depth and the spawn cut).
    /// The budget is what makes cost predictable: no wall clock, so the same board and budget always
    /// pick the same move.
    id_budget: ?usize = null,
    /// With `id_budget`, deepen only while the start depth's best move beats its runner-up by at most
    /// this NN-value gap (score deltas over `reward_scale`, so 0.05 = 50 points of one-move score). A
    /// root move that is already clearly better stops after its first iteration, which is where most of
    /// the saving comes from. `null` = always deepen.
    id_margin: ?f64 = null,
    /// Heterogeneous rerank of the 2-ply search's top two moves by a cheap foreign evaluator when the
    /// net's own gap is within `Rerank.margin`; null = the plain argmax.
    rerank: ?Rerank = null,
    /// Candidate: how far every chance node's score moves from its mean toward its CVaR, so a
    /// move that is usually good but sometimes losing ranks below one that is always equally good.
    /// `0` (the default) is the plain expectation, bit for bit as before; `1` is the CVaR alone. A
    /// phase row with a nonzero `Phase.risk_weight` overrides it for that row.
    risk_weight: f64 = 0,
    /// Candidate `dist`: teacher pairs from `computeTeachers`. Every self-play step also takes one
    /// regression step on a uniformly drawn pair, so 1/2 of the updates are distillation.
    distill: ?*Teachers = null,
    /// Candidate how many self-play steps run between two teacher regressions (`1` = every step,
    /// the historical behavior; `8` puts 1 teacher update in 9). The decay that showed is
    /// caused by the teacher signal drowning the shallow game, so the mix is the knob to sweep.
    distill_every: u32 = 1,
    /// Candidate learning rate of the teacher regression alone; `0` uses the TD rate. A teacher
    /// update that is 10x smaller cannot drag the shared parameters away as fast.
    distill_lr: f64 = 0,
    /// Candidate `dson`: the teacher regression only, with the self-play TD update switched off.
    /// The 50/50 mix is not for value quality: it exists so the shallow half of the game (which the deep
    /// buffer never covers) keeps being trained. `dson` answers the other half of that question — whether
    /// the teacher regression alone is stable, given that the parameters are shared with the early game.
    distill_only: bool = false,
};

/// Shape is implemented only by the per-move `td1` + `greedy1` TD(0) update. The variants
/// below re-train afterstates after the game, use another target, or own a different update path, so
/// they would silently ignore the shaped reward. Refuse them explicitly instead of falling back to
/// the raw reward. `shape_c == 0` is always allowed and takes the untouched historical path.
/// PopArt is also restricted to the plain serial TD(0) path: its EMA lives at `trainStep`, so
/// any other field would change the update, the target or the game against statistics that no longer
/// describe what is being trained.
pub const OptionsError = error{ ShapeRequiresPlainTd1, PopArtRequiresPlainTd1, ExpectedSpawnRequiresPlainTd1 };

/// Everything but the PopArt pointer itself must stay at its default; `move_cap` is a safety
/// bound on one game, not an update, so it is carried over.
fn popartIsPlainTd1(opts: Options) bool {
    return std.meta.eql(opts, Options{ .popart = opts.popart, .move_cap = opts.move_cap, .game_time_limit_ns = opts.game_time_limit_ns });
}

/// Same shape as `popartIsPlainTd1`: the expected-spawn target replaces the sampled bootstrap, so a
/// variant that changes the update, the target or the policy would be silently ignored. `move_cap` and
/// the game time limit bound one game instead of changing an update, so they are carried over.
fn expectedSpawnIsPlainTd1(opts: Options) bool {
    return std.meta.eql(opts, Options{ .expected_spawn = opts.expected_spawn, .move_cap = opts.move_cap, .game_time_limit_ns = opts.game_time_limit_ns });
}

pub fn validateOptions(opts: Options) OptionsError!void {
    if (opts.popart != null and !popartIsPlainTd1(opts)) return error.PopArtRequiresPlainTd1;
    if (opts.expected_spawn and !expectedSpawnIsPlainTd1(opts)) return error.ExpectedSpawnRequiresPlainTd1;
    if (opts.shape_c == 0) return;
    if (opts.nstep > 0 or opts.backward or opts.replay_k != 0 or opts.priority != null or
        opts.target_net != null or opts.death or opts.teacher != null or
        opts.distill != null or opts.distill_only or opts.deep_mix or opts.dump_record or opts.restart or
        opts.rsym or opts.rtgt or opts.lambda > 0 or
        opts.target != .td1 or opts.selfplay != .greedy1)
        return error.ShapeRequiresPlainTd1;
}

/// Prioritized replay: ring buffer of recent TD updates. After each game, `m` extra updates
/// are drawn from the buffer: with probability `uniform_share` uniformly, otherwise with probability
/// proportional to (|TD error| + err_floor)^alpha. The drawn entry's error is refreshed after its update.
pub const Priority = struct {
    pub const cap = 1024;
    pub const err_floor = 1e-3;
    m: usize = 32,
    alpha: f64 = 0.5,
    uniform_share: f64 = 0.5,
    len: usize = 0,
    head: usize = 0,
    items: [cap]Entry = undefined,

    /// One TD update: afterstate, scaled reward, next afterstate (ignored when terminal), last |error|.
    pub const Entry = struct { after: Board, reward: f64, next: Board, terminal: bool, err: f64 };

    pub fn push(self: *Priority, e: Entry) void {
        self.items[self.head] = e;
        self.head = (self.head + 1) % cap;
        self.len = @min(self.len + 1, cap);
    }

    /// Index into items[0..len]: uniform with prob uniform_share, else proportional to priority weight.
    pub fn sample(self: *const Priority, rng: std.Random) usize {
        if (rng.float(f64) < self.uniform_share) return rng.uintLessThan(usize, self.len);
        var total: f64 = 0;
        for (self.items[0..self.len]) |e| total += self.weight(e.err);
        var r = rng.float(f64) * total;
        for (self.items[0..self.len], 0..) |e, i| {
            r -= self.weight(e.err);
            if (r < 0) return i;
        }
        return self.len - 1;
    }

    /// `m` extra updates, midgame variant: uniform over entries with 4..7 empty cells,
    /// whole buffer when none exist.
    pub fn sampleMidgame(self: *const Priority, rng: std.Random) usize {
        var eligible: usize = 0;
        for (self.items[0..self.len]) |e| if (isMidgame(e.after)) {
            eligible += 1;
        };
        if (eligible == 0) return rng.uintLessThan(usize, self.len);
        var k = rng.uintLessThan(usize, eligible);
        for (self.items[0..self.len], 0..) |e, i| {
            if (!isMidgame(e.after)) continue;
            if (k == 0) return i;
            k -= 1;
        }
        unreachable;
    }

    /// Midgame branch of the midmix variant: entries with 4..7 empty cells, by priority weight.
    /// Whole buffer (uniform) when that bucket is empty.
    pub fn sampleMidmix(self: *const Priority, rng: std.Random) usize {
        if (rng.float(f64) < self.uniform_share) return rng.uintLessThan(usize, self.len);
        var total: f64 = 0;
        var eligible: usize = 0;
        for (self.items[0..self.len]) |e| if (isMidgame(e.after)) {
            total += self.weight(e.err);
            eligible += 1;
        };
        if (eligible == 0) return rng.uintLessThan(usize, self.len);
        var r = rng.float(f64) * total;
        var last: usize = 0;
        for (self.items[0..self.len], 0..) |e, i| {
            if (!isMidgame(e.after)) continue;
            last = i;
            r -= self.weight(e.err);
            if (r < 0) return i;
        }
        return last;
    }

    fn isMidgame(board: Board) bool {
        const empty = countEmpty(board);
        return empty >= 4 and empty <= 7;
    }

    fn weight(self: *const Priority, err: f64) f64 {
        return std.math.pow(f64, @abs(err) + err_floor, self.alpha);
    }
};

/// One extra update on a drawn deep afterstate for `ft2`: the target is the same `deep3` teacher
/// sample that `tch` writes offline.
///
/// The code used `pickMove(...).value` — the net's own 1-ply search value — as the target. On the
/// deep domain that is a max over moves of `r + V(spawn)`, so every step pushes `V` up to the best
/// branch's bootstrap: with stale deep boards and `replay_k` on top the net diverged to NaN within a few
/// thousand games at every rate tried (`|w|` 4e275, eval stuck at 11,478). A bounded teacher label
/// keeps the half stable and makes `ft2` the online twin of `tch` + `dist`, where deep3 decides the
/// target and the drawn board only decides where the update lands.
fn deepStep(net: *Net, rng: std.Random, buffer: *Dump, lr: f64, cache: *Cache, mode: Mode) void {
    const board = buffer.draw(rng);
    _ = forward(net, board, cache);
    train(net, cache, deepSampleTarget(net, board, rng, mode), lr);
}

/// The label of one deep afterstate: one spawn from `rng`, then the value of `mode`'s search on it. `tch`
/// (offline) and `ft2` (online) both call this, so the two teacher paths cannot drift apart.
fn deepSampleTarget(net: *const Net, board: Board, rng: std.Random, mode: Mode) f64 {
    var state = GameState{ .board = board, .score = 0 };
    _ = state.spawnRandomTile(rng);
    return teacherValue(.{ .single = net }, state, mode, false);
}

/// Deep-position dump: fixed-capacity buffer of afterstates whose max tile reached
/// `dump_min_tile`. The score does not enter the TD target (reward = score_delta / `reward_scale`),
/// so training from these positions means the same thing as training from them mid-game.
///
/// File layout (all little endian): 8-byte magic `2048-DMP`, u32 entry count, then `count` entries of
/// 16 bytes — one byte per cell, row-major, holding log2 of the tile (0 = empty, so a 4096 is 12 and
/// an 8192 is 13). Tiles are powers of two, so the round trip is exact.
pub const dump_magic = "2048-DMP";
pub const dump_min_tile: core.Tile = 4096;
/// 1 MiB of entries: far more deep positions than one training run needs, small enough to heap-allocate.
pub const dump_cap: usize = 1 << 16;
pub const dump_header_len = 12;
pub const dump_entry_len = core.BoardCellCount;

/// One board as `Dump` and `Teachers` store it: row-major log2 per cell, 0 for empty. Tiles are powers
/// of two, so `decodeBoard` restores the board exactly.
fn encodeBoard(board: Board) [dump_entry_len]u8 {
    var out: [dump_entry_len]u8 = undefined;
    for (0..core.BoardCellCount) |cell| {
        const value = board[cell / core.BoardSize][cell % core.BoardSize];
        out[cell] = if (value == 0) 0 else @intCast(std.math.log2_int(core.Tile, value));
    }
    return out;
}

fn decodeBoard(cells: *const [dump_entry_len]u8) Board {
    var out: Board = undefined;
    for (0..core.BoardCellCount) |cell| {
        const level = cells[cell];
        out[cell / core.BoardSize][cell % core.BoardSize] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
    }
    return out;
}

pub const Dump = struct {
    entries: [dump_cap][dump_entry_len]u8 = undefined,
    len: usize = 0,
    /// Deep afterstates seen after the buffer filled. They are discarded, so the dump stays the first
    /// `dump_cap` ones in training order (no rng involved, hence deterministic).
    dropped: u64 = 0,
    /// Deep afterstates drawn for training (`ft2`, `restart`). Not part of the file.
    draws: u64 = 0,

    /// Record `board` when its max tile reached `dump_min_tile`; shallower boards are ignored.
    pub fn offer(self: *Dump, board: Board) void {
        if (core.maxTile(board) < dump_min_tile) return;
        if (self.len == dump_cap) {
            self.dropped += 1;
            return;
        }
        self.append(board);
    }

    pub fn append(self: *Dump, board: Board) void {
        std.debug.assert(self.len < dump_cap);
        self.entries[self.len] = encodeBoard(board);
        self.len += 1;
    }

    /// The `index`-th stored afterstate, decoded back to tiles.
    pub fn boardAt(self: *const Dump, index: usize) Board {
        return decodeBoard(&self.entries[index]);
    }

    /// Uniform draw of one stored afterstate, counted so the 50/50 mix of `ft2` is observable.
    /// Uniform on purpose: lost with |TD error| and midgame resampling of existing data.
    pub fn draw(self: *Dump, rng: std.Random) Board {
        std.debug.assert(self.len > 0);
        self.draws += 1;
        return self.boardAt(rng.uintLessThan(usize, self.len));
    }

    fn encodeInto(self: *const Dump, bytes: []u8) void {
        var w = blob.Writer.init(bytes);
        w.raw(dump_magic);
        w.writeU32(@intCast(self.len));
        w.raw(std.mem.sliceAsBytes(self.entries[0..self.len]));
    }

    pub fn save(self: *const Dump, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try gpa.alloc(u8, dump_header_len + self.len * dump_entry_len);
        defer gpa.free(bytes);
        self.encodeInto(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    pub fn parse(self: *Dump, bytes: []const u8) !void {
        self.len = 0;
        self.dropped = 0;
        if (bytes.len < dump_header_len) return error.DumpTooShort;
        var r = blob.Reader.init(bytes);
        if (!std.mem.eql(u8, try r.take(dump_magic.len), dump_magic)) return error.BadDumpMagic;
        const count = try r.readU32();
        if (count > dump_cap) return error.DumpTooLarge;
        if (bytes.len != dump_header_len + @as(usize, count) * dump_entry_len) return error.WrongDumpFileSize;
        try r.raw(std.mem.sliceAsBytes(self.entries[0..count]));
        self.len = count;
    }

    pub fn load(self: *Dump, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(dump_header_len + dump_cap * dump_entry_len + 1));
        defer gpa.free(bytes);
        try self.parse(bytes);
    }
};

/// Offline teacher pass: the deep-search value of the stored deep positions, computed once.
///
/// Online distillation (variant `d3`) prices every training move with a 3-ply tree, which costs
/// 30-46 ms/move and puts 100k games at ~35-40 h per seed. The stored deep positions  only need
/// that tree once each: 65,536 x ~20 ms is ~22 min, and the regression that follows runs at the ordinary
/// 0.5 ms/move speed. This is the cheap path to a better `V`, which is what the 8192 rate is limited by
/// (depth alone did not move it: see `docs/solver-history.md` 「深さ別の比較」).
///
/// File layout (all little endian): 8-byte magic `2048-TCH`, u32 entry count, then `count` entries of
/// 20 bytes — 16 bytes board (the post-spawn board, log2 per cell like `Dump`), then the f32 target.
/// The board is post-spawn because `V` is defined on post-spawn boards (that is what `forward` sees).
pub const teacher_magic = "2048-TCH";
/// One `tch` pass is one dump (`dump_cap` = 65,536 samples); the cap is 32x that so several passes can be
/// merged into one teacher file (`tchadd`). The struct is ~40 MB, so it lives on the heap
/// (`nn_main.zig` allocates it), not on the stack.
pub const teacher_cap: usize = 32 * dump_cap;
pub const teacher_header_len = dump_header_len;
pub const teacher_entry_len = dump_entry_len + @sizeOf(f32);

pub const Teachers = struct {
    entries: [teacher_cap][dump_entry_len]u8 = undefined,
    targets: [teacher_cap]f32 = undefined,
    len: usize = 0,
    /// Regression steps drawn from the buffer (`dist`). Not part of the file.
    draws: u64 = 0,

    pub fn append(self: *Teachers, board: Board, target: f64) void {
        std.debug.assert(self.len < teacher_cap);
        self.entries[self.len] = encodeBoard(board);
        self.targets[self.len] = @floatCast(target);
        self.len += 1;
    }

    /// Appends every entry of `other` (a second `tch` pass), stopping at `teacher_cap`.
    /// Returns the number of samples that did not fit.
    pub fn appendBuffer(self: *Teachers, other: *const Teachers) usize {
        const room = teacher_cap - self.len;
        const n = @min(room, other.len);
        for (0..n) |i| {
            @memcpy(&self.entries[self.len + i], &other.entries[i]);
            self.targets[self.len + i] = other.targets[i];
        }
        self.len += n;
        return other.len - n;
    }

    pub fn boardAt(self: *const Teachers, index: usize) Board {
        return decodeBoard(&self.entries[index]);
    }

    pub fn targetAt(self: *const Teachers, index: usize) f64 {
        return self.targets[index];
    }

    fn encodeInto(self: *const Teachers, bytes: []u8) void {
        var w = blob.Writer.init(bytes);
        w.raw(teacher_magic);
        w.writeU32(@intCast(self.len));
        for (0..self.len) |i| {
            w.raw(&self.entries[i]);
            w.writeF32(self.targets[i]);
        }
    }

    pub fn save(self: *const Teachers, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try gpa.alloc(u8, teacher_header_len + self.len * teacher_entry_len);
        defer gpa.free(bytes);
        self.encodeInto(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    pub fn parse(self: *Teachers, bytes: []const u8) !void {
        self.len = 0;
        self.draws = 0;
        if (bytes.len < teacher_header_len) return error.TeacherTooShort;
        var r = blob.Reader.init(bytes);
        if (!std.mem.eql(u8, try r.take(teacher_magic.len), teacher_magic)) return error.BadTeacherMagic;
        const count = try r.readU32();
        if (count > teacher_cap) return error.TeacherTooLarge;
        if (bytes.len != teacher_header_len + @as(usize, count) * teacher_entry_len) return error.WrongTeacherFileSize;
        for (0..count) |i| {
            try r.raw(&self.entries[i]);
            self.targets[i] = try r.readF32();
        }
        self.len = count;
    }

    pub fn load(self: *Teachers, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(teacher_header_len + teacher_cap * teacher_entry_len + 1));
        defer gpa.free(bytes);
        try self.parse(bytes);
    }
};

/// One teacher pass over `dump`: each stored afterstate is completed with one uniformly drawn
/// spawn and priced with `teacherValue` at `mode` (the caller uses `.deep3`). One spawn per position on
/// purpose: the stored target is then a one-sample estimate of the spawn expectation, which is all a
/// single regression step on a single board can express anyway. The spawns come from `rng`, so the pass
/// is reproducible.
///
/// The sample's input is the dump's **afterstate**, not the post-spawn board: every consumer of `V`
/// (the policy at `pickMove`, the search leaf, the `td1` bootstrap) feeds it an afterstate, and the
/// `d3` variant pairs `A_t` with `r_t + deep3(S_{t+1})`. Using the post-spawn board as the input while
/// the target prices the same post-spawn board shifts the input distribution by one tile and puts
/// label noise between the two halves of the update (the TD half says `V(A) = r + V(A')` for the same
/// board contents), so the spawn stays in the target's search and out of the input.
pub fn computeTeachers(net: *const Net, buffer: *Teachers, dump: *const Dump, rng: std.Random, mode: Mode) void {
    buffer.len = 0;
    for (0..dump.len) |i| {
        // the input is the stored afterstate, exactly what every V consumer evaluates
        buffer.append(dump.boardAt(i), deepSampleTarget(net, dump.boardAt(i), rng, mode));
    }
}

/// One regression step on a drawn teacher pair (`dist`). Same gradient path as a TD step; the
/// target comes from the cached teacher file instead of a bootstrap. 1/2 of the steps are these, so the
/// self-play half keeps the early game (which the deep buffer never covers) from drifting.
fn distStep(net: *Net, rng: std.Random, buffer: *Teachers, lr: f64, cache: *Cache) void {
    const index = rng.uintLessThan(usize, buffer.len);
    buffer.draws += 1;
    _ = forward(net, buffer.boardAt(index), cache);
    train(net, cache, buffer.targetAt(index), lr);
}

/// TD(lambda) eligibility trace of the plain per-move path: `trace = lambda * trace + grad V`
/// (gamma = 1: the target of an afterstate is its reward plus the next afterstate value),
/// then `net -= lr * (V - target) * trace`.
fn trainTrace(net: *Net, cache: *const Cache, target: f64, lr: f64, trace: *Net, lambda: f64) void {
    var grad = gradOf(net, cache);
    const d = cache.out - target;
    const tz = flat(trace);
    for (flat(&grad), tz) |g, *z| z.* = lambda * z.* + g;
    for (flat(net), tz) |*w, z| w.* -= lr * d * z;
}

/// Gradient of `forward`'s output with respect to every parameter, using the activations in `cache`.
fn gradOf(net: *const Net, cache: *const Cache) Net {
    var g = std.mem.zeroes(Net);
    var da1: [h1_size]f64 = @splat(0);
    for (0..h2_size) |j| {
        const dz2: f64 = if (cache.z2[j] > 0) net.w3[j] else 0;
        g.w3[j] = cache.a2[j];
        g.b2[j] = dz2;
        for (0..h1_size) |i| {
            g.w2[j][i] = dz2 * cache.a1[i];
            da1[i] += net.w2[j][i] * dz2;
        }
    }
    g.b3 = 1;
    for (0..h1_size) |i| {
        const dz1 = if (cache.z1[i] > 0) da1[i] else 0;
        g.b1[i] = dz1;
        for (cache.rows) |row| g.w1[row][i] += dz1;
    }
    return g;
}

/// Stored afterstates of one game for the variants (ring buffer; the last `step_cap` steps are kept).
/// shortcut: games longer than step_cap only replay/sweep their last step_cap afterstates, upgrade if games get that long.
const step_cap = 8192;
const Step = struct { after: Board, reward: f64 };

/// Cap on the n-step lookahead the window allocates; the CLI variant uses 4.
/// shortcut: longer returns than `nstep_cap` are treated as `nstep_cap`, raise the cap (8 * 72 bytes per game) if tried.
const nstep_cap = 8;

/// Pending steps of an n-step return, oldest first. A step's target is known once the values of the n-1
/// steps after it are: their rewards plus V of the next afterstate (0 at game over).
const NStepWindow = struct {
    slots: [nstep_cap]Step = undefined,
    len: usize = 0,
    n: usize,

    pub const Done = struct { after: Board, target: f64 };

    fn init(n: usize) NStepWindow {
        std.debug.assert(n >= 1);
        return .{ .n = @min(n, nstep_cap) };
    }

    /// Add one step. `value` = V(afterstate) of that step, which closes the target of the step leaving
    /// the window; that step is returned once the window is full, otherwise null.
    fn push(self: *NStepWindow, step: Step, value: f64) ?Done {
        const closed: ?Done = if (self.len == self.n) .{ .after = self.slots[0].after, .target = self.rewardSum() + value } else null;
        if (closed != null) self.dropFront();
        self.slots[self.len] = step;
        self.len += 1;
        return closed;
    }

    /// Game over: the remaining steps, oldest first, with the truncated return of their rewards alone.
    fn nextFinish(self: *NStepWindow) ?Done {
        if (self.len == 0) return null;
        const done: Done = .{ .after = self.slots[0].after, .target = self.rewardSum() };
        self.dropFront();
        return done;
    }

    fn rewardSum(self: *const NStepWindow) f64 {
        var g: f64 = 0;
        for (self.slots[0..self.len]) |s| g += s.reward;
        return g;
    }

    fn dropFront(self: *NStepWindow) void {
        std.mem.copyForwards(Step, self.slots[0 .. self.len - 1], self.slots[1..self.len]);
        self.len -= 1;
    }
};

/// TD(0) on afterstates over one self-play game (greedy 1-ply moves, fixed rng).
/// Target for the previous afterstate = its scaled reward + V(next afterstate); 0 after game over.
/// `diag` (optional) records each step's TD error and value change; it does not touch the rng or the update.
pub fn trainGame(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag) u32 {
    return trainGameOptsUnchecked(net, rng, lr, diag, .{}).score;
}

pub const GameResult = struct {
    score: u32,
    moves: u32,
    capped: bool = false,
    timed_out: bool = false,
    /// Diagnostic partial result on a timeout; never a completed-game reach count.
    max_tile: core.Tile = 0,
};

/// `trainGame` with options. `diag` is only recorded on the default path (`opts` all off).
/// Returns `error.ShapeRequiresPlainTd1` rather than silently dropping a shaped reward on a
/// path that does not implement it.
pub fn trainGameOpts(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) OptionsError!GameResult {
    try validateOptions(opts);
    return trainGameOptsUnchecked(net, rng, lr, diag, opts);
}

fn trainGameOptsUnchecked(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) GameResult {
    if (opts.nstep > 0) return trainGameNStep(net, rng, lr, opts.nstep, opts.move_cap, opts.game_time_limit_ns);
    if (opts.priority != null and !opts.death) return trainGamePerMove(net, rng, lr, diag, opts);
    if (!opts.backward and opts.replay_k == 0 and !opts.death) return trainGamePerMove(net, rng, lr, diag, opts);
    const start = gameStartTime(opts.game_time_limit_ns);
    var steps: [step_cap]Step = undefined;
    var n: usize = 0;
    var state = GameState.gameStart(rng);
    var cache: Cache = .{};
    var pending: ?Pick = null;
    while (true) {
        // Stop before the pick: a capped game is pathological, and its unfinished tail must not
        // enter any target. The check reads no rng, so every normal game is unchanged.
        if (opts.move_cap > 0 and n >= @as(usize, opts.move_cap)) return .{ .score = state.score, .moves = @intCast(n), .capped = true };
        if (gameTimeLimitReached(start, opts.game_time_limit_ns)) return .{ .score = state.score, .moves = @intCast(n), .capped = true, .timed_out = true, .max_tile = core.maxTile(state.board) };
        const pick = pickMove(net, state, opts.death) orelse break;
        // With replay_k > 0 this path, not `tdBootstrap`, is the online TD update. Keep the
        // live `pick.value` unless a frozen target was asked for, so the default stays bit-identical.
        const next_value = if (opts.target_net != null) bootstrapValue(net, opts, pick.after) else pick.value;
        if (pending) |p| if (!opts.backward and !opts.death) trainAfter(net, &cache, p.after, p.reward + next_value, lr);
        steps[n % step_cap] = .{ .after = pick.after, .reward = pick.reward };
        n += 1;
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| if (!opts.backward and !opts.death) trainAfter(net, &cache, p.after, p.reward, lr);

    if (opts.backward or opts.death) {
        var i = n;
        while (i > n -| step_cap) {
            i -= 1;
            const s = steps[i % step_cap];
            const next = if (i + 1 < n) bootstrapValue(net, opts, steps[(i + 1) % step_cap].after) else 0;
            trainAfter(net, &cache, s.after, s.reward + next, lr);
        }
    }
    if (opts.replay_k > 0) {
        const lr_replay = lr * opts.replay_lr_scale;
        // One trace for the whole sweep: the TD(lambda) of the replayed tail. Unused (all zeros,
        // never read) when `sweep_lambda == 0`, which keeps the TD(0) sweep bit for bit.
        var sweep_trace: Net = std.mem.zeroes(Net);
        var i = n -| @min(opts.replay_k, step_cap);
        while (i < n) : (i += 1) {
            const s = steps[i % step_cap];
            const next = if (i + 1 < n) bootstrapValue(net, opts, steps[(i + 1) % step_cap].after) else 0;
            if (opts.sweep_lambda > 0) {
                _ = forward(net, s.after, &cache);
                trainTrace(net, &cache, s.reward + next, lr_replay, &sweep_trace, opts.sweep_lambda);
            } else trainAfter(net, &cache, s.after, s.reward + next, lr_replay);
        }
    }
    return .{ .score = state.score, .moves = @intCast(n) };
}

fn trainAfter(net: *Net, cache: *Cache, after: Board, target: f64, lr: f64) void {
    _ = forward(net, after, cache);
    train(net, cache, target, lr);
}

fn valueOf(net: *const Net, after: Board) f64 {
    var cache: Cache = .{};
    return forward(net, after, &cache);
}

/// V of the afterstate that closes a TD target. A frozen target net answers the bootstrap;
/// without one this is exactly the old `valueOf(net, after)`.
fn bootstrapValue(net: *const Net, opts: Options, after: Board) f64 {
    return valueOf(opts.target_net orelse net, after);
}

/// Global L2 norm the averaged symmetry gradient is clipped to. Measured averaged-gradient norms are
/// 0.04..0.26 on a fresh net but 4..300 late in training, so a unit-norm step is a no-op early and, with lr,
/// caps a late-game update at lr = 0.003 against a mean |w| of 0.44.
pub const sym_batch_clip: f64 = 1.0;

/// Batch symmetry: 8 gradients from one snapshot, averaged elementwise, one SGD step.
fn trainSymBatch(net: *Net, after: Board, target: f64, lr: f64) void {
    var grad = std.mem.zeroes(Net);
    var cache: Cache = .{};
    for (0..symmetry_count) |k| {
        const v = forward(net, transformBoard(k, after), &cache);
        const step = gradOf(net, &cache);
        for (flat(&grad), flat(@constCast(&step))) |*acc, x| acc.* += (v - target) * x;
    }
    for (flat(&grad)) |*acc| acc.* /= @as(f64, @floatFromInt(symmetry_count));
    var norm2: f64 = 0;
    for (flat(&grad)) |x| norm2 += x * x;
    const norm = @sqrt(norm2);
    const scale: f64 = if (norm > sym_batch_clip) sym_batch_clip / norm else 1;
    for (flat(net), flat(&grad)) |*w, x| w.* -= lr * scale * x;
}

/// Bound the TD target at `Options.clamp` (0 = untouched), so a diverging value cannot bootstrap
/// itself into `inf`.
fn clampTarget(target: f64, opts: Options) f64 {
    if (!(opts.clamp > 0)) return target;
    return std.math.clamp(target, -opts.clamp, opts.clamp);
}

/// The step size of one per-move TD sample: `lr` itself, or up to `lr * (1 + tail_alpha)` when the
/// board the target bootstrapped from is nearly full. Exact SGD loss weighting, see `Options.tail_alpha`.
fn tailWeight(opts: Options, board: Board, lr: f64) f64 {
    if (opts.tail_alpha <= 0) return lr;
    const empty: f64 = @floatFromInt(core.countEmpty(board));
    return lr * (1 + opts.tail_alpha * (15 - empty) / 15);
}

/// Empty-cell potential `Phi(board) = shape_c * countEmpty(board) / 15`: 0 on a full board,
/// `shape_c` on the empty board. It is dimensionless per empty cell, so `shape_c` sets the reward scale.
fn shapePotential(board: Board, shape_c: f64) f64 {
    const empty: f64 = @floatFromInt(core.countEmpty(board));
    return shape_c * empty / 15;
}

/// Shaped reward of one non-terminal TD transition. The `shape_c == 0` branch returns the raw
/// reward itself (not `raw + 0.0`), keeping `base`/`par` bit-identical.
fn shapedReward(p: Pick, next: Pick, opts: Options) f64 {
    if (opts.shape_c == 0) return p.reward;
    return p.reward + shapePotential(next.after, opts.shape_c) - shapePotential(p.after, opts.shape_c);
}

/// Shaped reward of the last transition: the terminal afterstate has no empty cell, so its
/// potential is 0 and only `-Phi(p.after)` remains.
fn shapedTerminalReward(p: Pick, opts: Options) f64 {
    if (opts.shape_c == 0) return p.reward;
    return p.reward - shapePotential(p.after, opts.shape_c);
}

test "tail weight scales only the nearly full boards (#232)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var open: Board = std.mem.zeroes(Board);
    open[0][0] = 1;
    try std.testing.expectEqual(@as(f64, 0.003), tailWeight(.{}, open, 0.003));
    try std.testing.expectEqual(@as(f64, 0.003), tailWeight(.{ .tail_alpha = 3 }, open, 0.003));
    var full: Board = std.mem.zeroes(Board);
    for (&full) |*row| {
        for (row) |*cell| cell.* = 1;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 0.012), tailWeight(.{ .tail_alpha = 3 }, full, 0.003), 1e-12);
    var half: Board = std.mem.zeroes(Board);
    for (0..4) |r| {
        for (0..4) |c| half[r][c] = if (c % 2 == 0) 1 else 0;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 0.0072), tailWeight(.{ .tail_alpha = 3 }, half, 0.003), 1e-12);
}

/// One per-move TD update on `after` with `target`: all 8 dihedral transforms when `opts.symmetry`,
/// and TD(lambda) through `trace` when `opts.lambda > 0` (otherwise the plain TD(0) `train` step).
fn trainStep(net: *Net, cache: *Cache, after: Board, target: f64, lr: f64, opts: Options, trace: *Net) void {
    if (opts.sym_batch) return trainSymBatch(net, after, target, lr);
    const transforms: usize = if (opts.symmetry) symmetry_count else 1;
    for (0..transforms) |k| {
        _ = forward(net, if (opts.symmetry) transformBoard(k, after) else after, cache);
        if (opts.popart) |popart| {
            const normalized_target = popart.startTrainingUpdate(net, cache, target);
            trainCapped(net, cache, normalized_target, lr, opts.step_clip);
            popart.finishTrainingUpdate(net);
        } else if (opts.lambda > 0) {
            trainTrace(net, cache, target, lr, trace, opts.lambda);
        } else {
            trainCapped(net, cache, target, lr, opts.step_clip);
        }
    }
}

/// Start of a `restart` game: a stored deep afterstate plus exactly one fresh spawn re-rolled from
/// `rng` (so the games from one dump differ, and one rng draw per game keeps them reproducible). Score 0:
/// only score deltas enter the target, and the stored board carries no score.
fn restartState(rng: std.Random, buffer: *Dump) GameState {
    var state = GameState{ .board = buffer.draw(rng), .score = 0 };
    _ = state.spawnRandomTile(rng);
    return state;
}

/// The dihedral transform drawn for one per-move TD update. `rtgt` draws once and gives the same
/// `k` to the input afterstate and to the target's next board, so the update is the TD update of the
/// rotated transition. `rsym` keeps the target canonical and rotates only the input.
const UpdateSym = struct {
    /// Transform of the stored afterstate (the trained input), or null for none.
    after: ?usize = null,
    /// Transform of the target's next board, or null for none. Only `rtgt` sets it.
    next: ?usize = null,
};

/// One draw per update for `rtgt`/`rsym`; `base` returns `.{}` and draws nothing, so its rng stream,
/// spawns and updates stay bit-identical. `distill_only` skips both `tdUpdate` calls below.
fn updateSym(rng: std.Random, opts: Options) UpdateSym {
    if (opts.rtgt) {
        const k = rng.uintLessThan(usize, symmetry_count);
        return .{ .after = k, .next = k };
    }
    if (opts.rsym) return .{ .after = rng.uintLessThan(usize, symmetry_count) };
    return .{};
}

fn symBoard(board: Board, k: ?usize) Board {
    return if (k) |index| transformBoard(index, board) else board;
}

fn symState(state: GameState, k: ?usize) GameState {
    return if (k) |index| .{ .board = transformBoard(index, state.board), .score = state.score } else state;
}

/// The expected-spawn bootstrap (#321): `bestAfterValue` of the pending afterstate averaged over every
/// spawn outcome the game could produce there, i.e. `V` of the greedy 1-ply reply to each. Cells are
/// uniform and the tile is 0.9/0.1, so the weights sum to 1 and a board with no empty cell has no spawn
/// left to average (0, the value of a finished game). Read-only: no spawn is drawn and no weight moves.
fn expectedSpawnValue(net: *const Net, after: Board) f64 {
    const empties = core.countEmpty(after);
    if (empties == 0) return 0;
    const vfn: Value = .{ .single = net };
    var total: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (after[r][c] != 0) continue;
            for ([_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } }) |spawn| {
                var child: GameState = .{ .board = after };
                child.board[r][c] = spawn.value;
                total += spawn.p * bestAfterValue(vfn, child);
            }
        }
    }
    return total / @as(f64, @floatFromInt(empties));
}

/// One non-terminal per-move TD update: draw the update's symmetry once, build the target
/// from the next board rotated by the same `k`, and train the rotated afterstate. Returns the target for
/// the diagnostics and the priority buffer. `spawned` is the post-spawn board the pending step led into
/// (this move has not been applied yet), the board the `search2` and `teacher` targets evaluate.
fn tdUpdate(net: *Net, cache: *Cache, p: Pick, next: Pick, spawned: GameState, lr: f64, rng: std.Random, opts: Options, trace: *Net) f64 {
    const sym = updateSym(rng, opts);
    const reward = shapedReward(p, next, opts);
    const bootstrap = if (opts.expected_spawn) expectedSpawnValue(net, p.after) else tdBootstrap(net, next, spawned, opts, sym.next);
    const target = clampTarget(reward + bootstrap, opts);
    trainStep(net, cache, symBoard(p.after, sym.after), target, tailWeight(opts, spawned.board, lr), opts, trace);
    return target;
}

/// The last update of a game: a terminal transition has no next board, so its target is
/// the (possibly shaped) reward alone and is never rotated; the input afterstate still gets the
/// update's one draw. With `expected_spawn` the target also prices the spawns that would have followed
/// `p.after` (#321), where the sampled run saw the one tile that ended the game: the drawn tile always
/// kills the board, but the spawns the expectation averages over are not all dead.
fn tdTerminalUpdate(net: *Net, cache: *Cache, p: Pick, spawned: GameState, lr: f64, rng: std.Random, opts: Options, trace: *Net) f64 {
    const sym = updateSym(rng, opts);
    const reward = shapedTerminalReward(p, opts);
    const target = if (opts.expected_spawn) reward + expectedSpawnValue(net, p.after) else reward;
    trainStep(net, cache, symBoard(p.after, sym.after), target, tailWeight(opts, spawned.board, lr), opts, trace);
    return target;
}

fn trainGamePerMove(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) GameResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    const pr = opts.priority;
    const vfn: Value = .{ .single = net };
    var state = if (opts.restart) restartState(rng, opts.dump.?) else GameState.gameStart(rng);
    var pending: ?Pick = null;
    var cache: Cache = .{};
    var dist_tick: u32 = 0;
    var moves: u32 = 0;
    var trace: Net = std.mem.zeroes(Net);
    while (true) {
        // See `Options.move_cap`: stop before the search, and before the pending step's update.
        if (opts.move_cap > 0 and moves >= opts.move_cap) {
            if (diag) |d| d.finish();
            return .{ .score = state.score, .moves = moves, .capped = true };
        }
        if (gameTimeLimitReached(start, opts.game_time_limit_ns)) {
            if (diag) |d| d.finish();
            return .{ .score = state.score, .moves = moves, .capped = true, .timed_out = true, .max_tile = core.maxTile(state.board) };
        }
        var pick = selfplayPick(vfn, state, opts.death, opts.selfplay, opts.teacher, opts.sym_leaf) orelse break;
        //: with probability `explore` play a uniformly random legal move, so the self-play walks
        // through boards the greedy policy would never visit. `0` (the default) draws nothing from the
        // rng, so every older seed still spawns the same tiles. Trained targets then include the random
        // step, which is why the variant loses badly: see `warm_open` for the target-free opening.
        if (opts.explore > 0 and rng.float(f64) < opts.explore) {
            if (randomPick(vfn, state, rng, opts.death)) |p| pick = p;
        }
        // opening diversity: the first `warm_open` moves are random and produce NO training
        // sample, so the games start from boards the greedy policy never reaches while every stored
        // target still comes from greedy play (random-move targets poisoned the value, see `explore`).
        if (moves < opts.warm_open) {
            if (randomPick(vfn, state, rng, opts.death)) |p| pick = p;
            moves += 1;
            pending = null;
            _ = state.move(pick.dir, rng);
            continue;
        }
        moves += 1;
        if (opts.dump_record) if (opts.dump) |buffer| buffer.offer(pick.after);
        if (pending) |p| {
            if (!opts.distill_only) {
                // `state` is still the post-spawn board the pending step led into (this move has not been
                // applied yet), which is the board the `search2` and `teacher` targets evaluate.
                const target = tdUpdate(net, &cache, p, pick, state, lr, rng, opts, &trace);
                if (diag) |d| d.push(p.after, @abs(target - p.value), pick.value - p.value);
                if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = pick.after, .terminal = false, .err = @abs(target - p.value) });
            }
        }
        pending = pick;
        // `ft2` mixes one deep update into every self-play step: 50% self-play / 50% deep afterstates.
        if (opts.deep_mix) if (opts.dump) |buffer| if (buffer.len > 0) {
            const every = @max(opts.deep_every, 1);
            if (moves % every == 0) deepStep(net, rng, buffer, lr, &cache, opts.deep_mode);
        };
        //: the teacher regression only every `distill_every` steps and at its own rate, so the
        // shallow half of the game keeps the parameters it learned.
        if (opts.distill) |buffer| if (buffer.len > 0) {
            dist_tick += 1;
            if (dist_tick % @max(opts.distill_every, 1) == 0) {
                distStep(net, rng, buffer, if (opts.distill_lr > 0) opts.distill_lr else lr, &cache);
            }
        };
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| {
        if (!opts.distill_only) {
            const target = tdTerminalUpdate(net, &cache, p, state, lr, rng, opts, &trace);
            if (diag) |d| d.push(p.after, @abs(target - p.value), -p.value);
            if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = p.after, .terminal = true, .err = @abs(target - p.value) });
        }
    }
    if (pr) |b| for (0..b.m) |_| {
        const e = &b.items[
            if (opts.midmix)
                b.sampleMidmix(rng)
            else if (opts.midgame)
                b.sampleMidgame(rng)
            else
                b.sample(rng)
        ];
        const target = e.reward + if (e.terminal) 0 else bootstrapValue(net, opts, e.next);
        _ = forward(net, e.after, &cache);
        e.err = @abs(target - cache.out);
        train(net, &cache, target, lr);
    };
    if (diag) |d| d.finish();
    return .{ .score = state.score, .moves = moves };
}

/// One self-play move of the per-move trainer. `greedy1` is the historical 1-ply argmax;
/// `deep2` plays the eval's 2-ply search; a `teacher` mode plays that mode's search (see
/// `pickTeacherValue`). None of them draws from the rng (the search enumerates spawns), so all policies
/// spawn the same tiles in the same order for the same seed and differ only in the move they pick.
fn selfplayPick(vfn: Value, state: GameState, death: bool, selfplay: SelfPlay, teacher: ?Mode, sym_leaf: bool) ?Pick {
    if (teacher) |mode| return pickTeacherValue(vfn, state, mode, sharedValueCache(), sym_leaf);
    return switch (selfplay) {
        .greedy1 => pickMoveValue(vfn, state, death),
        .deep2 => pickSearch2Value(vfn, state, sharedValueCache(), sym_leaf),
    };
}

/// Value of a post-spawn board under a teacher mode: `Search.value(state, ply, 1)` at the mode's
/// phase row, i.e. `ply` own moves with the eval's `V` at the leaf. For `.deep` (ply 2, full width)
/// that is the tree `search2Value` bootstraps (`t2`); for `.deep3` (ply 3, its `deep3_cutoff`) it is
/// the tree `Mode.deep` ranks its moves with, so the target is what the 2-ply play side already
/// believes the position is worth, one own move deeper than `t2` and cheap enough to train on.
fn teacherValue(vfn: Value, state: GameState, mode: Mode, sym_leaf: bool) f64 {
    // a fresh table per label: the previous one was filled with the weights as they stood a few steps
    // ago, and `getValue` cannot tell
    sharedValueCache().clear();
    const depth = phaseDepth(mode, state.board);
    var search: Search = .{
        .vfn = vfn,
        .cache = sharedValueCache(),
        .cutoff = depth.cutoff,
        .mass_cutoff = depth.mass_cutoff,
        .ply = depth.ply,
        .sym_leaf = sym_leaf,
    };
    return search.value(state, depth.ply, 1);
}

/// Bootstrap of one per-move TD target: the value of what follows the step's reward.
/// `td1` keeps the historical V of the afterstate the next move reached; `search2` bootstraps
/// `spawned`, the post-spawn board the step led into, with `search2_ply - 1` searched rewards and V of
/// the afterstate one move past the last of them (see `search2Value`). With a `deep2` policy `next.value`
/// is already a search value, so `td1` reads V of the afterstate directly then. A teacher
/// replaces the bootstrap entirely: its own search value of `spawned`, and `target` is ignored.
/// `next_sym` is the `rtgt` transform of the board being valued: `next.after` for `td1`, the
/// post-spawn `spawned` for `search2` and a teacher. When set, `td1` prices the rotated board directly
/// (a frozen `target_net` included) instead of reusing `next.value`, which is canonical.
fn tdBootstrap(net: *const Net, next: Pick, spawned: GameState, opts: Options, next_sym: ?usize) f64 {
    if (opts.teacher) |mode| return teacherValue(.{ .single = net }, symState(spawned, next_sym), mode, opts.sym_leaf);
    return switch (opts.target) {
        // A frozen target prices the next afterstate even when the live policy is `deep2`: `next.value`
        // would be the live net's search value, i.e. the moving target this variant removes.
        .td1 => if (next_sym) |k|
            bootstrapValue(net, opts, transformBoard(k, next.after))
        else if (opts.target_net != null)
            bootstrapValue(net, opts, next.after)
        else if (opts.selfplay == .deep2)
            valueOf(net, next.after)
        else
            next.value,
        .search2 => search2Value(.{ .single = net }, symState(spawned, next_sym), sharedValueCache(), opts.sym_leaf),
    };
}

/// Online n-step TD on the same 1-ply self-play policy as `trainGamePerMove`: each step enters
/// the window with V of its afterstate, and the step leaving the window is trained on its n rewards plus
/// that value. The last min(n, game length) steps are flushed at game over with the rewards alone.
fn trainGameNStep(net: *Net, rng: std.Random, lr: f64, n: usize, move_cap: u32, time_limit_ns: ?u64) GameResult {
    const start = gameStartTime(time_limit_ns);
    var window = NStepWindow.init(n);
    var state = GameState.gameStart(rng);
    var cache: Cache = .{};
    var moves: u32 = 0;
    while (true) {
        // A capped game is not terminal, so its window is not flushed as one either.
        if (move_cap > 0 and moves >= move_cap) return .{ .score = state.score, .moves = moves, .capped = true };
        if (gameTimeLimitReached(start, time_limit_ns)) return .{ .score = state.score, .moves = moves, .capped = true, .timed_out = true, .max_tile = core.maxTile(state.board) };
        const pick = pickMove(net, state, false) orelse break;
        moves += 1;
        if (window.push(.{ .after = pick.after, .reward = pick.reward }, pick.value)) |done|
            trainAfter(net, &cache, done.after, done.target, lr);
        _ = state.move(pick.dir, rng);
    }
    while (window.nextFinish()) |done| trainAfter(net, &cache, done.after, done.target, lr);
    return .{ .score = state.score, .moves = moves };
}

/// MS-TD self-play game over the staged value: the same 1-ply policy as `trainGamePerMove`,
/// one TD(0) step per move (gamma = 1), and each step trains only the net of the afterstate it reached.
pub fn trainGameStaged(staged: *Staged, rng: std.Random, lr: f64) GameResult {
    return trainGameStagedOpts(staged, rng, lr, .{});
}

pub fn trainGameStagedOpts(staged: *Staged, rng: std.Random, lr: f64, opts: Options) GameResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    const value: Value = .{ .staged = staged };
    var state = GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    while (true) {
        // Same cap on the MS-TD path; the staged weights are also finite-safe.
        if (moves >= default_move_cap) return .{ .score = state.score, .moves = moves, .capped = true };
        if (gameTimeLimitReached(start, opts.game_time_limit_ns)) return .{ .score = state.score, .moves = moves, .capped = true, .timed_out = true, .max_tile = core.maxTile(state.board) };
        const pick = pickMoveValue(value, state, false) orelse break;
        moves += 1;
        if (pending) |p| _ = staged.train(p.after, pick.after, p.reward, lr);
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| _ = staged.train(p.after, null, p.reward, lr);
    return .{ .score = state.score, .moves = moves };
}

/// Games per parallel training batch: min(cpu count, max_threads), at least 1.
pub fn trainBatchSize() usize {
    const cpus = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(cpus, parallel.max_threads));
}

/// Seed for training game `index`; depends only on the game index, not on thread count or batch layout.
pub fn trainingSeed(seed: u64, index: usize) u64 {
    return seed +% @as(u64, @intCast(index));
}

const BatchGameCtx = struct { net: Net, seed: u64, first_game: usize, lr: f64, want_diag: bool, opts: Options };

const BatchGame = struct {
    net: Net,
    score: u32,
    moves: u32,
    capped: bool = false,
    timed_out: bool = false,
    max_tile: core.Tile = 0,
    diag: Diag = .{},
};

fn runBatchGame(ctx: BatchGameCtx, i: usize) BatchGame {
    const game = ctx.first_game + i;
    var result: BatchGame = .{ .net = ctx.net, .score = 0, .moves = 0 };
    var rng = std.Random.DefaultPrng.init(trainingSeed(ctx.seed, game));
    const trained = trainGameOptsUnchecked(&result.net, rng.random(), ctx.lr, if (ctx.want_diag) &result.diag else null, ctx.opts);
    result.score = trained.score;
    result.moves = trained.moves;
    result.capped = trained.capped;
    result.timed_out = trained.timed_out;
    result.max_tile = trained.max_tile;
    return result;
}

/// What a batch reports back to the CLI: the score of the last game (the historical return of
/// `trainBatch`/`trainBatchOpts`) plus the moves the whole batch played. The moves are what the serial
/// loop adds to `total_moves`; without them the `par` path printed `0 moves`, which read as
/// "par trains nothing".
pub const BatchOutcome = struct {
    score: u32,
    moves: u64,
    /// Index (within this batch) and move count of the first game that hit `Options.move_cap`;
    /// null when every game ended normally. One pathological game stalls the whole batch, so the CLI
    /// must stop on this instead of waiting for the other games.
    capped_game: ?usize = null,
    capped_moves: u32 = 0,
    /// (#336) The first unfinished timed-out game and its diagnostic partial result. Its whole batch
    /// is rejected before the merge, so no gradient from that batch reaches the caller's net.
    timed_out_game: ?usize = null,
    timed_out_score: u32 = 0,
    timed_out_moves: u32 = 0,
    timed_out_max_tile: core.Tile = 0,
};

fn mergeDiag(dst: *Diag, src: *const Diag) void {
    for (0..3) |e| for (0..3) |k| {
        dst.stats[e][k].count += src.stats[e][k].count;
        dst.stats[e][k].abs_err += src.stats[e][k].abs_err;
        dst.stats[e][k].drop += src.stats[e][k].drop;
    };
    for (0..tile_buckets) |t| {
        dst.tiles[t].count += src.tiles[t].count;
        dst.tiles[t].abs_err += src.tiles[t].abs_err;
    }
}

/// Runs `games` (<= max_threads) self-play games in parallel from the same starting net, then sets
/// `net` to net_start + (1/games) * sum_i (net_after_i - net_start), summed in game order.
/// Each game i uses `trainingSeed(seed, first_game + i)`, so the result is the same for any thread count.
pub fn trainBatch(net: *Net, seed: u64, first_game: usize, games: usize, lr: f64, diag: ?*Diag) BatchOutcome {
    return trainBatchOpts(net, seed, first_game, games, lr, diag, .{}) catch unreachable;
}

fn validateBatchOptions(opts: Options) OptionsError!void {
    try validateOptions(opts);
    if (opts.popart != null) return error.PopArtRequiresPlainTd1;
}

/// `trainBatch` with options: every game trains with `trainGameOpts(.., opts)`, so any stateless
/// variant (`backward`, `replay`, `sym`, `symlam`, `nstep`) batches the same way `base` does.
/// Refuses a shaped run that the per-move path cannot implement. Refuses PopArt because
/// its EMA state is serial-only.
pub fn trainBatchOpts(net: *Net, seed: u64, first_game: usize, games: usize, lr: f64, diag: ?*Diag, opts: Options) OptionsError!BatchOutcome {
    try validateBatchOptions(opts);
    std.debug.assert(games >= 1 and games <= parallel.max_threads);
    // the priority buffer is cross-game state shared by pointer: batching it would be a data race
    std.debug.assert(opts.priority == null);
    var units: [parallel.max_threads]BatchGame = undefined;
    // the per-game nets train with plain SGD; Adam acts once on the merged gradient below
    var thread_opts = opts;
    thread_opts.adam = null;
    // a batch applies one step: the per-game nets accumulate the gradient, and the merge below
    // caps that single step. Capping inside the games would cap 8 sub-steps instead of one.
    thread_opts.step_clip = 0;
    parallel.map(BatchGame, units[0..games], BatchGameCtx{
        .net = net.*,
        .seed = seed,
        .first_game = first_game,
        .lr = lr,
        .want_diag = diag != null,
        .opts = thread_opts,
    }, runBatchGame);

    var moves: u64 = 0;
    var capped_game: ?usize = null;
    var capped_moves: u32 = 0;
    var timed_out_game: ?usize = null;
    var timed_out_score: u32 = 0;
    var timed_out_moves: u32 = 0;
    var timed_out_max_tile: core.Tile = 0;
    for (units[0..games], 0..) |u, i| {
        moves += u.moves;
        if (u.timed_out and timed_out_game == null) {
            timed_out_game = i;
            timed_out_score = u.score;
            timed_out_moves = u.moves;
            timed_out_max_tile = u.max_tile;
        }
        if (u.capped and !u.timed_out and capped_game == null) {
            capped_game = i;
            capped_moves = u.moves;
        }
    }
    const outcome: BatchOutcome = .{
        .score = units[games - 1].score,
        .moves = moves,
        .capped_game = capped_game,
        .capped_moves = capped_moves,
        .timed_out_game = timed_out_game,
        .timed_out_score = timed_out_score,
        .timed_out_moves = timed_out_moves,
        .timed_out_max_tile = timed_out_max_tile,
    };
    if (timed_out_game != null) return outcome;

    const start = net.*;
    const start_w = std.mem.bytesAsSlice(f64, std.mem.asBytes(&start));
    const out_w = std.mem.bytesAsSlice(f64, std.mem.asBytes(net));
    for (out_w) |*w| w.* = 0;
    for (units[0..games]) |*u| {
        const after = std.mem.bytesAsSlice(f64, std.mem.asBytes(&u.net));
        for (out_w, after, start_w) |*o, v, s| o.* += v - s;
    }
    const n: f64 = @floatFromInt(games);
    const scale = opts.batch_scale / n;
    for (out_w, start_w) |*o, s| o.* = s + o.* * scale;
    if (opts.adam) |state| {
        // The merge above is `-lr * (summed batch gradient)`, so dividing by `lr` gives Adam the gradient;
        // `start` is the net the games played with, and the optimizer step is applied to it.
        var grad: Net = undefined;
        const inv = -1 / lr;
        for (flat(&grad), flatConst(net), start_w) |*g, merged, s| g.* = (merged - s) * inv;
        net.* = start;
        applyAdam(net, &grad, state, opts.adam_lr);
    } else if (opts.grad_clip > 0 or opts.step_clip > 0) {
        clipMergedDelta(out_w, start_w, opts);
    }
    if (diag) |d| for (units[0..games]) |u| mergeDiag(d, &u.diag);
    return outcome;
}

/// Bound the single step `trainBatchOpts` applies: `grad_clip` scales the whole delta down to
/// that L2 norm (source and target of the step, so the comparison is on `dw` alone), then `step_clip`
/// caps every element. Both 0 (the default) leaves the merged delta untouched, bit for bit.
fn clipMergedDelta(merged: []f64, start: []const f64, opts: Options) void {
    if (opts.grad_clip > 0) {
        var norm2: f64 = 0;
        for (merged, start) |m, s| norm2 += (m - s) * (m - s);
        const norm = @sqrt(norm2);
        const scale: f64 = if (norm > opts.grad_clip) opts.grad_clip / norm else 1;
        for (merged, start) |*m, s| m.* = s + (m.* - s) * scale;
    }
    if (opts.step_clip > 0) {
        for (merged, start) |*m, s| m.* = s + std.math.clamp(m.* - s, -opts.step_clip, opts.step_clip);
    }
}

/// Weight file: 8-byte magic, u32 version, u32 endianness marker (all little-endian), then the f64 weights as little-endian.
pub const file_magic = "2048-NN\x00";
pub const file_version: u32 = 1;
const endian_marker: u32 = 0x01020304;
const header_len = 16;

/// The caller's own error names for the three header words, so `error.BadWeightMagic` and
/// `error.BadQuantMagic` stay distinct while the check itself is written once.
const HeaderErrors = struct { magic: anyerror, version: anyerror, endian: anyerror };

/// `bytes` starts with the 8-byte `magic` or else `err` — all `Dump` and `Teachers` need.
fn checkMagic(bytes: []const u8, magic: []const u8, err: anyerror) !void {
    if (!std.mem.eql(u8, bytes[0..8], magic)) return err;
}

/// The header every weight file shares (little endian): 8-byte magic, u32 version, u32 endianness
/// marker. `errs` names the three failures so each format keeps its own error values.
fn checkVersionedHeader(bytes: []const u8, magic: []const u8, version: u32, errs: HeaderErrors) !void {
    try checkMagic(bytes, magic, errs.magic);
    if (std.mem.readInt(u32, bytes[8..12], .little) != version) return errs.version;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return errs.endian;
}

pub const file_len = header_len + @sizeOf(Net);

/// Staged weight file: the same header with its own 8-byte magic and `stage_count` weight blocks.
pub const staged_file_magic = "2048-MS\x00";
pub const staged_file_len = header_len + @sizeOf(Net) * stage_count;

fn writeWeights(w: *blob.Writer, net: *const Net) void {
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(net))) |weight| w.writeF64(weight);
}

fn readWeights(r: *blob.Reader) !Net {
    var net: Net = undefined;
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&net))) |*weight| weight.* = try r.readF64();
    return net;
}

pub fn encode(net: *const Net) [file_len]u8 {
    var out: [file_len]u8 = undefined;
    @memcpy(out[0..8], file_magic);
    std.mem.writeInt(u32, out[8..12], file_version, .little);
    std.mem.writeInt(u32, out[12..16], endian_marker, .little);
    var w = blob.Writer.init(out[header_len..]);
    writeWeights(&w, net);
    return out;
}

/// Diagnostic: how many times `parse` ran on this thread (87 tests assert weights are
/// parsed once per thread, not once per move).
pub threadlocal var parse_count: usize = 0;

pub fn parse(bytes: []const u8) !Net {
    parse_count += 1;
    if (bytes.len != file_len) return error.WrongWeightFileSize;
    try checkVersionedHeader(bytes, file_magic, file_version, .{
        .magic = error.BadWeightMagic,
        .version = error.UnsupportedWeightVersion,
        .endian = error.WrongWeightEndianness,
    });
    var r = blob.Reader.init(bytes[header_len..]);
    return readWeights(&r);
}

pub fn save(net: *const Net, io: std.Io, path: []const u8) !void {
    const bytes = encode(net);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = &bytes });
}

pub fn load(net: *Net, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(file_len + 1));
    defer gpa.free(bytes);
    net.* = try parse(bytes);
}

fn flat(net: *Net) []f64 {
    return std.mem.bytesAsSlice(f64, std.mem.asBytes(net)[0..@sizeOf(Net)]);
}

fn flatConst(net: *const Net) []const f64 {
    return std.mem.bytesAsSlice(f64, std.mem.asBytes(net)[0..@sizeOf(Net)]);
}

/// TD with function approximation can diverge: on the 128/32 net the weights reached 1e86 (and 6,334 NaN)
/// somewhere between 100,000 and 300,000 games on training seed 2. A half-day `par` run that ends in
/// a NaN net wastes the whole run and, worse, overwrites the last good checkpoint, so training checks this
/// every few games and stops instead of saving (`main` reports `diverged at N games`).
pub fn weightsFinite(net: *const Net) bool {
    for (flatConst(net)) |w| {
        if (!std.math.isFinite(w) or @abs(w) > 1e6) return false;
    }
    return true;
}

// ---- quantized weights ----
//
// A second file format next to the f32 one, for shipping: the same 18,561 weights as int16 with one
// step per block (`w ≈ q * step`), 148,504 B -> 37,170 B. The f32 format is untouched and stays the
// default evaluation path; nothing here is reachable from `parse`/`forward`.

/// Weight blocks that share one step, in `Net` field order. `b3` gets its own block because it is an
/// order of magnitude larger than `w3` and would otherwise cost `w3` 13x its resolution.
pub const quant_block_count = if (feature_count > 0) 7 else 6;
/// Activations that cross a layer boundary, each with its own step: hidden layer 1 and 2.
pub const quant_activation_count = 2;

pub const quant_magic = "2048-NQ1";
pub const quant_version: u32 = 1;
/// magic + version + endian marker + one f32 per block and per activation step.
pub const quant_header_len = 16 + (quant_block_count + quant_activation_count) * @sizeOf(f32);
/// `Net` holds 18,561 f64 weights (256x64 + 64 + 32x64 + 32 + 32 + 1).
pub const weight_count = @sizeOf(Net) / @sizeOf(f64);
pub const quant_file_len = quant_header_len + weight_count * @sizeOf(i16);

const Block = struct { offset: usize, len: usize };

/// A block of `Net` as element offsets into its flat weight array. `QPayload` mirrors the fields in the
/// same order, so the same offsets index the i16 payload.
fn blockOf(comptime field: []const u8) Block {
    return .{
        .offset = @offsetOf(Net, field) / @sizeOf(f64),
        .len = @sizeOf(@FieldType(Net, field)) / @sizeOf(f64),
    };
}

pub const quant_blocks = if (feature_count > 0)
    [7]Block{ blockOf("w1"), blockOf("b1"), blockOf("w2"), blockOf("b2"), blockOf("w3"), blockOf("b3"), blockOf("wf") }
else
    [6]Block{ blockOf("w1"), blockOf("b1"), blockOf("w2"), blockOf("b2"), blockOf("w3"), blockOf("b3") };

/// The i16 mirror of `Net`, field for field. `extern` keeps the layout raw, so the payload in the file
/// is these bytes little-endian (2 B per weight instead of 8).
pub const QPayload = extern struct {
    w1: [input_count][h1_size]i16,
    b1: [h1_size]i16,
    w2: [h2_size][h1_size]i16,
    b2: [h2_size]i16,
    w3: [h2_size]i16,
    b3: i16,
    /// zero-length unless the build sets `-Dfeat`, so the shipped layout is unchanged.
    wf: [feature_count][h1_size]i16,
};

/// int16 weights plus the steps that turn them back into values: `w ≈ q * steps[block]`,
/// `activation ≈ q * a_steps[layer]`.
pub const Quantized = struct {
    q: QPayload = std.mem.zeroes(QPayload),
    steps: [quant_block_count]f64 = @splat(0),
    a_steps: [quant_activation_count]f64 = @splat(0),
};

/// Step for one block: `max|w| / 32767`, so the block's largest weight lands on the i16 limit. 0 for an
/// all-zero block, which then reads back as zeros.
fn blockStep(values: []const f64) f64 {
    var max: f64 = 0;
    for (values) |w| max = @max(max, @abs(w));
    return max / 32767;
}

/// `q = round(w / step)`, clamped to +-32767 (the step already puts `max|w|` on the limit, so the clamp
/// only catches the `max|w|/32767 * 32767 < max|w|` edge of the float division).
fn quantizeBlock(values: []const f64, step: f64, out: []i16) void {
    if (step == 0) return @memset(out, 0);
    for (values, 0..) |w, i| out[i] = @intFromFloat(@round(std.math.clamp(w / step, -32767, 32767)));
}

/// Worst-case activation of hidden layer 1 over every board: a board lights up 16 rows, so per hidden
/// unit the bound is `b1[i]` plus the 16 largest rows of column `i`. (Slightly loose: the 16 largest
/// rows may name the same cell twice, which no board does.) Deterministic and computed from the
/// weights alone, unlike a bound sampled from boards.
fn activationBound1(net: *const Net) f64 {
    var bound: f64 = 0;
    for (0..h1_size) |i| {
        var positive: [input_count]f64 = undefined;
        var n: usize = 0;
        for (0..input_count) |row| {
            const w = net.w1[row][i];
            if (w > 0) {
                positive[n] = w;
                n += 1;
            }
        }
        var sum = @max(net.b1[i], 0);
        for (0..@min(core.BoardCellCount, n)) |_| {
            var best: usize = 0;
            for (0..n) |k| if (positive[k] > positive[best]) {
                best = k;
            };
            sum += positive[best];
            positive[best] = -1; // removed: every remaining entry came in positive
        }
        bound = @max(bound, sum);
    }
    return bound;
}

/// Worst-case activation of hidden layer 2: `b2[j]` plus every positive `w2[j][i]` at the layer-1 bound.
/// Loose by construction (all 64 hidden units cannot peak at once), but ~1e-2 absolute on the value —
/// see `docs/solver-history.md`.
fn activationBound2(net: *const Net, a1_bound: f64) f64 {
    var bound: f64 = 0;
    for (0..h2_size) |j| {
        var sum = net.b2[j];
        for (0..h1_size) |i| sum += @max(net.w2[j][i], 0) * a1_bound;
        bound = @max(bound, sum);
    }
    return bound;
}

/// `Net` -> `Quantized`: every block gets `step = max|w| / 32767` and `q = round(w / step)`; the
/// hidden activations get the worst-case bound of the layer that produces them as their step, so they
/// fit i16 without clamping a reachable value.
pub fn quantize(net: *const Net) Quantized {
    var q = Quantized{};
    const values = flatConst(net);
    const qvalues = std.mem.bytesAsSlice(i16, std.mem.asBytes(&q.q));
    for (quant_blocks, 0..) |block, b| {
        const block_values = values[block.offset..][0..block.len];
        q.steps[b] = blockStep(block_values);
        quantizeBlock(block_values, q.steps[b], qvalues[block.offset..][0..block.len]);
    }
    const a1_bound = activationBound1(net);
    q.a_steps[0] = a1_bound / 32767;
    q.a_steps[1] = activationBound2(net, a1_bound) / 32767;
    return q;
}

/// Write the quantized weights back into an f32 net, so the rounding can be A/B'd with the same
/// evaluation code (`w = q * step` loses at most half a step per block).
pub fn dequantize(q: *const Quantized, net: *Net) void {
    const qvalues = std.mem.bytesAsSlice(i16, std.mem.asBytes(&q.q));
    const values = flat(net);
    for (quant_blocks, 0..) |block, b| {
        for (block.offset..block.offset + block.len) |i| {
            values[i] = @as(f64, @floatFromInt(qvalues[i])) * q.steps[b];
        }
    }
}

/// File layout (all little endian): 8-byte magic `2048-NQ1`, u32 version, u32 endianness
/// marker, one f32 step per block (`w1 b1 w2 b2 w3 b3`), one f32 step per activation (`a1 a2`), then
/// the 18,561 i16 weights in `Net` field order.
pub fn encodeQuantized(q: *const Quantized) [quant_file_len]u8 {
    var out: [quant_file_len]u8 = undefined;
    var w = blob.Writer.init(&out);
    blob.writeHeader(&w, quant_magic, quant_version);
    for (q.steps) |step| w.writeF32(@floatCast(step));
    for (q.a_steps) |step| w.writeF32(@floatCast(step));
    for (std.mem.bytesAsSlice(i16, std.mem.asBytes(&q.q))) |value| w.writeI16(value);
    return out;
}

pub fn parseQuantized(bytes: []const u8) !Quantized {
    if (bytes.len != quant_file_len) return error.WrongQuantFileSize;
    var r = blob.Reader.init(bytes);
    try blob.readHeader(&r, quant_magic, quant_version, .{
        .magic = error.BadQuantMagic,
        .version = error.UnsupportedQuantVersion,
        .endian = error.WrongQuantEndianness,
    });
    var q = Quantized{};
    for (&q.steps) |*step| step.* = @floatCast(try r.readF32());
    for (&q.a_steps) |*step| step.* = @floatCast(try r.readF32());
    for (std.mem.bytesAsSlice(i16, std.mem.asBytes(&q.q))) |*value| value.* = try r.readI16();
    return q;
}

pub fn saveQuantized(q: *const Quantized, io: std.Io, path: []const u8) !void {
    const bytes = encodeQuantized(q);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = &bytes });
}

pub fn loadQuantized(q: *Quantized, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(quant_file_len + 1));
    defer gpa.free(bytes);
    q.* = try parseQuantized(bytes);
}

/// ReLU followed by re-quantization to i16: `round(max(z, 0) / step)`, clamped to the i16 range.
fn quantizeActivation(z: f64, step: f64) i16 {
    if (!(z > 0) or step == 0) return 0; // `!(z > 0)` also catches NaN; a zero step is an all-zero net
    return @intFromFloat(@min(@round(z / step), 32767));
}

/// Afterstate value from the quantized net. Same arithmetic as `forward`, with int16 weights:
/// each hidden layer multiplies i16 x i16 into i32 and sums in i64 — the steps are ~1e-4, so the
/// accumulated integers reach 1e9 and an i32 sum would overflow — then the layer's step is applied
/// once, at the activation, and the activation is re-quantized with its own step for the next layer.
pub fn forwardQuantized(q: *const Quantized, board: Board) f64 {
    const s = q.steps;
    var h1: [h1_size]i16 = undefined;
    for (0..h1_size) |i| {
        var acc: i32 = 0;
        for (0..core.BoardCellCount) |cell| {
            const row = cell * levels + levelOf(board[cell / core.BoardSize][cell % core.BoardSize]);
            acc += q.q.w1[row][i];
        }
        const z = @as(f64, @floatFromInt(acc)) * s[0] + @as(f64, @floatFromInt(q.q.b1[i])) * s[1];
        h1[i] = quantizeActivation(z, q.a_steps[0]);
    }
    var h2: [h2_size]i16 = undefined;
    for (0..h2_size) |j| {
        var acc: i64 = 0;
        for (0..h1_size) |i| acc += @as(i32, q.q.w2[j][i]) * @as(i32, h1[i]);
        const z = @as(f64, @floatFromInt(acc)) * (q.a_steps[0] * s[2]) +
            @as(f64, @floatFromInt(q.q.b2[j])) * s[3];
        h2[j] = quantizeActivation(z, q.a_steps[1]);
    }
    var acc: i64 = 0;
    for (0..h2_size) |j| acc += @as(i32, q.q.w3[j]) * @as(i32, h2[j]);
    return @as(f64, @floatFromInt(acc)) * (q.a_steps[1] * s[4]) +
        @as(f64, @floatFromInt(q.q.b3)) * s[5];
}

/// f32 copy of the same weights (#348). The point is traffic: one leaf reads ~24 KiB instead of
/// ~48 KiB, and wasm gets an f32x4 first-layer accumulate. Training stays f64 (`Net`); this is a
/// shipping artefact, written by `zig build nn-f32`.
pub const Net32 = extern struct {
    w1: [input_count][h1_size]f32,
    b1: [h1_size]f32,
    w2: [h2_size][h1_size]f32,
    b2: [h2_size]f32,
    w3: [h2_size]f32,
    b3: f32,
    wf: [feature_count][h1_size]f32,
};

/// Narrow the trained f64 weights once, at conversion time.
pub fn toF32(net: *const Net) Net32 {
    var out: Net32 = std.mem.zeroes(Net32);
    for (0..input_count) |r| for (0..h1_size) |i| {
        out.w1[r][i] = @floatCast(net.w1[r][i]);
    };
    for (0..h1_size) |i| out.b1[i] = @floatCast(net.b1[i]);
    for (0..h2_size) |j| {
        for (0..h1_size) |i| out.w2[j][i] = @floatCast(net.w2[j][i]);
        out.b2[j] = @floatCast(net.b2[j]);
        out.w3[j] = @floatCast(net.w3[j]);
    }
    out.b3 = @floatCast(net.b3);
    for (0..feature_count) |k| for (0..h1_size) |i| {
        out.wf[k][i] = @floatCast(net.wf[k][i]);
    };
    return out;
}

/// Each lane adds its own row slice, so the four f32 lanes stay independent (no reassociation
/// across lanes, and the tail is scalar).
fn accumulateRowsSimdF32(dst: []f32, src: []const f32) void {
    const V = @Vector(4, f32);
    var i: usize = 0;
    while (i + 4 <= dst.len) : (i += 4) {
        const a: V = dst[i..][0..4].*;
        const b: V = src[i..][0..4].*;
        dst[i..][0..4].* = a + b;
    }
    while (i < dst.len) : (i += 1) dst[i] += src[i];
}

/// f32x4 dot product: the vector lanes keep their own partial sums, the tail is scalar.
fn dotSimdF32(a: []const f32, b: []const f32) f32 {
    const V = @Vector(4, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 4 <= a.len) : (i += 4) {
        const x: V = a[i..][0..4].*;
        const y: V = b[i..][0..4].*;
        acc += x * y;
    }
    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// Afterstate value from the f32 net: the same arithmetic as `forward`, in f32. `-Dnn-simd` picks
/// the f32x4 accumulate for the first layer and the f32x4 dot for the second.
pub fn forwardF32(net: *const Net32, board: Board) f64 {
    return forwardF32Impl(net, board, width.simd);
}

fn forwardF32Impl(net: *const Net32, board: Board, comptime simd: bool) f64 {
    var z1: [h1_size]f32 = net.b1;
    for (0..core.BoardCellCount) |cell| {
        const row = cell * levels + levelOf(board[cell / core.BoardSize][cell % core.BoardSize]);
        if (comptime simd) {
            accumulateRowsSimdF32(&z1, &net.w1[row]);
        } else {
            for (0..h1_size) |i| z1[i] += net.w1[row][i];
        }
    }
    if (comptime feature_count > 0) {
        const feat = features(board);
        for (feat, 0..) |f, k| {
            if (f == 0) continue;
            const fk: f32 = @floatCast(f);
            for (0..h1_size) |i| z1[i] += net.wf[k][i] * fk;
        }
    }
    var a1: [h1_size]f32 = undefined;
    for (0..h1_size) |i| a1[i] = @max(z1[i], 0);

    var out: f32 = net.b3;
    for (0..h2_size) |j| {
        var z: f32 = net.b2[j];
        if (comptime simd) {
            z += dotSimdF32(&net.w2[j], &a1);
        } else {
            for (0..h1_size) |i| z += net.w2[j][i] * a1[i];
        }
        out += net.w3[j] * @max(z, 0);
    }
    return out;
}

/// f32 weight file (#348): the same 16-byte header with its own magic, then `Net32`'s bytes.
pub const f32_file_magic = "2048-NF\x00";
pub const f32_file_len = header_len + @sizeOf(Net32);

pub fn encodeF32(net: *const Net32) [f32_file_len]u8 {
    var out: [f32_file_len]u8 = undefined;
    @memcpy(out[0..8], f32_file_magic);
    std.mem.writeInt(u32, out[8..12], file_version, .little);
    std.mem.writeInt(u32, out[12..16], endian_marker, .little);
    var w = blob.Writer.init(out[header_len..]);
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(net))) |weight| w.writeF32(weight);
    return out;
}

pub fn decodeF32(bytes: []const u8) !Net32 {
    if (bytes.len != f32_file_len) return error.WrongWeightFileSize;
    try checkVersionedHeader(bytes, f32_file_magic, file_version, .{
        .magic = error.BadWeightMagic,
        .version = error.UnsupportedWeightVersion,
        .endian = error.WrongWeightEndianness,
    });
    var net: Net32 = undefined;
    var r = blob.Reader.init(bytes[header_len..]);
    for (std.mem.bytesAsSlice(f32, std.mem.asBytes(&net))) |*weight| weight.* = try r.readF32();
    return net;
}

pub fn saveF32(net: *const Net32, io: std.Io, path: []const u8) !void {
    const bytes = encodeF32(net);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = &bytes });
}

pub fn loadF32(net: *Net32, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(f32_file_len + 1));
    defer gpa.free(bytes);
    net.* = try decodeF32(bytes);
}

fn lossOf(net: *const Net, board: Board, target: f64) f64 {
    var cache: Cache = .{};
    const d = forward(net, board, &cache) - target;
    return 0.5 * d * d;
}

const sample_board: Board = .{
    .{ 2, 0, 0, 0 },
    .{ 0, 4, 0, 0 },
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 8 },
};

test "parallel nn games match the serial loop over 8 seeds" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(5);
    var net = Net.init(prng.random());
    const playSeed = struct {
        fn f(n: *const Net, i: usize) PlayResult {
            var p = std.Random.DefaultPrng.init(1000 + i);
            return play(n, p.random(), .chance);
        }
    }.f;
    var out: [8]PlayResult = undefined;
    parallel.map(PlayResult, &out, &net, playSeed);
    for (out, 0..) |res, i| try std.testing.expectEqual(playSeed(&net, i), res);
}

test "SIMD input accumulation preserves scalar order and handles odd lengths (#317)" {
    var scalar = [_]f64{ 1.0, -2.0, 1e-12, 4.0, -5.0 };
    var vector = scalar;
    const rows = [_][5]f64{
        .{ 0.1, 0.2, 0.3, 0.4, 0.5 },
        .{ -0.5, 0.4, -0.3, 0.2, -0.1 },
        .{ 1e-14, -1e-14, 2e-14, 3e-14, 4e-14 },
    };
    for (rows) |row| {
        for (&scalar, row) |*x, y| x.* += y;
        accumulateRowsSimd(&vector, &row);
    }
    try std.testing.expectEqualSlices(f64, &scalar, &vector);
}

test "SIMD input forward matches scalar activations on varied boards (#317)" {
    var prng = std.Random.DefaultPrng.init(317);
    const rng = prng.random();
    var init_rng = std.Random.DefaultPrng.init(318);
    const net = Net.init(init_rng.random());
    for (0..128) |_| {
        var board: Board = undefined;
        for (&board) |*row| for (row) |*tile| {
            const level = rng.uintLessThan(u5, 17);
            tile.* = if (level == 0) 0 else @as(u32, 1) << level;
        };
        var scalar: Cache = .{};
        var vector: Cache = .{};
        const a = forwardImpl(&net, board, &scalar, false);
        const b = forwardImpl(&net, board, &vector, true);
        try std.testing.expectEqual(a, b);
        try std.testing.expectEqualSlices(usize, &scalar.rows, &vector.rows);
        try std.testing.expectEqualSlices(f64, &scalar.z1, &vector.z1);
        try std.testing.expectEqualSlices(f64, &scalar.a1, &vector.a1);
        try std.testing.expectEqualSlices(f64, &scalar.z2, &vector.z2);
        try std.testing.expectEqualSlices(f64, &scalar.a2, &vector.a2);
    }
}

test "SIMD input forward matches scalar with shipped weights (#317)" {
    if (h1_size != 128 or h2_size != 32 or feature_count != 0) return error.SkipZigTest;
    const net = try parse(@embedFile("nn_weights.bin"));
    var prng = std.Random.DefaultPrng.init(3170);
    for (0..256) |_| {
        var board: Board = undefined;
        for (&board) |*row| for (row) |*tile| {
            const level = prng.random().uintLessThan(u5, 17);
            tile.* = if (level == 0) 0 else @as(u32, 1) << level;
        };
        var scalar: Cache = .{};
        var vector: Cache = .{};
        try std.testing.expectEqual(forwardImpl(&net, board, &scalar, false), forwardImpl(&net, board, &vector, true));
        try std.testing.expectEqualSlices(f64, &scalar.z1, &vector.z1);
        try std.testing.expectEqualSlices(f64, &scalar.z2, &vector.z2);
    }
}

test "forward gives one finite scalar and is deterministic" {
    var prng = std.Random.DefaultPrng.init(5);
    const net = Net.init(prng.random());
    var a: Cache = .{};
    var b: Cache = .{};
    const first = forward(&net, sample_board, &a);
    try std.testing.expect(std.math.isFinite(first));
    try std.testing.expectEqual(first, forward(&net, sample_board, &b));
    try std.testing.expectEqual(first, b.out);
    // empty board differs from the sample: the input encoding reaches the output
    var empty: Cache = .{};
    try std.testing.expect(forward(&net, std.mem.zeroes(Board), &empty) != first);
}

test "PopArt target-stat update preserves the raw network output (#289)" {
    var rng = std.Random.DefaultPrng.init(289);
    var net = Net.init(rng.random());
    var state: PopArtState = .{};
    var cache: Cache = .{};
    const before = forward(&net, sample_board, &cache);
    const raw_target = 7.5;
    const normalized_target = state.startTrainingUpdate(&net, &cache, raw_target);
    const new_sigma = state.sigma();
    try std.testing.expectApproxEqAbs((before - state.mu) / new_sigma, cache.out, 1e-12);
    try std.testing.expectApproxEqAbs((raw_target - state.mu) / new_sigma, normalized_target, 1e-12);
    try std.testing.expectEqual(@as(u64, 1), state.value_updates);
    try std.testing.expectApproxEqAbs(popart_beta * raw_target, state.mu, 1e-12);
    try std.testing.expectApproxEqAbs(@max((1 - popart_beta) + popart_beta * raw_target * raw_target, state.mu * state.mu + popart_min_variance), state.nu, 1e-12);
    try std.testing.expectApproxEqAbs(popart_beta * before, state.value_mu, 1e-12);
    try std.testing.expectApproxEqAbs(@max((1 - popart_beta) + popart_beta * before * before, state.value_mu * state.value_mu + popart_min_variance), state.value_nu, 1e-12);
    try std.testing.expectApproxEqAbs(@max(state.value_nu - state.value_mu * state.value_mu, popart_min_variance), state.valueVariance(), 1e-12);

    var normalized_cache: Cache = .{};
    const normalized_output = forward(&net, sample_board, &normalized_cache);
    try std.testing.expectApproxEqAbs(before, state.denormalize(normalized_output), 1e-12);
    state.finishTrainingUpdate(&net);
    var restored_cache: Cache = .{};
    try std.testing.expectApproxEqAbs(before, forward(&net, sample_board, &restored_cache), 1e-12);
}

test "PopArt trainStep applies one normalized update and restores raw-value weights (#289)" {
    var rng = std.Random.DefaultPrng.init(290);
    var net = Net.init(rng.random());
    const initial = net;
    var expected = net;
    var state: PopArtState = .{};
    var expected_state: PopArtState = .{};
    var cache: Cache = .{};
    var expected_cache: Cache = .{};
    _ = forward(&expected, sample_board, &expected_cache);
    const normalized_target = expected_state.startTrainingUpdate(&expected, &expected_cache, 7.5);
    trainCapped(&expected, &expected_cache, normalized_target, 0.001, 0);
    expected_state.finishTrainingUpdate(&expected);

    trainStep(&net, &cache, sample_board, 7.5, 0.001, .{ .popart = &state }, &net);
    try std.testing.expectEqualSlices(f64, flatConst(&expected), flatConst(&net));
    try std.testing.expect(!std.mem.eql(f64, flatConst(&initial), flatConst(&net)));
    try std.testing.expectEqual(expected_state.mu, state.mu);
    try std.testing.expectEqual(expected_state.nu, state.nu);
    try std.testing.expectEqual(expected_state.value_mu, state.value_mu);
    try std.testing.expectEqual(expected_state.value_nu, state.value_nu);
    try std.testing.expectEqual(@as(u64, 1), state.updates);
    try std.testing.expectEqual(@as(u64, 1), state.value_updates);
    try std.testing.expect(weightsFinite(&net));
    var verify_cache: Cache = .{};
    try std.testing.expect(std.math.isFinite(forward(&net, sample_board, &verify_cache)));
}

test "PopArt stats sidecar round-trips its versioned state (#289)" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/popart.bin", .{tmp.sub_path});
    defer std.testing.allocator.free(path);

    const saved = PopArtState{
        .mu = 12.5,
        .nu = 200,
        .updates = 4321,
        .value_mu = 3.25,
        .value_nu = 20,
        .value_updates = 4321,
    };
    try saved.save(io, path);
    var loaded: PopArtState = .{};
    try loaded.load(io, std.testing.allocator, path);
    try std.testing.expectEqual(saved.mu, loaded.mu);
    try std.testing.expectEqual(saved.nu, loaded.nu);
    try std.testing.expectEqual(saved.updates, loaded.updates);
    try std.testing.expectEqual(saved.value_mu, loaded.value_mu);
    try std.testing.expectEqual(saved.value_nu, loaded.value_nu);
    try std.testing.expectEqual(saved.value_updates, loaded.value_updates);
    try std.testing.expectEqual(saved.beta, loaded.beta);
}

test "levelOf maps tiles to log2 levels and clamps large tiles" {
    try std.testing.expectEqual(@as(usize, 0), levelOf(0));
    try std.testing.expectEqual(@as(usize, 1), levelOf(2));
    try std.testing.expectEqual(@as(usize, 11), levelOf(2048));
    try std.testing.expectEqual(@as(usize, levels - 1), levelOf(65536));
}

test "same seed gives identical weights" {
    var a = std.Random.DefaultPrng.init(9);
    var b = std.Random.DefaultPrng.init(9);
    const net_a = Net.init(a.random());
    const net_b = Net.init(b.random());
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&net_a), std.mem.asBytes(&net_b));
}

test "train step matches the numerical gradient" {
    var prng = std.Random.DefaultPrng.init(3);
    var net = Net.init(prng.random());
    var cache: Cache = .{};
    _ = forward(&net, sample_board, &cache);
    const target = 0.7;
    var stepped = net;
    // lr 1: the parameter change equals minus the gradient
    train(&stepped, &cache, target, 1.0);

    const old = flat(&net);
    const new = flat(&stepped);
    const w1_base = @offsetOf(Net, "w1") / @sizeOf(f64);
    const b1_base = @offsetOf(Net, "b1") / @sizeOf(f64);
    const w2_base = @offsetOf(Net, "w2") / @sizeOf(f64);
    const b2_base = @offsetOf(Net, "b2") / @sizeOf(f64);
    const w3_base = @offsetOf(Net, "w3") / @sizeOf(f64);
    const b3_index = @offsetOf(Net, "b3") / @sizeOf(f64);
    // probes: a row for the tile 2 at cell 0 (level 1), the 4 at cell 5, a w1 row not on the board (level 3 at cell 0),
    // and one entry each from b1, w2, b2, w3, b3
    const probes = [_]usize{
        w1_base + 1 * h1_size + 7,
        w1_base + (5 * levels + 2) * h1_size + 3,
        w1_base + 3 * h1_size + 7,
        b1_base + 2,
        w2_base + 2 * h1_size + 5,
        b2_base + 1,
        w3_base + 4,
        b3_index,
    };
    const eps = 1e-6;
    for (probes) |idx| {
        var plus = net;
        var minus = net;
        flat(&plus)[idx] += eps;
        flat(&minus)[idx] -= eps;
        const numeric = (lossOf(&plus, sample_board, target) - lossOf(&minus, sample_board, target)) / (2 * eps);
        const analytic = old[idx] - new[idx];
        try std.testing.expectApproxEqAbs(numeric, analytic, 1e-5);
    }
}

test "one training step reduces loss on a fixed sample" {
    var prng = std.Random.DefaultPrng.init(4);
    var net = Net.init(prng.random());
    const target = 1.5;
    const before = lossOf(&net, sample_board, target);
    var cache: Cache = .{};
    _ = forward(&net, sample_board, &cache);
    train(&net, &cache, target, 0.05);
    try std.testing.expect(lossOf(&net, sample_board, target) < before);
}

test "bestMove returns null on a dead board and a legal move otherwise" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), bestMove(&net, dead, .greedy));
    try std.testing.expectEqual(@as(?Direction, null), bestMove(&net, dead, .chance));
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var moved = state;
    try std.testing.expect(moved.slide(bestMove(&net, state, .greedy).?).changed);
    try std.testing.expect(moved.slide(bestMove(&net, state, .chance).?).changed);
}

/// Board with a few empties: enough spawns for a 2-ply search to have work to do.
const mid_board: GameState = .{ .board = .{
    .{ 2, 4, 8, 16 },
    .{ 0, 2, 0, 4 },
    .{ 8, 0, 2, 0 },
    .{ 0, 16, 0, 32 },
} };

test "deep returns only legal moves and null on a dead board" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), bestMove(&net, dead, .deep));
    var moved = mid_board;
    try std.testing.expect(moved.slide(bestMove(&net, mid_board, .deep).?).changed);
}

test "deep is deterministic and searches more leaves than chance" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    var chance_stats: Stats = .{};
    var a: Stats = .{};
    var b: Stats = .{};
    _ = bestMoveWithStats(&net, mid_board, .chance, &chance_stats);
    const first = bestMoveWithStats(&net, mid_board, .deep, &a);
    const second = bestMoveWithStats(&net, mid_board, .deep, &b);
    try std.testing.expect(first != null);
    try std.testing.expectEqual(first, second);
    try std.testing.expectEqual(a.leaves, b.leaves);
    try std.testing.expectEqual(a.hits, b.hits);
    try std.testing.expect(a.leaves > chance_stats.leaves);
}

/// Late-game board (4 empties): the unpruned 3-ply tree is cheap enough to compare against the cut.
const late_board: GameState = .{ .board = .{
    .{ 2, 4, 8, 16 },
    .{ 32, 64, 128, 0 },
    .{ 2, 4, 8, 0 },
    .{ 4, 2, 0, 0 },
} };

test "deep3 returns only legal moves on random boards" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(108);
    const net = Net.init(prng.random());
    var rng = std.Random.DefaultPrng.init(3);
    const random = rng.random();
    for (0..10) |_| {
        var state: GameState = .{};
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (random.uintLessThan(u8, 4) != 0) continue;
                state.board[r][c] = @as(core.Tile, 2) << random.uintLessThan(u5, 8);
            }
        }
        if (bestMove(&net, state, .deep3)) |d| {
            var moved = state;
            try std.testing.expect(moved.slide(d).changed);
        } else {
            for ([_]Direction{ .up, .left, .down, .right }) |d| {
                var moved = state;
                try std.testing.expect(!moved.slide(d).changed);
            }
        }
    }
}

test "independent root scores match the serial search including illegal moves (#319)" {
    var prng = std.Random.DefaultPrng.init(319);
    const net = Net.init(prng.random());
    const boards = [_]Board{
        .{ .{ 2, 4, 8, 16 }, .{ 32, 64, 128, 256 }, .{ 512, 1024, 2048, 4096 }, .{ 8192, 16384, 2, 2 } },
        .{ .{ 2, 4, 2, 4 }, .{ 4, 2, 4, 2 }, .{ 2, 4, 2, 4 }, .{ 4, 2, 4, 2 } },
    };
    for (boards) |board| {
        const state = GameState{ .board = board };
        for ([_]Mode{ .deep, .deep3 }) |mode| {
            var cache: ValueCache = .{};
            const depth = searchDepth(mode, board, false);
            var search = newSearch(.{ .single = &net }, &cache, null, depth, false, 0);
            const best = rootMove(&search, state, dir_order, null);
            var independent_dir: ?Direction = null;
            var independent_value = -std.math.inf(f64);
            var worker_cache: ValueCache = .{};
            for (dir_order) |dir| {
                const serial_value = rootMoveScore(&search, state, dir);
                const worker_value = scoreRootMove(&net, state, mode, dir, &worker_cache);
                try std.testing.expectEqual(serial_value, worker_value);
                const value = worker_value orelse continue;
                if (independent_dir == null or value > independent_value) {
                    independent_dir = dir;
                    independent_value = value;
                }
            }
            try std.testing.expectEqual(best.dir, independent_dir);
            try std.testing.expectEqual(best.value, independent_value);
        }
    }
}

test "node cache key separates board, kind, depth and a relevant cprob (#263)" {
    const a = NodeCache.keyFor(mid_board.board, .value, 3, 0.25);
    var changed = mid_board.board;
    changed[0][0] = 4;
    const b = NodeCache.keyFor(changed, .value, 3, 0.25);
    try std.testing.expect(!std.mem.eql(u8, &a.cells, &b.cells));
    const kind = NodeCache.keyFor(mid_board.board, .chance, 3, 0.25);
    try std.testing.expect(a.kind != kind.kind);
    const depth = NodeCache.keyFor(mid_board.board, .value, 2, 0.25);
    try std.testing.expect(a.depth != depth.depth);
    const cprob = NodeCache.keyFor(mid_board.board, .value, 3, 0.5);
    try std.testing.expect(a.cprob != cprob.cprob);

    // the fixed cut is only tested at chance depth > 1, so value depth <= 2 and chance depth <= 1
    // never read cprob and deliberately share one key
    const shallow_value = NodeCache.keyFor(mid_board.board, .value, 2, 0.01);
    const shallow_chance = NodeCache.keyFor(mid_board.board, .chance, 1, 0.01);
    try std.testing.expectEqual(@as(f64, 0), shallow_value.cprob);
    try std.testing.expectEqual(@as(f64, 0), shallow_chance.cprob);
    const deep_chance = NodeCache.keyFor(mid_board.board, .chance, 2, 0.5);
    try std.testing.expectEqual(@as(f64, 0.5), deep_chance.cprob);
}

test "node cache collision is a miss, never another board's value (#263)" {
    var table: NodeCache = .{};
    var stats: Stats = .{};
    const wanted = NodeCache.keyFor(mid_board.board, .value, 3, 0.25);
    const other = NodeCache.keyFor(late_board.board, .value, 3, 0.25);
    // force the two keys into one slot, as a hash collision would
    table.entries[NodeCache.slotOf(wanted)] = .{ .key = other, .result = .{ .value = 99 } };
    try std.testing.expect(table.get(wanted, &stats) == null);
    try std.testing.expectEqual(@as(u64, 1), stats.node_probes);
    try std.testing.expectEqual(@as(u64, 0), stats.node_hits);

    table.put(wanted, .{ .value = 1.5 });
    try std.testing.expectEqual(@as(f64, 1.5), table.get(wanted, &stats).?.value);
    try std.testing.expectEqual(@as(u64, 1), stats.node_hits);
    try std.testing.expect(table.get(NodeCache.keyFor(mid_board.board, .value, 2, 0.25), &stats) == null);
    try std.testing.expect(table.get(NodeCache.keyFor(mid_board.board, .chance, 3, 0.25), &stats) == null);
    try std.testing.expect(table.get(NodeCache.keyFor(mid_board.board, .value, 3, 0.5), &stats) == null);
}

test "node cache slot is 48 bytes and the table size follows the capacity (#263)" {
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(NodeCache.Entry));
    try std.testing.expectEqual(@as(usize, NodeCache.capacity) * 48, @sizeOf(NodeCache));
    const expected = if (builtin.target.cpu.arch.isWasm()) 512 else 16384;
    try std.testing.expectEqual(@as(usize, expected), NodeCache.capacity);
}

test "node cache keeps the direct-leaf flag (#263/#260)" {
    var table: NodeCache = .{};
    var stats: Stats = .{};
    const key = NodeCache.keyFor(mid_board.board, .chance, 3, 0.25);
    table.put(key, .{ .value = -0.25, .direct_leaf = true });
    const got = table.get(key, &stats).?;
    try std.testing.expectEqual(@as(f64, -0.25), got.value);
    try std.testing.expect(got.direct_leaf);
}

test "node table is cleared at the start of every top-level search (#263)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    var cache: ValueCache = .{};
    var stats: Stats = .{};
    const table = sharedNodeCache();
    // a depth no 3-ply search can store, so only the per-search clear can remove it
    const stale = NodeCache.keyFor(late_board.board, .value, 200, 1);
    table.put(stale, .{ .value = 123 });
    _ = bestMoveWithCacheValueOpts(vfn, mid_board, .deep3, &stats, &cache, .{ .node_tt = true });
    try std.testing.expect(table.get(stale, &stats) == null);

    // explicit off: no probes and no clear, so the table is untouched
    table.put(stale, .{ .value = 123 });
    var off_stats: Stats = .{};
    _ = bestMoveWithCacheValueOpts(vfn, mid_board, .deep3, &off_stats, &cache, .{ .node_tt = false });
    try std.testing.expectEqual(@as(u64, 0), off_stats.node_probes);
    try std.testing.expectEqual(@as(f64, 123), table.get(stale, &off_stats).?.value);
}

test "node TT defaults on for 3-ply and never for 2-ply (#266)" {
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    var cache: ValueCache = .{};
    const table = sharedNodeCache();
    // a depth no 3-ply search can store, so only a 3-ply search's per-search clear can remove it
    const stale = NodeCache.keyFor(late_board.board, .value, 200, 1);

    // `.deep` is 2-ply: the default leaves the table alone, exactly as an explicit off would
    table.put(stale, .{ .value = 123 });
    var deep_stats: Stats = .{};
    const deep_move = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &deep_stats, &cache, .{});
    try std.testing.expectEqual(@as(u64, 0), deep_stats.node_probes);
    try std.testing.expectEqual(@as(f64, 123), table.get(stale, &deep_stats).?.value);

    // ... and takes the same path: same move and leaf work, so the 2-ply result is bit-identical
    var off_stats: Stats = .{};
    var off_cache: ValueCache = .{};
    const off_move = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &off_stats, &off_cache, .{ .node_tt = false });
    try std.testing.expectEqual(deep_move, off_move);
    try std.testing.expectEqual(deep_stats.leaves, off_stats.leaves);
    try std.testing.expectEqual(deep_stats.hits, off_stats.hits);

    // `.deep3` is 3-ply: the default clears, probes and hits like `.node_tt = true`
    var deep3_stats: Stats = .{};
    _ = bestMoveWithCacheValueOpts(vfn, mid_board, .deep3, &deep3_stats, &cache, .{});
    try std.testing.expect(deep3_stats.node_probes > 0);
    try std.testing.expect(deep3_stats.node_hits > 0);
    try std.testing.expect(table.get(stale, &deep3_stats) == null);
}

test "node cache withholds a node from a budget-exhausted search (#263)" {
    var prng = std.Random.DefaultPrng.init(263);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var cache: ValueCache = .{};
    var table: NodeCache = .{};
    var search: Search = .{ .vfn = vfn, .cache = &cache, .cutoff = deep3_cutoff, .ply = 3, .nodes = &table };
    search.nodes_left = 3;
    _ = search.value(mid_board, 3, 1);
    try std.testing.expect(search.out_of_budget);
    try std.testing.expect(table.get(NodeCache.keyFor(mid_board.board, .value, 3, 1), &search.stats) == null);

    // a completed rerun of the same root stores it
    search = .{ .vfn = vfn, .cache = &cache, .cutoff = deep3_cutoff, .ply = 3, .nodes = &table };
    _ = search.value(mid_board, 3, 1);
    try std.testing.expect(!search.out_of_budget);
    try std.testing.expect(table.get(NodeCache.keyFor(mid_board.board, .value, 3, 1), &search.stats) != null);
}

test "node cache values match a fresh search at the same cprob (#263)" {
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    inline for ([_]struct { depth: u8, cprob: f64 }{
        .{ .depth = 2, .cprob = 0.37 },
        .{ .depth = 3, .cprob = 0.37 },
        .{ .depth = 3, .cprob = 1 },
    }) |case| {
        var plain_leaf: ValueCache = .{};
        var cached_leaf: ValueCache = .{};
        var table: NodeCache = .{};
        var plain: Search = .{ .vfn = vfn, .cache = &plain_leaf, .cutoff = deep3_cutoff, .ply = 3 };
        var cached: Search = .{ .vfn = vfn, .cache = &cached_leaf, .cutoff = deep3_cutoff, .ply = 3, .nodes = &table };
        const want = plain.value(mid_board, case.depth, case.cprob);
        const got = cached.value(mid_board, case.depth, case.cprob);
        try std.testing.expectApproxEqAbs(want, got, 1e-12);
        try std.testing.expectEqual(got, cached.value(mid_board, case.depth, case.cprob));
        try std.testing.expect(cached.stats.node_hits >= 1);
    }
}

test "node cache keeps the chosen move along a game (#263/#266)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    var prng = std.Random.DefaultPrng.init(263);
    const random = prng.random();
    var state = GameState.gameStart(random);
    var plain_leaf: ValueCache = .{};
    var cached_leaf: ValueCache = .{};
    var default_leaf: ValueCache = .{};
    for (0..4) |i| {
        const mode: Mode = if (i % 2 == 0) .deep3 else .deep;
        var plain_stats: Stats = .{};
        var cached_stats: Stats = .{};
        var default_stats: Stats = .{};
        const plain = bestMoveWithCacheValueOpts(vfn, state, mode, &plain_stats, &plain_leaf, .{ .node_tt = false });
        const cached = bestMoveWithCacheValueOpts(vfn, state, mode, &cached_stats, &cached_leaf, .{ .node_tt = true });
        // : the default is the same search as explicit on for 3-ply and as explicit off for 2-ply
        const default = bestMoveWithCacheValueOpts(vfn, state, mode, &default_stats, &default_leaf, .{});
        try std.testing.expectEqual(plain, cached);
        try std.testing.expectEqual(plain, default);
        if (mode == .deep3) {
            try std.testing.expect(default_stats.node_probes > 0);
        } else {
            try std.testing.expectEqual(@as(u64, 0), default_stats.node_probes);
        }
        const dir = plain orelse break;
        var moved = state;
        try std.testing.expect(moved.slide(dir).changed);
        _ = state.move(dir, random);
    }
}

test "deep3 keeps the value within 5% of the unpruned 3-ply search" {
    const net = try parse(@embedFile("nn_weights.bin"));
    var cache: ValueCache = .{};
    var full: Search = .{ .vfn = .{ .single = &net }, .cache = &cache };
    var pruned: Search = .{ .vfn = .{ .single = &net }, .cache = &cache, .cutoff = deep3_cutoff };
    const plain = full.value(late_board, 3, 1);
    const cut = pruned.value(late_board, 3, 1);
    try std.testing.expect(@abs(cut - plain) <= 0.05 * @abs(plain));
}

test "shallow fallback tree preserves each spawn probability and shallow-evaluates cut children" {
    const branches = [_]struct { p: f64, cut: bool, deep: f64, shallow: f64 }{
        .{ .p = 0.45, .cut = false, .deep = 10, .shallow = 1 },
        .{ .p = 0.05, .cut = true, .deep = 20, .shallow = 2 },
        .{ .p = 0.45, .cut = false, .deep = 30, .shallow = 3 },
        .{ .p = 0.05, .cut = true, .deep = 40, .shallow = 4 },
    };
    var probabilities: [branches.len]f64 = undefined;
    var values: [branches.len]f64 = undefined;
    var probability_sum: f64 = 0;
    for (branches, 0..) |branch, i| {
        probabilities[i] = branch.p;
        probability_sum += branch.p;
        const depth = spawnBranchDepth(3, branch.cut, true);
        try std.testing.expectEqual(if (branch.cut) @as(u8, 2) else @as(u8, 3), depth);
        values[i] = if (depth == 2) branch.shallow else branch.deep;
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1), probability_sum, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 18.3), weightedChanceValue(&values, &probabilities), 1e-12);
    try std.testing.expectEqual(@as(u8, 1), spawnBranchDepth(1, true, true));
    try std.testing.expectEqual(@as(u8, 3), spawnBranchDepth(3, true, false));
}

test "all-cut fallback evaluates every child one ply shallower" {
    const probabilities = [_]f64{ 0.45, 0.05, 0.45, 0.05 };
    const shallow_values = [_]f64{ 1, 2, 3, 4 };
    for (0..probabilities.len) |_| try std.testing.expectEqual(@as(u8, 2), spawnBranchDepth(3, true, true));
    try std.testing.expectApproxEqAbs(@as(f64, 2.1), weightedChanceValue(&shallow_values, &probabilities), 1e-12);
}

test "shallow fallback evaluates actual spawned boards at full and reduced depth" {
    var prng = std.Random.DefaultPrng.init(2610);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    const empties = countEmpty(late_board.board);
    var expected_cache: ValueCache = .{};
    var expected_search: Search = .{ .vfn = vfn, .cache = &expected_cache };

    var recovered_cache: ValueCache = .{};
    var recovered: Search = .{ .vfn = vfn, .cache = &recovered_cache, .cutoff = 0.1, .recover_cut_branches = true };
    var expected: f64 = 0;
    for (0..core.BoardSize) |r| for (0..core.BoardSize) |c| {
        if (late_board.board[r][c] != 0) continue;
        for (spawn_tiles) |spawn| {
            var child = late_board;
            child.board[r][c] = spawn.value;
            const cut = spawn.p / @as(f64, @floatFromInt(empties)) < recovered.cutoff;
            expected += spawn.p / @as(f64, @floatFromInt(empties)) * expected_search.value(child, if (cut) 1 else 2, spawn.p);
        }
    };
    try std.testing.expectApproxEqAbs(expected, recovered.chance(late_board, 2, 1), 1e-10);
}

test "all-cut fallback uses shallow child search and complete local spawn mass" {
    var prng = std.Random.DefaultPrng.init(2611);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    const empties = countEmpty(late_board.board);
    var expected_cache: ValueCache = .{};
    var expected_search: Search = .{ .vfn = vfn, .cache = &expected_cache };
    var recovered_cache: ValueCache = .{};
    var recovered: Search = .{ .vfn = vfn, .cache = &recovered_cache, .cutoff = 1, .recover_cut_branches = true };
    var expected: f64 = 0;
    for (0..core.BoardSize) |r| for (0..core.BoardSize) |c| {
        if (late_board.board[r][c] != 0) continue;
        for (spawn_tiles) |spawn| {
            var child = late_board;
            child.board[r][c] = spawn.value;
            expected += spawn.p / @as(f64, @floatFromInt(empties)) * expected_search.value(child, 1, spawn.p);
        }
    };
    try std.testing.expectApproxEqAbs(expected, recovered.chance(late_board, 2, 1), 1e-10);
}

test "disabled cutoff fallback is identical to deep3 and terminal values remain zero" {
    var prng = std.Random.DefaultPrng.init(261);
    const net = Net.init(prng.random());
    var full_cache: ValueCache = .{};
    var full: Search = .{ .vfn = .{ .single = &net }, .cache = &full_cache, .ply = 3 };
    var recover_cache: ValueCache = .{};
    var recover: Search = .{ .vfn = .{ .single = &net }, .cache = &recover_cache, .ply = 3, .recover_cut_branches = true };
    try std.testing.expectEqual(full.value(mid_board, 3, 1), recover.value(mid_board, 3, 1));

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(f64, 0), recover.value(dead, 3, 1));
}

test "mass cutoff is off at 0 and full width at 1 (#182)" {
    var prng = std.Random.DefaultPrng.init(182);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    // (a) the phase rows carry no mass cut, so `.deep` / `.deep3` search what they searched before
    for ([_]Mode{ .deep, .deep3, .deep4, .deep3_recover, .phase_h1, .phase_h2, .phase_h3, .phase_h4 }) |mode| {
        try std.testing.expectEqual(@as(f64, 0), phaseDepth(mode, mid_board.board).mass_cutoff);
    }
    var plain_cache: ValueCache = .{};
    var plain: Search = .{ .vfn = vfn, .cache = &plain_cache, .cutoff = deep3_cutoff, .ply = 3 };
    var zero_cache: ValueCache = .{};
    var zero: Search = .{ .vfn = vfn, .cache = &zero_cache, .cutoff = deep3_cutoff, .mass_cutoff = 0, .ply = 3 };
    try std.testing.expectEqual(plain.value(mid_board, 3, 1), zero.value(mid_board, 3, 1));
    // (b) `mass_cutoff = 1` is the full-width spelling (as `full_width_cutoff` is for `cutoff`) and
    // every branch is evaluated, so the value is the `mass_cutoff = 0` one
    try std.testing.expectEqual(@as(f64, 0), searchMassCutoff(1));
    var full_cache: ValueCache = .{};
    var full: Search = .{ .vfn = vfn, .cache = &full_cache, .mass_cutoff = 1, .ply = 2 };
    var off_cache: ValueCache = .{};
    var off: Search = .{ .vfn = vfn, .cache = &off_cache, .ply = 2 };
    try std.testing.expectEqual(off.value(mid_board, 2, 1), full.value(mid_board, 2, 1));
}

test "mass cutoff 0.5 drops the low-probability tail of one chance node (#182)" {
    // `late_board` has 4 empty cells, so its chance node has 8 branches of `0.9/4` and `0.1/4`
    var branches: [max_chance_branches]ChanceBranch = undefined;
    var count: usize = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (late_board.board[r][c] != 0) continue;
            for (spawn_tiles) |spawn| {
                branches[count] = .{ .r = r, .c = c, .tile = spawn.value, .p = spawn.p };
                count += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 8), count);
    var dropped: [max_chance_branches]bool = @splat(false);
    const share = massCutMask(branches[0..count], 0.5, &dropped);
    // the node's total is 4.0, so the limit is 2.0: the four 4s (0.1 each) go first, then one 2 (0.9)
    // still fits at 1.3 and the second one would not (2.2 > 2.0)
    try std.testing.expect(share > 0);
    try std.testing.expect(share <= 0.5);
    try std.testing.expectApproxEqAbs(0.325, share, 1e-12);
    var kept_fours: usize = 0;
    var kept_twos: usize = 0;
    var dropped_twos: usize = 0;
    for (branches[0..count], 0..) |b, i| {
        if (dropped[i]) {
            if (b.tile == 4) continue;
            dropped_twos += 1;
            // the dropped 2 is the first one in board order: the tie-break of equal probabilities
            try std.testing.expectEqual(@as(usize, 0), i);
        } else if (b.tile == 4) kept_fours += 1 else kept_twos += 1;
    }
    // the whole `0.1` tail goes before any `0.9` one, and then exactly one `0.9` fits in the budget
    try std.testing.expectEqual(@as(usize, 0), kept_fours);
    try std.testing.expectEqual(@as(usize, 1), dropped_twos);
    try std.testing.expectEqual(@as(usize, 3), kept_twos);
    // the same cut really moves the searched value: the value leaves drop the whole 4 spawn
    var prng = std.Random.DefaultPrng.init(182);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var off_cache: ValueCache = .{};
    var off: Search = .{ .vfn = vfn, .cache = &off_cache, .ply = 1 };
    var cut_cache: ValueCache = .{};
    var cut: Search = .{ .vfn = vfn, .cache = &cut_cache, .mass_cutoff = 0.5, .ply = 1 };
    try std.testing.expect(off.chance(late_board, 1, 1) != cut.chance(late_board, 1, 1));
}

/// Modes that pick their own depth from the phase table.
const phase_modes = [_]Mode{ .phase_h1, .phase_h2, .phase_h3 };

test "risk weight 0 is the plain expectation, bit for bit (#187)" {
    var prng = std.Random.DefaultPrng.init(187);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    // every existing phase row carries no risk weight, so the phase table is unchanged
    for ([_]Mode{ .deep, .deep3, .deep4, .deep3_recover, .deep_mass, .phase_h1, .phase_h2, .phase_h3, .phase_h4 }) |mode| {
        try std.testing.expectEqual(@as(f64, 0), phaseDepth(mode, mid_board.board).risk_weight);
    }
    var plain_cache: ValueCache = .{};
    var plain: Search = .{ .vfn = vfn, .cache = &plain_cache, .ply = 2 };
    var zero_cache: ValueCache = .{};
    var zero: Search = .{ .vfn = vfn, .cache = &zero_cache, .ply = 2, .risk_weight = 0 };
    try std.testing.expectEqual(plain.value(mid_board, 2, 1), zero.value(mid_board, 2, 1));
    // and `chance` at weight 0 is the branch mean recomputed here in the old way, with no tail term
    const chance_value = zero.chance(mid_board, 2, 1);
    var total: f64 = 0;
    var mass: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (mid_board.board[r][c] != 0) continue;
            for (spawn_tiles) |spawn| {
                var child = mid_board;
                child.board[r][c] = spawn.value;
                total += spawn.p * plain.value(child, 2, spawn.p);
                mass += spawn.p;
            }
        }
    }
    try std.testing.expectEqual(total / mass, chance_value);
    try std.testing.expectEqual(total / mass, riskScore(total / mass, &.{ 0, 1 }, &.{ 0.5, 0.5 }, 0));
    // the chosen move is the same as well, not just the value
    var off_stats: Stats = .{};
    var off_cache: ValueCache = .{};
    var default_stats: Stats = .{};
    var default_cache: ValueCache = .{};
    try std.testing.expectEqual(
        bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &default_stats, &default_cache, .{}),
        bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &off_stats, &off_cache, .{ .risk_weight = 0 }),
    );
}

test "risk weight 1 scores a chance node by its CVaR, below its mean (#187)" {
    var prng = std.Random.DefaultPrng.init(1870);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var mean_cache: ValueCache = .{};
    var mean: Search = .{ .vfn = vfn, .cache = &mean_cache, .ply = 2 };
    var cvar_cache: ValueCache = .{};
    var cvar: Search = .{ .vfn = vfn, .cache = &cvar_cache, .ply = 2, .risk_weight = 1 };
    const m = mean.value(mid_board, 2, 1);
    const c = cvar.value(mid_board, 2, 1);
    // the tail mean of a node never beats the node's mean, so CVaR-only scoring can only lower it
    try std.testing.expect(c <= m);
    try std.testing.expect(c < m);
    // per node: the CVaR of a spread node sits strictly below its mean, and equals it when flat
    const even = [_]f64{ 2, 2, 2 };
    const even_p = [_]f64{ 0.9, 0.05, 0.05 };
    try std.testing.expectEqual(@as(f64, 2), downsideRisk(&even, &even_p, risk_alpha));
    const spread = [_]f64{ 0, 10 };
    const spread_p = [_]f64{ 0.9, 0.1 };
    try std.testing.expect(downsideRisk(&spread, &spread_p, risk_alpha) < riskScore(1, &spread, &spread_p, 0));
}

test "risk weight picks the lower-variance move at equal expectation (#187)" {
    // two moves of the same expectation: `narrow` always scores 1, `wide` scores 0 nine times out of
    // ten and 10 on the 4 spawn, so its 90% branch is the whole lower tail
    const narrow_v = [_]f64{ 1, 1 };
    const narrow_p = [_]f64{ 0.9, 0.1 };
    const wide_v = [_]f64{ 0, 10 };
    const wide_p = [_]f64{ 0.9, 0.1 };
    var narrow_total: f64 = 0;
    for (narrow_v, narrow_p) |v, p| narrow_total += v * p;
    var wide_total: f64 = 0;
    for (wide_v, wide_p) |v, p| wide_total += v * p;
    try std.testing.expectApproxEqAbs(narrow_total, wide_total, 1e-12);
    // at weight 0 the two tie: the plain expectation cannot tell them apart
    const narrow_mean = narrow_total;
    const wide_mean = wide_total;
    try std.testing.expectEqual(narrow_mean, riskScore(narrow_mean, &narrow_v, &narrow_p, 0));
    try std.testing.expectApproxEqAbs(
        riskScore(narrow_mean, &narrow_v, &narrow_p, 0),
        riskScore(wide_mean, &wide_v, &wide_p, 0),
        1e-12,
    );
    // above 0 the wide move's tail (0) scores it below the flat one, which has no tail below its mean
    const narrow = riskScore(narrow_mean, &narrow_v, &narrow_p, 0.3);
    const wide = riskScore(wide_mean, &wide_v, &wide_p, 0.3);
    try std.testing.expect(narrow > wide);
    // the tail is the lowest `risk_alpha` of the mass: one of four equal branches is 25%
    const quartile_v = [_]f64{ 3, 1, 4, 2 };
    const quartile_p = [_]f64{ 0.25, 0.25, 0.25, 0.25 };
    try std.testing.expectEqual(@as(f64, 1), downsideRisk(&quartile_v, &quartile_p, risk_alpha));
}

test "phase modes return only legal moves on random boards (#157)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(157);
    const net = Net.init(prng.random());
    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    var rng = std.Random.DefaultPrng.init(3);
    const random = rng.random();
    var states: [5]GameState = undefined;
    for (&states) |*state| {
        state.* = .{};
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (random.uintLessThan(u8, 4) != 0) continue;
                state.board[r][c] = @as(core.Tile, 2) << random.uintLessThan(u5, 8);
            }
        }
    }
    for (phase_modes) |mode| {
        try std.testing.expectEqual(@as(?Direction, null), bestMove(&net, dead, mode));
        for (states) |state| {
            if (bestMove(&net, state, mode)) |d| {
                var moved = state;
                try std.testing.expect(moved.slide(d).changed);
            } else {
                for ([_]Direction{ .up, .left, .down, .right }) |d| {
                    var moved = state;
                    try std.testing.expect(!moved.slide(d).changed);
                }
            }
        }
    }
}

test "phase_h1 equals deep once the board has 8 or more empty cells (#157)" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    // 13 empty cells: phase_h1 takes its 2-ply row, and so does phase_h3 (row boundary 10)
    const open: GameState = .{ .board = sample_board };
    var deep_stats: Stats = .{};
    var h1_stats: Stats = .{};
    var h3_stats: Stats = .{};
    const deep_move = bestMoveWithStats(&net, open, .deep, &deep_stats);
    const h1_move = bestMoveWithStats(&net, open, .phase_h1, &h1_stats);
    const h3_move = bestMoveWithStats(&net, open, .phase_h3, &h3_stats);
    try std.testing.expect(deep_move != null);
    try std.testing.expectEqual(deep_move, h1_move);
    try std.testing.expectEqual(deep_move, h3_move);
    // same ply and no cut: the same search, not just the same move
    try std.testing.expectEqual(deep_stats.leaves, h1_stats.leaves);
    try std.testing.expectEqual(deep_stats.leaves, h3_stats.leaves);
}

test "phase_h2 and phase_h3 equal deep3 on a board with 4 empty cells (#157)" {
    const net = try parse(@embedFile("nn_weights.bin"));
    // 4 empty cells: both cut values keep every spawn branch that matters, so the moves agree
    var deep3_stats: Stats = .{};
    var h2_stats: Stats = .{};
    var h3_stats: Stats = .{};
    const deep3_move = bestMoveWithStats(&net, late_board, .deep3, &deep3_stats);
    const h2_move = bestMoveWithStats(&net, late_board, .phase_h2, &h2_stats);
    const h3_move = bestMoveWithStats(&net, late_board, .phase_h3, &h3_stats);
    try std.testing.expect(deep3_move != null);
    try std.testing.expectEqual(deep3_move, h2_move);
    try std.testing.expectEqual(deep3_move, h3_move);
}

test "phase h4 depth 4 only in the last two empty cells (#171)" {
    if (!width.slow_tests) return error.SkipZigTest;
    // 4-ply from 2 empty cells down; everything above that is the `deep3` baseline
    for (0..17) |empties| {
        const depth = phaseDepth(.phase_h4, boardWithEmpties(empties));
        try std.testing.expectEqual(if (empties <= 2) @as(u8, 4) else @as(u8, 3), depth.ply);
        try std.testing.expectEqual(deep3_cutoff, depth.cutoff);
    }
    // the extra ply is really searched: one more own move and spawn layer than `deep3` costs on the same board
    var prng = std.Random.DefaultPrng.init(171);
    const net = Net.init(prng.random());
    const endgame: GameState = .{ .board = boardWithEmpties(2) };
    var h4_stats: Stats = .{};
    var deep3_stats: Stats = .{};
    const h4_move = bestMoveWithStats(&net, endgame, .phase_h4, &h4_stats);
    _ = bestMoveWithStats(&net, endgame, .deep3, &deep3_stats);
    try std.testing.expect(h4_move != null);
    try std.testing.expect(h4_stats.leaves > deep3_stats.leaves);
}

test "deep4 searches one row at 4-ply and the deep3 cutoff (#267)" {
    if (!width.slow_tests) return error.SkipZigTest;
    // every empty-cell count uses the same row: the extra ply is the only difference from `deep3`
    for (0..17) |empties| {
        try std.testing.expectEqual(
            Depth{ .ply = 4, .cutoff = deep3_cutoff },
            phaseDepth(.deep4, boardWithEmpties(empties)),
        );
    }
    // and the extra ply is really searched, even where `phase_h4` would still be at 3-ply
    var prng = std.Random.DefaultPrng.init(267);
    const net = Net.init(prng.random());
    const midgame: GameState = .{ .board = boardWithEmpties(3) };
    var deep4_stats: Stats = .{};
    var deep3_stats: Stats = .{};
    const deep4_move = bestMoveWithStats(&net, midgame, .deep4, &deep4_stats);
    _ = bestMoveWithStats(&net, midgame, .deep3, &deep3_stats);
    try std.testing.expect(deep4_move != null);
    try std.testing.expect(deep4_stats.leaves > deep3_stats.leaves);
}

test "spawn sampler is deterministic, bounded and full-width safe (#320)" {
    // four empty cells = eight branches, the same board order `chanceResult` builds
    var branches: [max_chance_branches]ChanceBranch = undefined;
    var count: usize = 0;
    for (0..2) |r| for (0..2) |c| for (spawn_tiles) |spawn| {
        branches[count] = .{ .r = r, .c = c, .tile = spawn.value, .p = spawn.p };
        count += 1;
    };
    const hash = ValueCache.keyOf(mid_board.board);

    // `keep` at the branch count, and above it, is the full-width search
    for ([_]usize{ count, count + 1, count + 10 }) |keep| {
        var dropped: [max_chance_branches]bool = @splat(false);
        sampleMask(branches[0..count], keep, default_sample_seed, hash, 4, &dropped);
        for (dropped[0..count]) |d| try std.testing.expect(!d);
    }

    // a finite sample keeps exactly `keep` branches, deterministically per seed and node
    for ([_]usize{ 1, 2, 3, 6 }) |keep| {
        var a: [max_chance_branches]bool = @splat(false);
        var b: [max_chance_branches]bool = @splat(false);
        sampleMask(branches[0..count], keep, 7, hash, 4, &a);
        sampleMask(branches[0..count], keep, 7, hash, 4, &b);
        var kept: usize = 0;
        for (a[0..count], b[0..count]) |x, y| {
            try std.testing.expectEqual(x, y);
            if (!x) kept += 1;
        }
        try std.testing.expectEqual(keep, kept);
    }

    // another seed samples another set (the 8-branch case has room to differ)
    var same: [max_chance_branches]bool = @splat(false);
    var other: [max_chance_branches]bool = @splat(false);
    sampleMask(branches[0..count], 2, 7, hash, 4, &same);
    sampleMask(branches[0..count], 2, 8, hash, 4, &other);
    try std.testing.expect(!std.mem.eql(bool, same[0..count], other[0..count]));

    // the mass cut's drops survive the sample
    var cut: [max_chance_branches]bool = @splat(false);
    cut[0] = true;
    sampleMask(branches[0..count], 1, 7, hash, 4, &cut);
    try std.testing.expect(cut[0]);
}

test "spawn sample above a node's width is the plain search (#320)" {
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    // a depth-1 chance node has no deeper chance node, so a sample at or above its branch count
    // (mid_board: 6 empties = 12 branches) must reproduce the unsampled value bit for bit
    var cache: ValueCache = .{};
    var plain: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff };
    var sampled: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff, .sample = 12 };
    const expected = plain.chance(mid_board, 1, 1);
    try std.testing.expectEqual(expected, sampled.chance(mid_board, 1, 1));
    // and exactly that width: one fewer branch is a different (sampled) node
    var narrower: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff, .sample = 11 };
    try std.testing.expect(narrower.chance(mid_board, 1, 1) != expected);
}

test "sample_min_depth 2 keeps the last spawn layer exact (#320)" {
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    var cache: ValueCache = .{};
    var plain: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff };
    var sampled: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff, .sample = 2 };
    var last_full: Search = .{ .vfn = vfn, .cache = &cache, .ply = 1, .cutoff = deep3_cutoff, .sample = 2, .sample_min_depth = 2 };
    // a depth-1 chance node sits directly above the leaves; min depth 2 must leave it bit-identical
    const expected = plain.chance(mid_board, 1, 1);
    try std.testing.expectEqual(expected, last_full.chance(mid_board, 1, 1));
    try std.testing.expect(sampled.chance(mid_board, 1, 1) != expected);
    // a depth-2 chance node is inside the sample in both settings, but the layer below it is not
    try std.testing.expect(sampled.chance(mid_board, 2, 1) != plain.chance(mid_board, 2, 1));
    try std.testing.expect(last_full.chance(mid_board, 2, 1) != plain.chance(mid_board, 2, 1));
}

test "sampled 4-ply is deterministic and cheaper (#320)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    var a: Stats = .{};
    var b: Stats = .{};
    var cache_a: ValueCache = .{};
    var cache_b: ValueCache = .{};
    const first = bestMoveWithCacheValueOpts(vfn, late_board, .deep4, &a, &cache_a, .{ .sample = 3 });
    const second = bestMoveWithCacheValueOpts(vfn, late_board, .deep4, &b, &cache_b, .{ .sample = 3 });
    try std.testing.expect(first != null);
    try std.testing.expectEqual(first, second);
    try std.testing.expectEqual(a.leaves, b.leaves);
    // the seed is a real knob: a different one may pick another sample, never an illegal move
    var c: Stats = .{};
    var cache_c: ValueCache = .{};
    const third = bestMoveWithCacheValueOpts(vfn, late_board, .deep4, &c, &cache_c, .{ .sample = 3, .sample_seed = 42 });
    try std.testing.expect(third != null);
    // keeping the last spawn layer full width costs leaves but must stay deterministic too
    var d: Stats = .{};
    var e: Stats = .{};
    var cache_d: ValueCache = .{};
    var cache_e: ValueCache = .{};
    const fourth = bestMoveWithCacheValueOpts(vfn, late_board, .deep4, &d, &cache_d, .{ .sample = 2, .sample_min_depth = 2 });
    const fifth = bestMoveWithCacheValueOpts(vfn, late_board, .deep4, &e, &cache_e, .{ .sample = 2, .sample_min_depth = 2 });
    try std.testing.expectEqual(fourth, fifth);
    try std.testing.expectEqual(d.leaves, e.leaves);
}

test "phase_h4 leaves the earlier schedules alone (#171)" {
    // the phase table is shared, so adding a row must not move where the existing modes switch
    for (0..17) |empties| {
        const board = boardWithEmpties(empties);
        try std.testing.expectEqual(Depth{ .ply = 2, .cutoff = 0 }, phaseDepth(.deep, board));
        try std.testing.expectEqual(Depth{ .ply = 3, .cutoff = deep3_cutoff }, phaseDepth(.deep3, board));
        try std.testing.expectEqual(Depth{ .ply = 3, .cutoff = deep3_cutoff, .recover_cut_branches = true }, phaseDepth(.deep3_recover, board));
        const h1: Depth = if (empties >= 8) .{ .ply = 2, .cutoff = 0 } else .{ .ply = 3, .cutoff = deep3_cutoff };
        try std.testing.expectEqual(h1, phaseDepth(.phase_h1, board));
        const h2: Depth = if (empties >= 8)
            .{ .ply = 2, .cutoff = 0 }
        else if (empties >= 5)
            .{ .ply = 3, .cutoff = deep3_cutoff }
        else
            .{ .ply = 3, .cutoff = phase_tight_cutoff };
        try std.testing.expectEqual(h2, phaseDepth(.phase_h2, board));
        const h3: Depth = if (empties >= 10)
            .{ .ply = 2, .cutoff = 0 }
        else if (empties >= 6)
            .{ .ply = 3, .cutoff = deep3_cutoff }
        else
            .{ .ply = 3, .cutoff = phase_tight_cutoff };
        try std.testing.expectEqual(h3, phaseDepth(.phase_h3, board));
    }
}

test "phase modes are deterministic for a fixed board (#157)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(157);
    const net = Net.init(prng.random());
    for (phase_modes) |mode| {
        var a: Stats = .{};
        var b: Stats = .{};
        const first = bestMoveWithStats(&net, mid_board, mode, &a);
        const second = bestMoveWithStats(&net, mid_board, mode, &b);
        try std.testing.expect(first != null);
        try std.testing.expectEqual(first, second);
        try std.testing.expectEqual(a.leaves, b.leaves);
        try std.testing.expectEqual(a.hits, b.hits);
    }
}

test "value cache keeps one entry per board and search depth (#157)" {
    var prng = std.Random.DefaultPrng.init(8);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    const ply2: Depth = .{ .ply = 2, .cutoff = 0 };
    const ply3: Depth = .{ .ply = 3, .cutoff = deep3_cutoff };
    var cache: ValueCache = .{};
    var stats: Stats = .{};
    var scratch: Cache = .{};
    const plain = forward(&net, sample_board, &scratch);
    // the leaf itself is depth-free, so both depths give the same number ...
    try std.testing.expectEqual(plain, cache.getValueAtDepth(vfn, sample_board, ply2, &stats));
    try std.testing.expectEqual(plain, cache.getValueAtDepth(vfn, sample_board, ply3, &stats));
    // ... but the 3-ply lookup is a separate entry: a miss, not a hit on the 2-ply value
    try std.testing.expectEqual(@as(u64, 2), stats.leaves);
    try std.testing.expectEqual(@as(u64, 0), stats.hits);
    _ = cache.getValueAtDepth(vfn, sample_board, ply3, &stats);
    try std.testing.expectEqual(@as(u64, 2), stats.leaves);
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
}

test "a cache warmed at another depth still gives the fresh move (#157)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(157);
    const net = Net.init(prng.random());
    // mid_board has 6 empty cells, so phase_h1 searches 3-ply here while the warm-up searched 2-ply
    var warm: ValueCache = .{};
    var warm_stats: Stats = .{};
    _ = bestMoveWithCache(&net, mid_board, .deep, &warm_stats, &warm);
    var fresh: ValueCache = .{};
    var fresh_stats: Stats = .{};
    const fresh_move = bestMoveWithCache(&net, mid_board, .phase_h1, &fresh_stats, &fresh);
    const warm_move = bestMoveWithCache(&net, mid_board, .phase_h1, &warm_stats, &warm);
    try std.testing.expect(fresh_move != null);
    try std.testing.expectEqual(fresh_move, warm_move);
}

test "greedy and chance moves are unchanged by the deep mode" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    // recorded on the pre-#69 code: both modes still route to pickMove / chanceMove
    // re-recorded at : the width change (64 -> 128) moves the random init, and with it the
    // directions this pin recorded on `mid_board` (was `.left` for both at 64/32).
    try std.testing.expectEqual(@as(?Direction, .up), bestMove(&net, mid_board, .greedy));
    try std.testing.expectEqual(@as(?Direction, .down), bestMove(&net, mid_board, .chance));
}

/// Full board without any legal move: the 2-ply search has nothing to score.
const dead_board: Board = .{
    .{ 2, 4, 2, 4 },
    .{ 4, 2, 4, 2 },
    .{ 2, 4, 2, 4 },
    .{ 4, 2, 4, 2 },
};

test "deep2 self-play picks the eval's 2-ply move, legally and deterministically (#158)" {
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    var cache_a: ValueCache = .{};
    var cache_b: ValueCache = .{};
    try std.testing.expectEqual(@as(?Pick, null), pickSearch2Value(.{ .single = &net }, .{ .board = dead_board }, &cache_a, false));

    const first = pickSearch2Value(.{ .single = &net }, mid_board, &cache_a, false).?;
    const second = pickSearch2Value(.{ .single = &net }, mid_board, &cache_b, false).?;
    try std.testing.expectEqual(first.dir, second.dir);
    try std.testing.expectEqual(first.after, second.after);
    try std.testing.expectEqual(first.reward, second.reward);
    var moved = mid_board;
    try std.testing.expect(moved.slide(first.dir).changed);
    try std.testing.expectEqual(first.after, moved.board);
    // the self-play policy and the eval policy must look at the same tree, otherwise compares apples to oranges
    try std.testing.expectEqual(@as(?Direction, first.dir), bestMove(&net, mid_board, .deep));
}

/// Post-spawn board: the spawn left a 2 next to a 2, so the next move merges 4 points. A 1-ply value
/// function sees nothing of it, the 2-ply search does.
const post_spawn_board: Board = .{
    .{ 2, 2, 4, 8 },
    .{ 4, 8, 16, 32 },
    .{ 8, 16, 32, 64 },
    .{ 16, 32, 64, 128 },
};

test "search leaf reward can be omitted without changing intermediate or terminal rewards (#260)" {
    var zero_net = std.mem.zeroes(Net);
    const vfn: Value = .{ .single = &zero_net };
    var plain_cache: ValueCache = .{};
    var omit_cache: ValueCache = .{};
    var plain: Search = .{ .vfn = vfn, .cache = &plain_cache, .ply = 1 };
    var omit: Search = .{ .vfn = vfn, .cache = &omit_cache, .ply = 1, .omit_leaf_reward = true };
    const full: GameState = .{ .board = post_spawn_board };

    // At depth 1, the move's afterstate is the leaf: retain the old r + V(A), or use only V(A).
    try std.testing.expectApproxEqAbs(4 / reward_scale, plain.value(full, 1, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), omit.value(full, 1, 1), 1e-12);

    // The top-level root uses the same convention rather than bypassing `Search.value`'s flag.
    var root_plain_cache: ValueCache = .{};
    var root_omit_cache: ValueCache = .{};
    var root_plain: Search = .{ .vfn = vfn, .cache = &root_plain_cache, .ply = 0 };
    var root_omit: Search = .{ .vfn = vfn, .cache = &root_omit_cache, .ply = 0, .omit_leaf_reward = true };
    const order = [_]Direction{ .up, .left, .down, .right };
    try std.testing.expectApproxEqAbs(4 / reward_scale, rootMove(&root_plain, full, order, null).value, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), rootMove(&root_omit, full, order, null).value, 1e-12);

    // A full-board chance node continues at the same depth instead of becoming a direct leaf. At the
    // following depth boundary only the last reward is omitted (the earlier reward remains); a
    // terminal board with no legal move is 0.
    const no_spawn = plain.chanceResult(full, 1, 1);
    try std.testing.expect(!no_spawn.direct_leaf);
    var full_plain_cache: ValueCache = .{};
    var full_omit_cache: ValueCache = .{};
    var full_plain: Search = .{ .vfn = vfn, .cache = &full_plain_cache, .ply = 2 };
    var full_omit: Search = .{ .vfn = vfn, .cache = &full_omit_cache, .ply = 2, .omit_leaf_reward = true };
    try std.testing.expect(full_plain.value(full, 2, 1) > full_omit.value(full, 2, 1));
    try std.testing.expectApproxEqAbs(8 / reward_scale, full_plain.value(full, 2, 1) - full_omit.value(full, 2, 1), 1e-12);
    const dead: GameState = .{ .board = dead_board };
    try std.testing.expectEqual(@as(f64, 0), full_plain.value(dead, 2, 1));
    try std.testing.expectEqual(@as(f64, 0), full_omit.value(dead, 2, 1));

    // At depth 3, cutoff=1 prunes every spawn at the first chance node. That direct leaf follows the
    // root move, so the candidate omits exactly that reward just as at a normal depth boundary.
    var cut_plain_cache: ValueCache = .{};
    var cut_omit_cache: ValueCache = .{};
    var cut_plain: Search = .{ .vfn = vfn, .cache = &cut_plain_cache, .ply = 3, .cutoff = 1 };
    var cut_omit: Search = .{ .vfn = vfn, .cache = &cut_omit_cache, .ply = 3, .cutoff = 1, .omit_leaf_reward = true };
    var cut_state = mid_board;
    cut_state.board[0][1] = 2; // give the open board one known merge while retaining its spawn holes
    var best_reward: f64 = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var moved = cut_state;
        const result = moved.slide(d);
        if (result.changed) best_reward = @max(best_reward, @as(f64, @floatFromInt(result.score_delta)) / reward_scale);
    }
    try std.testing.expect(best_reward > 0);
    try std.testing.expectApproxEqAbs(best_reward, cut_plain.value(cut_state, 3, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), cut_omit.value(cut_state, 3, 1), 1e-12);

    // The disabled option follows the identical search path and preserves the baseline move.
    const weights = try parse(@embedFile("nn_weights.bin"));
    var default_cache: ValueCache = .{};
    var explicit_off_cache: ValueCache = .{};
    var default_stats: Stats = .{};
    var explicit_off_stats: Stats = .{};
    const default_move = bestMoveWithCacheValueOpts(.{ .single = &weights }, mid_board, .deep, &default_stats, &default_cache, .{});
    const explicit_off_move = bestMoveWithCacheValueOpts(.{ .single = &weights }, mid_board, .deep, &explicit_off_stats, &explicit_off_cache, .{ .omit_leaf_reward = false });
    try std.testing.expectEqual(default_move, explicit_off_move);
}

test "search2 bootstraps from the post-spawn board, td1 from V of the next afterstate (#158)" {
    // zero net: V is 0 everywhere, so the td1 bootstrap cannot see a single future reward
    const net = std.mem.zeroes(Net);
    const next: Pick = .{ .dir = .left, .after = post_spawn_board, .reward = 0, .value = 0 };
    const spawned: GameState = .{ .board = post_spawn_board };
    const td1 = tdBootstrap(&net, next, spawned, .{ .target = .td1 }, null);
    const search2 = tdBootstrap(&net, next, spawned, .{ .target = .search2 }, null);
    try std.testing.expectEqual(@as(f64, 0), td1);
    // merging the 2s is 4 points / reward_scale, and the 2-ply search starts on this post-spawn board
    try std.testing.expect(search2 >= 4 / reward_scale);
    // same target on a board with no legal move stays 0: the bootstrap is the search of the given board
    const dead_spawned: GameState = .{ .board = dead_board };
    try std.testing.expectEqual(@as(f64, 0), tdBootstrap(&net, next, dead_spawned, .{ .target = .search2 }, null));
    // with the committed weights the two targets also differ on a midgame board
    const weights = try parse(@embedFile("nn_weights.bin"));
    const trained_next: Pick = .{ .dir = .left, .after = mid_board.board, .reward = 0, .value = valueOf(&weights, mid_board.board) };
    const boot1 = tdBootstrap(&weights, trained_next, mid_board, .{ .target = .td1 }, null);
    const boot2 = tdBootstrap(&weights, trained_next, mid_board, .{ .target = .search2 }, null);
    try std.testing.expectEqual(trained_next.value, boot1);
    try std.testing.expect(@abs(boot2 - boot1) > 1e-6);
}

test "search2 target does not double count the last reward (#158)" {
    // Constant value function (a bias only, every weight 0): every afterstate is worth `v`, so the
    // search collapses to the rewards it sums plus one `v` and the expected target is handwritten.
    // Local caches: the shared one is keyed by board alone, and this test brings its own net.
    var net = std.mem.zeroes(Net);
    const v: f64 = 0.25;
    net.b3 = v;
    const vfn: Value = .{ .single = &net };
    // `post_spawn_board` is full, so nothing spawns before its first move: left and right both merge
    // the 2s for 4 points, up and down do not change the board. That is r_t = 4 / reward_scale, and
    // the bootstrap must be that one reward plus V of the afterstate one move past it, r_t + V(A_{t+1}).
    const spawned: GameState = .{ .board = post_spawn_board };
    const r_t = 4 / reward_scale;
    var cache: ValueCache = .{};
    const boot = search2Value(vfn, spawned, &cache, false);
    try std.testing.expectApproxEqAbs(r_t + v, boot, 1e-12);
    // the target of the step is the pending reward in front of it, r_{t-1} + [r_t + V(A_{t+1})]
    const r_prev = 1 / reward_scale;
    try std.testing.expectApproxEqAbs(r_prev + (r_t + v), r_prev + boot, 1e-12);
    // the pre-fix bootstrap, `Search.value` at the same count, also sums one reward whose afterstate
    // the leaf's V already holds: `V(A_{t+1})` would count r_{t+1} twice. The excess is exactly that
    // extra reward, here the best second move, 8 points (the 4s of `post_spawn_board` merged).
    var old_cache: ValueCache = .{};
    var search: Search = .{ .vfn = vfn, .cache = &old_cache };
    const double_counted = search.value(spawned, search2_ply, 1);
    try std.testing.expectApproxEqAbs(8 / reward_scale, double_counted - boot, 1e-12);

    // zero rewards is the leaf: the best V one move away, with no reward added at it
    const weights = try parse(@embedFile("nn_weights.bin"));
    var weight_cache: ValueCache = .{};
    var boot_search: Search = .{ .vfn = .{ .single = &weights }, .cache = &weight_cache };
    const state: GameState = .{ .board = mid_board.board };
    var best_after: f64 = -std.math.inf(f64);
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var moved = state;
        if (!moved.slide(d).changed) continue;
        best_after = @max(best_after, valueOf(&weights, moved.board));
    }
    try std.testing.expectEqual(best_after, boot_search.bootValue(state, 0, 1));
}

test "search2 target and deep2 pick price with the current weights, not the shared cache (#228)" {
    // A bias-only net is a constant value function, so a target is `search_best_reward + b3` and the
    // difference between two such nets is exactly the bias delta, whatever the board and the spawns.
    var old_net = std.mem.zeroes(Net);
    old_net.b3 = 0.25;
    var new_net = old_net;
    new_net.b3 = 0.5;
    const state: GameState = .{ .board = mid_board.board };

    // `t2`: both calls share one table on purpose. Before the second one answered from the leaves
    // `old_net` had priced, so `after - before` was 0 instead of the bias delta.
    var cache: ValueCache = .{};
    const before = search2Value(.{ .single = &old_net }, state, &cache, false);
    const after = search2Value(.{ .single = &new_net }, state, &cache, false);
    try std.testing.expectApproxEqAbs(0.25, after - before, 1e-12);

    // `p2`: the self-play pick walks the same path (`pickSearch2Value` -> `pickTeacherValue`) on the
    // thread-local table, and a warm table must not change the Pick it returns.
    const shared = sharedValueCache();
    shared.clear();
    const old_pick = pickSearch2Value(.{ .single = &old_net }, state, shared, false).?;
    const new_pick = pickSearch2Value(.{ .single = &new_net }, state, shared, false).?;
    shared.clear();
    const fresh_pick = pickSearch2Value(.{ .single = &new_net }, state, shared, false).?;
    try std.testing.expectEqual(fresh_pick.dir, new_pick.dir);
    try std.testing.expectEqual(fresh_pick.value, new_pick.value);
    try std.testing.expectApproxEqAbs(0.25, new_pick.value - old_pick.value, 1e-12);
}

test "teacher target equals the mode's search value (#167)" {
    if (!width.slow_tests) return error.SkipZigTest;
    // `teacherValue` clears the thread-local cache at entry, so a leaf another test left in it
    // cannot answer for this test's net; the reset below is belt and braces.
    sharedValueCache().* = .{};
    const weights = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &weights };
    const next: Pick = .{ .dir = .left, .after = mid_board.board, .reward = 0, .value = valueOf(&weights, mid_board.board) };
    const state: GameState = .{ .board = mid_board.board };

    // `.deep` is the (ply 2, full width) row: the tree `search2Value` bootstraps, `Search.value` at 2
    var deep_cache: ValueCache = .{};
    var deep_search: Search = .{ .vfn = vfn, .cache = &deep_cache, .cutoff = 0, .ply = 2 };
    const want_deep = deep_search.value(state, 2, 1);
    try std.testing.expectEqual(want_deep, tdBootstrap(&weights, next, state, .{ .teacher = .deep }, null));
    // one own move deeper than `V` of the afterstate: the teacher target carries a searched reward
    try std.testing.expect(@abs(want_deep - next.value) > 1e-6);

    // `.deep3` is the (ply 3, `deep3_cutoff`) row, which is the tree `Mode.deep` ranks its moves with
    var deep3_cache: ValueCache = .{};
    var deep3_search: Search = .{ .vfn = vfn, .cache = &deep3_cache, .cutoff = deep3_cutoff, .ply = 3 };
    const want_deep3 = deep3_search.value(state, 3, 1);
    try std.testing.expectEqual(want_deep3, tdBootstrap(&weights, next, state, .{ .teacher = .deep3 }, null));
    try std.testing.expect(std.math.isFinite(want_deep3) and want_deep3 != want_deep);

    // the teacher owns the bootstrap: `target` is not consulted any more (a policy/target mismatch is
    // exactly the failure this cell exists to avoid)
    try std.testing.expectEqual(want_deep, tdBootstrap(&weights, next, state, .{ .teacher = .deep, .target = .search2 }, null));
    try std.testing.expectEqual(want_deep3, tdBootstrap(&weights, next, state, .{ .teacher = .deep3, .target = .td1 }, null));

    // no legal move means no search value, whichever teacher asks
    const dead: GameState = .{ .board = dead_board };
    try std.testing.expectEqual(@as(f64, 0), tdBootstrap(&weights, next, dead, .{ .teacher = .deep }, null));
    try std.testing.expectEqual(@as(f64, 0), tdBootstrap(&weights, next, dead, .{ .teacher = .deep3 }, null));
}

/// Board hashes of the first `teacher_trace_moves` moves of a game under `opts`, spawned from one rng
/// exactly as the per-move trainer does. Equal traces = the same tiles in the same cells in the same
/// order, so equal spawn sequences.
const teacher_trace_moves = 6;

fn spawnTrace(net: *const Net, seed: u64, opts: Options) [teacher_trace_moves]u64 {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    var state = GameState.gameStart(random);
    var trace: [teacher_trace_moves]u64 = @splat(0);
    for (&trace) |*hash| {
        const pick = selfplayPick(.{ .single = net }, state, false, opts.selfplay, opts.teacher, opts.sym_leaf) orelse break;
        _ = state.move(pick.dir, random);
        hash.* = std.hash.Crc32.hash(std.mem.asBytes(&state.board));
    }
    return trace;
}

/// One game of `opts` from a copy of `net` with lr = 0, so the net (and with it the move the policy
/// picks) stays fixed: the game then depends only on the policy and the spawns.
fn frozenGame(net: *const Net, seed: u64, opts: Options) GameResult {
    var copy = net.*;
    var rng = std.Random.DefaultPrng.init(seed);
    return trainGameOpts(&copy, rng.random(), 0, null, opts) catch unreachable;
}

test "teacher self-play matches the deep2 spawn sequence (#167)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(167);
    const net = Net.init(prng.random());
    // `.deep` is `pickSearch2Value`'s own tree (`pickTeacherValue(.., .deep)`), so the teacher policy
    // makes the `.deep2` policy's moves, and since neither search draws from the rng both runs spawn
    // the same tiles in the same order for the same seed.
    try std.testing.expectEqual(
        spawnTrace(&net, 167, .{ .selfplay = .deep2 }),
        spawnTrace(&net, 167, .{ .teacher = .deep }),
    );
    // same for the whole game: with lr = 0 the net does not move, so the move count and the score have
    // to agree too — only the target the run computes differs (that is the point of the cell)
    const deep2 = frozenGame(&net, 167, .{ .selfplay = .deep2 });
    const teacher = frozenGame(&net, 167, .{ .teacher = .deep });
    try std.testing.expect(deep2.moves > 0);
    try std.testing.expectEqual(deep2.moves, teacher.moves);
    try std.testing.expectEqual(deep2.score, teacher.score);
    // `.deep3` plays the 3-ply row, so its moves differ from `.deep2`'s, but it spawns from the same rng
    // draws: the trace is populated and stays within one tile per move
    const deep3 = spawnTrace(&net, 167, .{ .teacher = .deep3 });
    try std.testing.expect(deep3[teacher_trace_moves - 1] != 0);
}

test "value cache returns the forward value and counts leaf evaluations" {
    var prng = std.Random.DefaultPrng.init(8);
    const net = Net.init(prng.random());
    var cache: ValueCache = .{};
    var stats: Stats = .{};
    var scratch: Cache = .{};
    const plain = forward(&net, sample_board, &scratch);
    try std.testing.expectEqual(plain, cache.get(&net, sample_board, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.leaves);
    try std.testing.expectEqual(@as(u64, 0), stats.hits);
    try std.testing.expectEqual(plain, cache.get(&net, sample_board, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.leaves);
    try std.testing.expectEqual(@as(u64, 1), stats.hits);
}

test "reused value cache returns the same moves as a fresh cache (#88)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(6);
    const net = Net.init(prng.random());
    var warm: ValueCache = .{};
    var play_rng = std.Random.DefaultPrng.init(2048);
    const rng = play_rng.random();
    var state = GameState.gameStart(rng);
    for (0..30) |_| {
        var fresh: ValueCache = .{};
        var warm_stats: Stats = .{};
        var fresh_stats: Stats = .{};
        const fresh_move = bestMoveWithCache(&net, state, .deep, &fresh_stats, &fresh);
        const warm_move = bestMoveWithCache(&net, state, .deep, &warm_stats, &warm);
        try std.testing.expectEqual(fresh_move, warm_move);
        const dir = warm_move orelse break;
        _ = state.move(dir, rng);
    }
    // a repeated search on one board is mostly served from the warm workspace (the few
    // recomputes are the entries a slot collision evicted inside the first search)
    var once: Stats = .{};
    const once_move = bestMoveWithCache(&net, mid_board, .deep, &once, &warm);
    var twice: Stats = .{};
    const twice_move = bestMoveWithCache(&net, mid_board, .deep, &twice, &warm);
    try std.testing.expectEqual(once_move, twice_move);
    try std.testing.expect(once.leaves > 0);
    try std.testing.expect(twice.hits > 0);
    try std.testing.expect(twice.leaves < once.leaves);
}

test "value cache slot collision drops the entry, never returns a wrong value (#88)" {
    var prng = std.Random.DefaultPrng.init(8);
    const net = Net.init(prng.random());
    // deterministic pair of distinct boards (valid power-of-two tiles) sharing one cache slot
    var boards: [512]Board = undefined;
    for (&boards, 0..) |*board, i| {
        board.* = @splat(@splat(0));
        board.*[0][0] = std.math.shl(core.Tile, 1, 1 + i % 15);
        board.*[0][1] = std.math.shl(core.Tile, 1, 1 + (i / 15) % 15);
        board.*[3][3] = std.math.shl(core.Tile, 1, 1 + (i / 225) % 15);
    }
    var pair: ?[2]Board = null;
    search: for (0..boards.len) |i| {
        for (i + 1..boards.len) |j| {
            const ka = ValueCache.keyOfDepth(boards[i], leaf_depth);
            const kb = ValueCache.keyOfDepth(boards[j], leaf_depth);
            if (ka != kb and (ka & (ValueCache.capacity - 1)) == (kb & (ValueCache.capacity - 1))) {
                pair = .{ boards[i], boards[j] };
                break :search;
            }
        }
    }
    try std.testing.expect(pair != null);
    const a_board = pair.?[0];
    const b_board = pair.?[1];
    var cache: ValueCache = .{};
    var stats: Stats = .{};
    var scratch: Cache = .{};
    const va = forward(&net, a_board, &scratch);
    const vb = forward(&net, b_board, &scratch);
    try std.testing.expectEqual(va, cache.get(&net, a_board, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.leaves);
    // the colliding slot must be treated as a miss: a fresh evaluation, never board a's value
    try std.testing.expectEqual(vb, cache.get(&net, b_board, &stats));
    try std.testing.expectEqual(@as(u64, 2), stats.leaves);
    // board a's entry was dropped by the collision and is recomputed correctly
    try std.testing.expectEqual(va, cache.get(&net, a_board, &stats));
    try std.testing.expectEqual(@as(u64, 3), stats.leaves);
}

test "trainGame is deterministic for a fixed seed" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&a, rng_a.random(), 0.001, null);
    _ = trainGame(&b, rng_b.random(), 0.001, null);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

test "death reward is the scaled -1 per move instead of the score delta (#101)" {
    var net = std.mem.zeroes(Net);
    const state = GameState{ .board = .{ .{ 2, 2, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } } };
    const scored = pickMove(&net, state, false).?;
    try std.testing.expectEqual(Direction.left, scored.dir);
    try std.testing.expectApproxEqAbs(@as(f64, 4) / reward_scale, scored.reward, 1e-12);
    const death = pickMove(&net, state, true).?;
    try std.testing.expectEqual(@as(f64, -1) / reward_scale, death.reward);
}

test "death training drives values toward minus moves to game over (#101)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var train_rng = std.Random.DefaultPrng.init(12);
    // 200 games at the 128/32 default: at 64 games the values had barely moved off 0
    for (0..200) |_| _ = try trainGameOpts(&net, train_rng.random(), 0.02, null, .{ .death = true });

    var play_rng = std.Random.DefaultPrng.init(13);
    const rng = play_rng.random();
    var state = GameState.gameStart(rng);
    var first_value: f64 = 0;
    var last_value: f64 = 0;
    var moves: u32 = 0;
    while (pickMove(&net, state, true)) |p| {
        if (moves == 0) first_value = p.value;
        last_value = p.value;
        moves += 1;
        _ = state.move(p.dir, rng);
    }
    try std.testing.expect(moves > 10);
    try std.testing.expect(first_value < last_value);
    try std.testing.expect(last_value < 0);
}

/// Mean and standard error live in `stats.zig` so the n-tuple runner and the shared eval row use
/// the same arithmetic.
pub const MeanErr = eval_stats.MeanErr;
pub const meanStderr = eval_stats.meanStderr;

/// Linear correlation of two equal-length samples; 0 when either is constant.
pub fn pearson(xs: []const f64, ys: []const f64) f64 {
    std.debug.assert(xs.len == ys.len);
    if (xs.len < 2) return 0;
    const n: f64 = @floatFromInt(xs.len);
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (xs, ys) |x, y| {
        sx += x;
        sy += y;
    }
    const mx = sx / n;
    const my = sy / n;
    var cov: f64 = 0;
    var vx: f64 = 0;
    var vy: f64 = 0;
    for (xs, ys) |x, y| {
        const dx = x - mx;
        const dy = y - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if (vx == 0 or vy == 0) return 0;
    return cov / @sqrt(vx * vy);
}

test "pearson gives the linear correlation of two samples (#101)" {
    const xs = [_]f64{ 1, 2, 3, 4 };
    const ys = [_]f64{ 2, 4, 6, 8 };
    try std.testing.expectApproxEqAbs(@as(f64, 1), pearson(&xs, &ys), 1e-12);
    const down = [_]f64{ 8, 6, 4, 2 };
    try std.testing.expectApproxEqAbs(@as(f64, -1), pearson(&xs, &down), 1e-12);
    const steady = [_]f64{ 3, 3, 3, 3 };
    try std.testing.expectEqual(@as(f64, 0), pearson(&xs, &steady));
}

test "weight file header round trips" {
    var prng = std.Random.DefaultPrng.init(21);
    const net = Net.init(prng.random());
    const bytes = encode(&net);
    try std.testing.expectEqualStrings("2048-NN\x00", bytes[0..8]);
    const back = try parse(&bytes);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&net), std.mem.asBytes(&back));
}

test "parse rejects bad magic" {
    var prng = std.Random.DefaultPrng.init(21);
    var bytes = encode(&Net.init(prng.random()));
    bytes[0] ^= 0xff;
    try std.testing.expectError(error.BadWeightMagic, parse(&bytes));
}

test "parse rejects wrong version" {
    var prng = std.Random.DefaultPrng.init(21);
    var bytes = encode(&Net.init(prng.random()));
    std.mem.writeInt(u32, bytes[8..12], 2, .little);
    try std.testing.expectError(error.UnsupportedWeightVersion, parse(&bytes));
}

test "parse rejects wrong endianness marker" {
    var prng = std.Random.DefaultPrng.init(21);
    var bytes = encode(&Net.init(prng.random()));
    std.mem.writeInt(u32, bytes[12..16], 0x04030201, .little);
    try std.testing.expectError(error.WrongWeightEndianness, parse(&bytes));
}

test "parse rejects wrong size" {
    var prng = std.Random.DefaultPrng.init(21);
    const bytes = encode(&Net.init(prng.random()));
    try std.testing.expectError(error.WrongWeightFileSize, parse(bytes[0 .. bytes.len - 1]));
    try std.testing.expectError(error.WrongWeightFileSize, parse(&bytes ++ [_]u8{0}));
}

test "embedded weights file parses" {
    _ = try parse(@embedFile("nn_weights.bin"));
}

test "quantized file is 48 B of header plus 2 B per weight (#170)" {
    // 37,057 = 256*128 + 128 + 32*128 + 32 + 32 + 1: the 128/32 shipped net of (18,561 at 64/32)
    try std.testing.expectEqual(@as(usize, 37_057), weight_count);
    try std.testing.expectEqual(weight_count * @sizeOf(i16), @sizeOf(QPayload));
    try std.testing.expectEqual(@as(usize, 48), quant_header_len);
    try std.testing.expectEqual(@as(usize, 74_162), quant_file_len);
    // the i16 payload indexes the same element offsets as `Net`'s fields
    for (quant_blocks, 0..) |block, b| {
        if (b > 0) try std.testing.expectEqual(quant_blocks[b - 1].offset + quant_blocks[b - 1].len, block.offset);
    }
}

test "encodeQuantized writes the int16 weight layout byte for byte (#302)" {
    var q = Quantized{};
    for (&q.steps, 0..) |*step, b| {
        const scale: f64 = @floatFromInt(@as(u32, 1) << @as(u5, @intCast(b)));
        step.* = 0.5 * scale;
    }
    q.a_steps = .{ 32, 64 };
    for (std.mem.bytesAsSlice(i16, std.mem.asBytes(&q.q)), 0..) |*value, i| {
        value.* = @intCast(@as(i32, @intCast(i % 2001)) - 1000);
    }

    const bytes = encodeQuantized(&q);
    // magic, version, endianness marker, the six block steps (0.5, 1, 2, 4, 8, 16) and a1/a2 (32, 64)
    try std.testing.expectEqualSlices(u8, &[_]u8{
        '2',  '0',  '4',  '8',  '-',  'N',  'Q',  '1',
        1,    0,    0,    0,    4,    3,    2,    1,
        0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40,
        0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x80, 0x41,
        0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x80, 0x42,
    }, bytes[0..quant_header_len]);
    // the payload starts with -1000, then -999, as i16 little endian
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x18, 0xfc, 0x19, 0xfc, 0x1a, 0xfc, 0x1b, 0xfc }, bytes[quant_header_len..][0..8]);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings("a6660a0c97dd859eeab2500cf2008903f118e64d8e63e14e72653f1bb332e8b2", &hex);
}

test "the f32 file is half the f64 block and round-trips every weight (#348)" {
    // 37,057 weights at 4 B instead of 8 B, plus the same 16 B header.
    try std.testing.expectEqual(@as(usize, 37_057 * @sizeOf(f32) + 16), f32_file_len);
    // same header, half the payload: 16 + 4 B/weight here, 16 + 8 B/weight in `Net`.
    try std.testing.expectEqual(header_len + @sizeOf(Net), file_len);
    try std.testing.expectEqual(header_len + @sizeOf(Net32), f32_file_len);
    // 8 B per weight in `Net` become 4 B in `Net32`, so the payload halves exactly.
    try std.testing.expectEqual(header_len + @sizeOf(Net), 2 * (header_len + @sizeOf(Net32)) - header_len);

    const net = try parse(@embedFile("nn_weights.bin"));
    const narrowed = toF32(&net);
    const back = try decodeF32(&encodeF32(&narrowed));
    try std.testing.expectEqualSlices(
        f32,
        std.mem.bytesAsSlice(f32, std.mem.asBytes(&narrowed)),
        std.mem.bytesAsSlice(f32, std.mem.asBytes(&back)),
    );

    // Narrowing is the only loss: every weight is the f64 one rounded to f32.
    const wide_values = flatConst(&net);
    const narrow_values = std.mem.bytesAsSlice(f32, std.mem.asBytes(&back));
    try std.testing.expectEqual(wide_values.len, narrow_values.len);
    for (wide_values, narrow_values) |w, n| {
        try std.testing.expectEqual(@as(f32, @floatCast(w)), n);
    }
}

test "forwardF32 agrees with forward on every board of a game (#348)" {
    const net = try parse(@embedFile("nn_weights.bin"));
    const net32 = toF32(&net);
    var prng = std.Random.DefaultPrng.init(9000);
    const rng = prng.random();
    var cache: Cache = .{};
    var worst: f64 = 0;
    var boards: usize = 0;
    for (0..2) |_| {
        var state = core.GameState.gameStart(rng);
        while (!state.isGameOver()) {
            const want = forward(&net, state.board, &cache);
            const got = forwardF32(&net32, state.board);
            // f32 sums 256 row additions and a 128-term dot; 1e-3 is 1e-4 of a typical value.
            worst = @max(worst, @abs(got - want));
            if (comptime width.simd) {
                // The f32x4 lanes keep separate partial sums, so the vector and scalar f32 paths
                // differ by rounding only.
                worst = @max(worst, @abs(forwardF32Impl(&net32, state.board, false) - got));
            }
            const dir = bestMoveValue(.{ .single = &net }, state, .greedy) orelse break;
            _ = state.move(dir, rng);
            boards += 1;
        }
    }
    try std.testing.expect(boards > 100);
    try std.testing.expect(worst < 1e-3);
}

test "quantized round trip keeps every weight within its block step (#170)" {
    const net = try parse(@embedFile("nn_weights.bin"));
    const q = quantize(&net);
    const back = try parseQuantized(&encodeQuantized(&q));
    var requantized: Net = undefined;
    dequantize(&back, &requantized);

    const before = flatConst(&net);
    const after = flatConst(&requantized);
    for (quant_blocks, 0..) |block, b| {
        // the step the block was quantized with, recomputed from the f32 weights it came from
        const step = blockStep(before[block.offset..][0..block.len]);
        try std.testing.expectApproxEqRel(step, back.steps[b], 1e-6);
        for (block.offset..block.offset + block.len) |i| {
            const delta = @abs(before[i] - after[i]);
            try std.testing.expect(delta <= step); // max|w| / 32767: the contract of the format
            // what round actually costs. The step is taken from the f32 weights, so the observed
            // ratio reaches 0.5003 x step; the slack is f32 round-off, not a broken contract.
            try std.testing.expect(delta <= step / 2 + step * 1e-3);
        }
    }
    // the activation steps are real steps, not the zero default
    try std.testing.expect(back.a_steps[0] > 0 and back.a_steps[1] > 0);
}

test "quantized inference of a constant net matches the f32 net within 1e-2 (#170)" {
    var net = std.mem.zeroes(Net);
    for (&net.w1, 0..) |*row, r| for (row, 0..) |*w, i| {
        w.* = @as(f64, @floatFromInt((r * 7 + i * 3) % 11)) * 0.01 - 0.05;
    };
    for (&net.b1, 0..) |*b, i| b.* = @as(f64, @floatFromInt(i % 5)) * 0.02 - 0.04;
    for (&net.w2, 0..) |*row, j| for (row, 0..) |*w, i| {
        w.* = @as(f64, @floatFromInt((j * 5 + i) % 7)) * 0.02 - 0.06;
    };
    for (&net.b2, 0..) |*b, j| b.* = @as(f64, @floatFromInt(j % 3)) * 0.03;
    for (&net.w3, 0..) |*w, j| w.* = @as(f64, @floatFromInt(j % 4)) * 0.05 - 0.07;
    net.b3 = 0.25;
    const q = quantize(&net);

    const empty: Board = @splat(@splat(0));
    var cache: Cache = .{};
    try std.testing.expectApproxEqAbs(forward(&net, empty, &cache), forwardQuantized(&q, empty), 1e-2);

    var board: Board = @splat(@splat(0));
    board[0][0] = 2;
    board[0][1] = 4;
    board[1][0] = 8;
    board[2][2] = 256;
    board[3][3] = 1024;
    try std.testing.expectApproxEqAbs(forward(&net, board, &cache), forwardQuantized(&q, board), 1e-2);
}

test "parseQuantized rejects a foreign magic, another version and a short file (#170)" {
    var prng = std.Random.DefaultPrng.init(7);
    const q = quantize(&Net.init(prng.random()));
    const full = encodeQuantized(&q);

    var bytes = full;
    bytes[0] ^= 0xff;
    try std.testing.expectError(error.BadQuantMagic, parseQuantized(&bytes));

    bytes = full;
    std.mem.writeInt(u32, bytes[8..12], quant_version + 1, .little);
    try std.testing.expectError(error.UnsupportedQuantVersion, parseQuantized(&bytes));

    bytes = full;
    std.mem.writeInt(u32, bytes[12..16], 0x04030201, .little);
    try std.testing.expectError(error.WrongQuantEndianness, parseQuantized(&bytes));

    try std.testing.expectError(error.WrongQuantFileSize, parseQuantized(full[0 .. quant_file_len - 1]));
    // the f32 weights are not a quantized file: the length alone rules them out
    try std.testing.expectError(error.WrongQuantFileSize, parseQuantized(@embedFile("nn_weights.bin")));
}

test "emptyBucket splits empty-cell counts into 0..3, 4..7, 8+" {
    try std.testing.expectEqual(@as(usize, 0), emptyBucket(0));
    try std.testing.expectEqual(@as(usize, 0), emptyBucket(3));
    try std.testing.expectEqual(@as(usize, 1), emptyBucket(4));
    try std.testing.expectEqual(@as(usize, 1), emptyBucket(7));
    try std.testing.expectEqual(@as(usize, 2), emptyBucket(8));
    try std.testing.expectEqual(@as(usize, 2), emptyBucket(16));
}

test "distBucket splits moves left into last 10, 11..30, earlier" {
    try std.testing.expectEqual(@as(usize, 0), distBucket(0));
    try std.testing.expectEqual(@as(usize, 0), distBucket(9));
    try std.testing.expectEqual(@as(usize, 1), distBucket(10));
    try std.testing.expectEqual(@as(usize, 1), distBucket(29));
    try std.testing.expectEqual(@as(usize, 2), distBucket(30));
    try std.testing.expectEqual(@as(usize, 2), distBucket(500));
}

test "Diag.add accumulates count, abs error and signed drop per bucket" {
    var diag: Diag = .{};
    diag.add(2, 5, 0.5, -0.25);
    diag.add(3, 7, 0.25, 0.25);
    diag.add(9, 5, 1.0, 0);
    const near = diag.stats[0][0];
    try std.testing.expectEqual(@as(u64, 2), near.count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), near.abs_err, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), near.drop, 1e-12);
    try std.testing.expectEqual(@as(u64, 1), diag.stats[2][0].count);
    try std.testing.expectEqual(@as(u64, 0), diag.stats[1][1].count);
}

test "Diag push and finish bucket each step by moves to game over" {
    var diag: Diag = .{};
    // 35 steps in one game: step i has 34 - i moves left
    for (0..35) |_| diag.push(boardWithEmpties(0), 1.0, 0);
    diag.finish();
    try std.testing.expectEqual(@as(u64, 10), diag.stats[0][0].count);
    try std.testing.expectEqual(@as(u64, 20), diag.stats[0][1].count);
    try std.testing.expectEqual(@as(u64, 5), diag.stats[0][2].count);
    try std.testing.expectApproxEqAbs(@as(f64, 35), diag.stats[0][0].abs_err + diag.stats[0][1].abs_err + diag.stats[0][2].abs_err, 1e-12);
}

test "tileBucket splits the max tile into <=256, 512, 1024, 2048, 4096 and 8192+" {
    try std.testing.expectEqual(@as(usize, 0), tileBucket(0));
    try std.testing.expectEqual(@as(usize, 0), tileBucket(256));
    try std.testing.expectEqual(@as(usize, 1), tileBucket(512));
    try std.testing.expectEqual(@as(usize, 2), tileBucket(1024));
    try std.testing.expectEqual(@as(usize, 3), tileBucket(2048));
    try std.testing.expectEqual(@as(usize, 4), tileBucket(4096));
    try std.testing.expectEqual(@as(usize, 5), tileBucket(8192));
    try std.testing.expectEqual(@as(usize, 5), tileBucket(65536));
}

test "Diag tile table counts steps and |err| per max tile bucket" {
    var diag: Diag = .{};
    var small: Board = @splat(@splat(0));
    small[0][0] = 256;
    var mid: Board = @splat(@splat(0));
    mid[3][2] = 4096;
    var deep: Board = @splat(@splat(0));
    deep[1][1] = 8192;
    // hand-computed: two steps at 256 (errors 0.5 + 0.25), one at 4096 (1.0), three at 8192+ (0.1 + 0.2 + 0.3)
    diag.push(small, 0.5, 0);
    diag.push(small, 0.25, 0);
    diag.push(mid, 1.0, 0);
    diag.push(deep, 0.1, 0);
    diag.push(deep, 0.2, 0);
    diag.push(deep, 0.3, 0);
    try std.testing.expectEqual(@as(u64, 2), diag.tiles[0].count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), diag.tiles[0].abs_err, 1e-12);
    try std.testing.expectEqual(@as(u64, 0), diag.tiles[3].count);
    try std.testing.expectEqual(@as(u64, 1), diag.tiles[4].count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), diag.tiles[4].abs_err, 1e-12);
    try std.testing.expectEqual(@as(u64, 3), diag.tiles[5].count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), diag.tiles[5].abs_err, 1e-12);
    // the empty-cell table still sees the same six steps (15 empties each, no finish yet)
    try std.testing.expectEqual(@as(u64, 0), diag.stats[2][0].count);
    diag.finish();
    try std.testing.expectEqual(@as(u64, 6), diag.stats[2][0].count);
}

test "Dump keeps only afterstates at or above dump_min_tile" {
    var dump: Dump = .{};
    var shallow: Board = @splat(@splat(0));
    shallow[0][0] = 2048;
    var deep: Board = @splat(@splat(0));
    deep[2][1] = dump_min_tile;
    dump.offer(shallow);
    try std.testing.expectEqual(@as(usize, 0), dump.len);
    dump.offer(deep);
    try std.testing.expectEqual(@as(usize, 1), dump.len);
    try std.testing.expectEqual(deep, dump.boardAt(0));
}

test "Dump writes the header and log2 cells, and load reads them back byte for byte" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/dump.bin", .{tmp.sub_path});
    defer std.testing.allocator.free(path);

    var dump: Dump = .{};
    // board 0: 4096 at (0,0), empty elsewhere except a 2 at (3,3)
    var first: Board = @splat(@splat(0));
    first[0][0] = 4096;
    first[3][3] = 2;
    // board 1: 8192 at (1,2) and 128 at (2,0)
    var second: Board = @splat(@splat(0));
    second[1][2] = 8192;
    second[2][0] = 128;
    dump.offer(first);
    dump.offer(second);
    try dump.save(io, std.testing.allocator, path);

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(dump_header_len + 2 * dump_entry_len, bytes.len);
    try std.testing.expectEqualStrings(dump_magic, bytes[0..8]);
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, bytes[8..12], .little));
    // log2 of the tile, row-major: 4096 -> 12 at cell 0, 2 -> 1 at cell 15
    try std.testing.expectEqual(@as(u8, 12), bytes[dump_header_len + 0]);
    try std.testing.expectEqual(@as(u8, 0), bytes[dump_header_len + 1]);
    try std.testing.expectEqual(@as(u8, 1), bytes[dump_header_len + 15]);
    // 8192 -> 13 at cell 6 (1,2), 128 -> 7 at cell 8 (2,0)
    try std.testing.expectEqual(@as(u8, 13), bytes[dump_header_len + 16 + 6]);
    try std.testing.expectEqual(@as(u8, 7), bytes[dump_header_len + 16 + 8]);

    var loaded: Dump = .{};
    try loaded.load(io, std.testing.allocator, path);
    try std.testing.expectEqual(@as(usize, 2), loaded.len);
    try std.testing.expectEqual(@as(u64, 0), loaded.dropped);
    try std.testing.expectEqual(first, loaded.boardAt(0));
    try std.testing.expectEqual(second, loaded.boardAt(1));
}

test "Dump encodes the fixed boards to the same bytes as before the refactor" {
    var dump: Dump = .{};
    var first: Board = @splat(@splat(0));
    first[0][0] = 4096;
    first[3][3] = 2;
    var second: Board = @splat(@splat(0));
    second[1][2] = 8192;
    second[2][0] = 128;
    dump.append(first);
    dump.append(second);

    // `2048-DMP`, count 2, then 12 at cell 0 and 1 at cell 15 of the first board, 13 at cell 6 and
    // 7 at cell 8 of the second (log2, row-major) — recorded before the cursor refactor.
    const golden = [_]u8{
        0x32, 0x30, 0x34, 0x38, 0x2d, 0x44, 0x4d, 0x50, 0x02, 0x00, 0x00, 0x00,
        0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00,
        0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    var bytes: [golden.len]u8 = undefined;
    dump.encodeInto(&bytes);
    try std.testing.expectEqualSlices(u8, &golden, &bytes);
}

test "encodeBoard and decodeBoard round-trip every tile a board can hold" {
    var full: Board = undefined;
    for (0..core.BoardCellCount) |cell| {
        full[cell / core.BoardSize][cell % core.BoardSize] = @as(core.Tile, 2) << @intCast(cell);
    }
    try std.testing.expectEqual(full, decodeBoard(&encodeBoard(full)));

    var holes: Board = @splat(@splat(0));
    holes[0][0] = 1 << 17;
    holes[1][2] = 8192;
    holes[2][0] = 128;
    holes[3][3] = 2;
    try std.testing.expectEqual(holes, decodeBoard(&encodeBoard(holes)));
}

test "Dump parse rejects a wrong magic and a truncated file" {
    var dump: Dump = .{};
    var bytes: [dump_header_len]u8 = @splat(0);
    @memcpy(bytes[0..8], "2048-XXX");
    try std.testing.expectError(error.BadDumpMagic, dump.parse(&bytes));
    @memcpy(bytes[0..8], dump_magic);
    std.mem.writeInt(u32, bytes[8..12], 1, .little);
    try std.testing.expectError(error.WrongDumpFileSize, dump.parse(&bytes));
    try std.testing.expectError(error.DumpTooShort, dump.parse("2048"));
    std.mem.writeInt(u32, bytes[8..12], @intCast(dump_cap + 1), .little);
    try std.testing.expectError(error.DumpTooLarge, dump.parse(&bytes));
}

test "dump_record leaves the weights and the rng consumption unchanged" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    const plain = try trainGameOpts(&a, rng_a.random(), 0.003, null, .{});
    var dump: Dump = .{};
    var c = b;
    var rng_c = std.Random.DefaultPrng.init(12);
    const recorded = try trainGameOpts(&b, rng_b.random(), 0.003, null, .{ .dump_record = true, .dump = &dump });
    const again = try trainGameOpts(&c, rng_c.random(), 0.003, null, .{ .dump_record = true, .dump = &dump });
    try std.testing.expectEqual(plain.score, recorded.score);
    try std.testing.expectEqual(plain.moves, recorded.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    // recording is a pure side effect: a second identical game overwrites the same entries in the same order
    try std.testing.expectEqual(recorded.score, again.score);
    try std.testing.expectEqual(recorded.moves, again.moves);
    // a fresh net stays shallow, so most games record nothing; whatever it does record must be deep enough
    for (0..dump.len) |i| try std.testing.expect(core.maxTile(dump.boardAt(i)) >= dump_min_tile);
}

test "explore plays random legal moves and explore 0 keeps the greedy game (#224)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(21);
    const start = Net.init(init_rng.random());
    var plain_net = start;
    var plain_rng = std.Random.DefaultPrng.init(22);
    const plain = try trainGameOpts(&plain_net, plain_rng.random(), 0.003, null, .{});

    var zero_net = start;
    var zero_rng = std.Random.DefaultPrng.init(22);
    const zero = try trainGameOpts(&zero_net, zero_rng.random(), 0.003, null, .{ .explore = 0 });
    try std.testing.expectEqual(plain.score, zero.score);
    try std.testing.expectEqual(plain.moves, zero.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain_net), std.mem.asBytes(&zero_net));

    // explore = 1 never follows the policy, so the walk leaves the greedy game
    var wild_net = start;
    var wild_rng = std.Random.DefaultPrng.init(22);
    const wild = try trainGameOpts(&wild_net, wild_rng.random(), 0.003, null, .{ .explore = 1 });
    try std.testing.expect(wild.moves != plain.moves or wild.score != plain.score);
    try std.testing.expect(wild.moves > 0);

    // the random moves come from the game rng, so the same seed replays the same game
    var again_net = start;
    var again_rng = std.Random.DefaultPrng.init(22);
    const again = try trainGameOpts(&again_net, again_rng.random(), 0.003, null, .{ .explore = 1 });
    try std.testing.expectEqual(wild.moves, again.moves);
    try std.testing.expectEqual(wild.score, again.score);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&wild_net), std.mem.asBytes(&again_net));
}

test "warm_open skips training for the random opening and still trains the rest (#224)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(31);
    const start = Net.init(init_rng.random());
    var plain_net = start;
    var plain_rng = std.Random.DefaultPrng.init(32);
    const plain = try trainGameOpts(&plain_net, plain_rng.random(), 0.003, null, .{});

    var warm_net = start;
    var warm_rng = std.Random.DefaultPrng.init(32);
    const warm = try trainGameOpts(&warm_net, warm_rng.random(), 0.003, null, .{ .warm_open = 8 });
    try std.testing.expect(warm.moves != plain.moves or warm.score != plain.score);
    // the opening trains nothing, the greedy tail does: the net still moves away from `start`
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&start), std.mem.asBytes(&warm_net)));

    var again_net = start;
    var again_rng = std.Random.DefaultPrng.init(32);
    const again = try trainGameOpts(&again_net, again_rng.random(), 0.003, null, .{ .warm_open = 8 });
    try std.testing.expectEqual(warm.moves, again.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&warm_net), std.mem.asBytes(&again_net));
}

test "ft2 mixes one deep draw per self-play step and is deterministic" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var plain_net = a;
    var dump: Dump = .{};
    var first: Board = @splat(@splat(0));
    first[0][0] = dump_min_tile;
    first[0][1] = 4;
    var second: Board = @splat(@splat(0));
    second[2][2] = dump_min_tile * 2;
    second[3][0] = 2;
    dump.append(first);
    dump.append(second);

    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_plain = std.Random.DefaultPrng.init(12);
    const game = try trainGameOpts(&a, rng_a.random(), 0.001, null, .{ .deep_mix = true, .dump = &dump });
    const again = try trainGameOpts(&b, rng_b.random(), 0.001, null, .{ .deep_mix = true, .dump = &dump });
    _ = try trainGameOpts(&plain_net, rng_plain.random(), 0.001, null, .{});

    // one deep draw per self-play step, and no draw at all without the dump: 50% self-play / 50% deep
    try std.testing.expectEqual(@as(u64, game.moves) * 2, dump.draws);
    // same seed -> same games and byte-identical weights
    try std.testing.expectEqual(game.moves, again.moves);
    try std.testing.expectEqual(game.score, again.score);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    // the deep half is not a no-op: it changes the weights (and with them the rest of the game)
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&plain_net)));
}

test "restart starts on a stored deep board with one fresh spawn, deterministically" {
    var dump: Dump = .{};
    var stored: Board = @splat(@splat(0));
    stored[0][0] = dump_min_tile;
    dump.append(stored);

    var rng_a = std.Random.DefaultPrng.init(5);
    var rng_b = std.Random.DefaultPrng.init(5);
    const a = restartState(rng_a.random(), &dump);
    const b = restartState(rng_b.random(), &dump);
    try std.testing.expectEqual(@as(u32, 0), a.score); // the dump has no score, only deltas matter
    try std.testing.expectEqual(a.board, b.board); // same seed -> same spawn
    try std.testing.expectEqual(@as(u64, 2), dump.draws);
    // exactly one spawn on top of the stored board
    try std.testing.expectEqual(dump_min_tile, core.maxTile(a.board));
    var tiles: usize = 0;
    for (a.board) |row| for (row) |v| {
        if (v != 0) tiles += 1;
    };
    try std.testing.expectEqual(@as(usize, 2), tiles);
}

test "restart games are deterministic and start from a deep position" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var plain_net = a;
    var dump: Dump = .{};
    var stored: Board = @splat(@splat(0));
    stored[0][0] = dump_min_tile;
    stored[0][1] = 128;
    dump.append(stored);

    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_plain = std.Random.DefaultPrng.init(12);
    const game = try trainGameOpts(&a, rng_a.random(), 0.001, null, .{ .restart = true, .dump = &dump });
    const again = try trainGameOpts(&b, rng_b.random(), 0.001, null, .{ .restart = true, .dump = &dump });
    const plain = try trainGameOpts(&plain_net, rng_plain.random(), 0.001, null, .{});

    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    try std.testing.expectEqual(game.moves, again.moves);
    try std.testing.expectEqual(game.score, again.score);
    // the restart half is a different trajectory: it starts from a 4096 board, not from two random tiles
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&plain_net)));
    try std.testing.expect(game.moves < plain.moves);
}

test "trainGame with diagnostic gives the same weights and records steps" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var diag: Diag = .{};
    _ = trainGame(&a, rng_a.random(), 0.001, null);
    _ = trainGame(&b, rng_b.random(), 0.001, &diag);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    var total: u64 = 0;
    for (diag.stats) |row| for (row) |s| {
        total += s.count;
    };
    try std.testing.expect(total > 0);
}

test "default trainGame weights match the recorded checksum (5 games, seeds 11/12)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(12);
    for (0..5) |_| _ = trainGame(&net, rng.random(), 0.003, null);
    try std.testing.expectEqual(@as(u32, 3042928886), std.hash.Crc32.hash(std.mem.asBytes(&net)));
}

test "trainGameOpts with default options equals trainGame" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&a, rng_a.random(), 0.003, null);
    _ = try trainGameOpts(&b, rng_b.random(), 0.003, null, .{});
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

test "td1 + greedy1 is the default self-play path, bit for bit (#158)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var default_opts = base;
    var explicit = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    const plain = trainGame(&base, rng_a.random(), 0.003, null);
    const with_default = (try trainGameOpts(&default_opts, rng_b.random(), 0.003, null, .{})).score;
    const with_td1 = (try trainGameOpts(&explicit, rng_c.random(), 0.003, null, .{ .target = .td1, .selfplay = .greedy1 })).score;
    try std.testing.expectEqual(plain, with_default);
    try std.testing.expectEqual(plain, with_td1);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&default_opts));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&explicit));
}

test "search2, deep2 and tp2 train one game without NaN (#158)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const init = Net.init(init_rng.random());
    const variants = [_]Options{
        .{ .target = .search2 },
        .{ .selfplay = .deep2 },
        .{ .target = .search2, .selfplay = .deep2 },
    };
    for (variants) |opts| {
        var net = init;
        var rng = std.Random.DefaultPrng.init(12);
        const result = try trainGameOpts(&net, rng.random(), 0.003, null, opts);
        try std.testing.expect(result.score > 0);
        for (flat(&net)) |w| try std.testing.expect(std.math.isFinite(w));
        // the variant has to actually move the weights, otherwise it silently fell back to the default path
        try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&net), std.mem.asBytes(&init)));
    }
}

test "n-step window closes the oldest step with n rewards plus the closing value" {
    var window = NStepWindow.init(2);
    const a = boardWithEmpties(2);
    const b = boardWithEmpties(3);
    const c = boardWithEmpties(4);
    try std.testing.expectEqual(@as(?NStepWindow.Done, null), window.push(.{ .after = a, .reward = 0.1 }, 0.5));
    try std.testing.expectEqual(@as(?NStepWindow.Done, null), window.push(.{ .after = b, .reward = 0.2 }, 0.7));
    // third step fills the window: step a closes with 0.1 + 0.2 + V(c)
    const done = window.push(.{ .after = c, .reward = 0.3 }, 1.25).?;
    try std.testing.expectEqual(a, done.after);
    try std.testing.expectApproxEqAbs(@as(f64, 1.55), done.target, 1e-12);
}

test "n-step window finish flushes the pending steps with truncated returns" {
    var window = NStepWindow.init(4);
    const a = boardWithEmpties(2);
    const b = boardWithEmpties(3);
    const c = boardWithEmpties(4);
    _ = window.push(.{ .after = a, .reward = 0.5 }, 0);
    _ = window.push(.{ .after = b, .reward = 0.25 }, 0);
    _ = window.push(.{ .after = c, .reward = 1.0 }, 0);
    // game over: V(terminal) = 0, so each remaining step gets the rewards that follow it
    const f0 = window.nextFinish().?;
    try std.testing.expectEqual(a, f0.after);
    try std.testing.expectApproxEqAbs(@as(f64, 1.75), f0.target, 1e-12);
    const f1 = window.nextFinish().?;
    try std.testing.expectEqual(b, f1.after);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), f1.target, 1e-12);
    const f2 = window.nextFinish().?;
    try std.testing.expectEqual(c, f2.after);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), f2.target, 1e-12);
    try std.testing.expectEqual(@as(?NStepWindow.Done, null), window.nextFinish());
}

test "nstep = 1 reproduces TD(0) byte for byte" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var one = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    const td0 = try trainGameOpts(&base, rng_a.random(), 0.003, null, .{});
    const nstep = try trainGameOpts(&one, rng_b.random(), 0.003, null, .{ .nstep = 1 });
    try std.testing.expectEqual(td0, nstep);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&one));
}

test "nstep = 4 changes the weights, stays finite and is deterministic" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var plain = base;
    var four = base;
    var again = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&plain, rng_a.random(), 0.003, null, .{});
    _ = try trainGameOpts(&four, rng_b.random(), 0.003, null, .{ .nstep = 4 });
    _ = try trainGameOpts(&again, rng_c.random(), 0.003, null, .{ .nstep = 4 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain), std.mem.asBytes(&four)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&four), std.mem.asBytes(&again));
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&four))) |w| try std.testing.expect(std.math.isFinite(w));
}

test "nstep = 0 equals the default path" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var off = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&base, rng_a.random(), 0.003, null, .{});
    _ = try trainGameOpts(&off, rng_b.random(), 0.003, null, .{ .nstep = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&off));
}

test "sweep_lambda = 0 keeps the replay sweep's TD(0) step byte for byte (#300)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var plain = base;
    var zero = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&plain, rng_a.random(), 0.003, null, .{ .replay_k = 8 });
    _ = try trainGameOpts(&zero, rng_b.random(), 0.003, null, .{ .replay_k = 8, .sweep_lambda = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain), std.mem.asBytes(&zero));
}

test "a fresh sweep trace applies the plain TD(0) step whatever lambda is (#300)" {
    // `trainTrace` accumulates `lambda * trace + grad`, so a zeroed trace starts at the gradient itself:
    // the replay sweep's first step (and a `replay_k = 1` sweep) is the TD(0) step up to associativity.
    var init_rng = std.Random.DefaultPrng.init(14);
    const start_net = Net.init(init_rng.random());
    var plain = start_net;
    var lam = start_net;
    var cache_a: Cache = .{};
    var cache_b: Cache = .{};
    var trace: Net = std.mem.zeroes(Net);
    _ = forward(&plain, sample_board, &cache_a);
    _ = forward(&lam, sample_board, &cache_b);
    train(&plain, &cache_a, 0.7, 0.01);
    trainTrace(&lam, &cache_b, 0.7, 0.01, &trace, 0.7);
    for (flat(&plain), flat(&lam)) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-15);
}

test "sweep_lambda with replay_k = 1 matches the plain replay sweep (#300)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var td0 = base;
    var lam = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&td0, rng_a.random(), 0.003, null, .{ .replay_k = 1 });
    _ = try trainGameOpts(&lam, rng_b.random(), 0.003, null, .{ .replay_k = 1, .sweep_lambda = 0.7 });
    for (flatConst(&td0), flatConst(&lam)) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-12);
}

test "sweep_lambda = 0.7 changes the weights, stays finite and is deterministic (#300)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var td0 = base;
    var lam = base;
    var again = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&td0, rng_a.random(), 0.003, null, .{ .replay_k = 8 });
    _ = try trainGameOpts(&lam, rng_b.random(), 0.003, null, .{ .replay_k = 8, .sweep_lambda = 0.7 });
    _ = try trainGameOpts(&again, rng_c.random(), 0.003, null, .{ .replay_k = 8, .sweep_lambda = 0.7 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&td0), std.mem.asBytes(&lam)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&lam), std.mem.asBytes(&again));
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&lam))) |w| try std.testing.expect(std.math.isFinite(w));
}

test "sweep_lambda = 0 leaves a non-replay run unchanged (#300)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var off = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&base, rng_a.random(), 0.003, null, .{});
    _ = try trainGameOpts(&off, rng_b.random(), 0.003, null, .{ .sweep_lambda = 0.7 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&off));
}

test "backward sweep changes the weights" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var back = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = try trainGameOpts(&back, rng_b.random(), 0.003, null, .{ .backward = true });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&base), std.mem.asBytes(&back)));
}

test "replay tail changes the weights and K=0 is off" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var replay = base;
    var off = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = try trainGameOpts(&replay, rng_b.random(), 0.003, null, .{ .replay_k = 30 });
    _ = try trainGameOpts(&off, rng_c.random(), 0.003, null, .{ .replay_k = 0 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&base), std.mem.asBytes(&replay)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&off));
}

// A frozen target must be the only source of the bootstrap value; otherwise the regression
// target still moves with the live weights and the variant measures nothing.
test "bootstrapValue reads the frozen target net while the live net is the default" {
    var init_rng = std.Random.DefaultPrng.init(11);
    const live = Net.init(init_rng.random());
    var target = live;
    target.b3 += 1; // `forward` adds b3 once, so the two values differ by exactly one
    var game_rng = std.Random.DefaultPrng.init(12);
    const state = GameState.gameStart(game_rng.random());
    const pick = pickMove(&live, state, false).?;

    const frozen = bootstrapValue(&live, .{ .target_net = &target }, pick.after);
    try std.testing.expectEqual(valueOf(&target, pick.after), frozen);
    try std.testing.expectApproxEqAbs(valueOf(&live, pick.after) + 1, frozen, 1e-12);
    try std.testing.expectEqual(valueOf(&live, pick.after), bootstrapValue(&live, .{}, pick.after));
}

// lr = 0 makes every update a no-op, so the live net never moves. Any use of the frozen
// target for move selection would change the game; keeping score and moves equal proves the policy
// and the spawn tape still come from the live net.
test "frozen target leaves the live policy and spawn tape unchanged" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var target = base;
    target.b3 += 1;
    const target_before = target;
    var with_target = base;
    var live = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    const a = try trainGameOpts(&with_target, rng_a.random(), 0, null, .{ .target_net = &target });
    const b = try trainGameOpts(&live, rng_b.random(), 0, null, .{});
    try std.testing.expectEqual(b.score, a.score);
    try std.testing.expectEqual(b.moves, a.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&with_target));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&target_before), std.mem.asBytes(&target));
}

// The per-move path, the online half of the replay path (`replay_lr_scale = 0` leaves only
// the online updates at a nonzero rate) and the replay path itself all have to see the target. The
// same test asserts the target snapshot is never written during a game.
test "frozen target changes the online and replay updates without being mutated" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var target = base;
    target.b3 += 1;
    const target_before = target;

    var plain_with = base;
    var plain_without = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&plain_with, rng_a.random(), 0.003, null, .{ .target_net = &target });
    _ = try trainGameOpts(&plain_without, rng_b.random(), 0.003, null, .{});
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain_with), std.mem.asBytes(&plain_without)));

    var online_with = base;
    var online_without = base;
    var rng_c = std.Random.DefaultPrng.init(13);
    var rng_d = std.Random.DefaultPrng.init(13);
    _ = try trainGameOpts(&online_with, rng_c.random(), 0.003, null, .{ .target_net = &target, .replay_k = 30, .replay_lr_scale = 0 });
    _ = try trainGameOpts(&online_without, rng_d.random(), 0.003, null, .{ .replay_k = 30, .replay_lr_scale = 0 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&online_with), std.mem.asBytes(&online_without)));

    var replay_with = base;
    var replay_without = base;
    var rng_e = std.Random.DefaultPrng.init(14);
    var rng_f = std.Random.DefaultPrng.init(14);
    _ = try trainGameOpts(&replay_with, rng_e.random(), 0.003, null, .{ .target_net = &target, .replay_k = 30 });
    _ = try trainGameOpts(&replay_without, rng_f.random(), 0.003, null, .{ .replay_k = 30 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&replay_with), std.mem.asBytes(&replay_without)));

    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&target_before), std.mem.asBytes(&target));
}

test "priority sampling gives the uniform share plus the |err|^alpha share" {
    var pr: Priority = .{};
    // entry 0 has a large error, entries 1..3 have none: weights 4^0.5 = 2 vs (1e-3)^0.5 each
    pr.len = 4;
    pr.items[0].err = 4;
    for (pr.items[1..4]) |*e| e.err = 0;
    var rng = std.Random.DefaultPrng.init(7);
    const draws = 40000;
    var hits: usize = 0;
    for (0..draws) |_| {
        if (pr.sample(rng.random()) == 0) hits += 1;
    }
    // expected: 0.5 * 1/4 + 0.5 * w0 / (w0 + 3 * w_small), w0 = 2, w_small = sqrt(eps)
    const w_small = @sqrt(Priority.err_floor);
    const expected = 0.5 * 0.25 + 0.5 * 2.0 / (2.0 + 3 * w_small);
    const share = @as(f64, @floatFromInt(hits)) / draws;
    try std.testing.expectApproxEqAbs(expected, share, 0.01);
}

test "priority pass changes the weights" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var prio = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var pr: Priority = .{};
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = try trainGameOpts(&prio, rng_b.random(), 0.003, null, .{ .priority = &pr });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&base), std.mem.asBytes(&prio)));
    try std.testing.expect(pr.len > 0);
}

/// Board with exactly `empties` empty cells (zeros first), for the midgame sampler tests.
fn boardWithEmpties(empties: usize) Board {
    var b: Board = undefined;
    var i: usize = 0;
    for (&b) |*row| for (row) |*v| {
        v.* = if (i < empties) 0 else 2;
        i += 1;
    };
    return b;
}

test "midgame sampling only returns entries with 4..7 empty cells" {
    var pr: Priority = .{};
    const empties = [_]usize{ 0, 2, 3, 4, 5, 6, 7, 8, 12, 16 };
    for (empties, 0..) |e, i| pr.items[i].after = boardWithEmpties(e);
    pr.len = empties.len;
    var rng = std.Random.DefaultPrng.init(7);
    var seen: [4]bool = @splat(false);
    for (0..1000) |_| {
        const n = countEmpty(pr.items[pr.sampleMidgame(rng.random())].after);
        try std.testing.expect(n >= 4 and n <= 7);
        seen[n - 4] = true;
    }
    for (seen) |hit| try std.testing.expect(hit);
}

test "midgame sampling falls back to the whole buffer when no 4..7 entry exists" {
    var pr: Priority = .{};
    const empties = [_]usize{ 0, 1, 2, 3, 8, 9 };
    for (empties, 0..) |e, i| pr.items[i].after = boardWithEmpties(e);
    pr.len = empties.len;
    var rng = std.Random.DefaultPrng.init(7);
    var hits: [6]usize = @splat(0);
    for (0..6000) |_| hits[pr.sampleMidgame(rng.random())] += 1;
    for (hits) |h| try std.testing.expect(h > 0);
}

test "midmix sampling mixes the uniform share with the weighted 4..7 share" {
    var pr: Priority = .{};
    // 0 and 1 are midgame entries (err 4 and 0), 2 and 3 are not: only the uniform share can hit those.
    pr.items[0] = .{ .after = boardWithEmpties(5), .reward = 0, .next = undefined, .terminal = false, .err = 4 };
    pr.items[1] = .{ .after = boardWithEmpties(5), .reward = 0, .next = undefined, .terminal = false, .err = 0 };
    pr.items[2] = .{ .after = boardWithEmpties(2), .reward = 0, .next = undefined, .terminal = false, .err = 0 };
    pr.items[3] = .{ .after = boardWithEmpties(12), .reward = 0, .next = undefined, .terminal = false, .err = 0 };
    pr.len = 4;
    var rng = std.Random.DefaultPrng.init(7);
    const draws = 40000;
    var hits: [4]usize = @splat(0);
    for (0..draws) |_| hits[pr.sampleMidmix(rng.random())] += 1;
    const w_small = @sqrt(Priority.err_floor);
    // entry 0: 0.5 * 1/4 + 0.5 * 2 / (2 + w_small); entries 2..3: 0.5 * 1/4
    const p0 = 0.5 * 0.25 + 0.5 * 2.0 / (2.0 + w_small);
    const p_outside = 0.5 * 0.25;
    try std.testing.expectApproxEqAbs(p0, @as(f64, @floatFromInt(hits[0])) / draws, 0.01);
    try std.testing.expectApproxEqAbs(p_outside, @as(f64, @floatFromInt(hits[2])) / draws, 0.01);
    try std.testing.expectApproxEqAbs(p_outside, @as(f64, @floatFromInt(hits[3])) / draws, 0.01);
}

test "midmix weighted branch only returns entries with 4..7 empty cells" {
    var pr: Priority = .{ .uniform_share = 0 };
    const empties = [_]usize{ 0, 4, 5, 6, 7, 8, 16 };
    for (empties, 0..) |e, i| pr.items[i] = .{ .after = boardWithEmpties(e), .reward = 0, .next = undefined, .terminal = false, .err = @floatFromInt(i) };
    pr.len = empties.len;
    var rng = std.Random.DefaultPrng.init(7);
    var seen: [4]bool = @splat(false);
    for (0..1000) |_| {
        const n = countEmpty(pr.items[pr.sampleMidmix(rng.random())].after);
        try std.testing.expect(n >= 4 and n <= 7);
        seen[n - 4] = true;
    }
    for (seen) |hit| try std.testing.expect(hit);
}

test "midmix sampling falls back to the whole buffer when no 4..7 entry exists" {
    var pr: Priority = .{ .uniform_share = 0 };
    const empties = [_]usize{ 0, 1, 2, 3, 8, 9 };
    for (empties, 0..) |e, i| pr.items[i] = .{ .after = boardWithEmpties(e), .reward = 0, .next = undefined, .terminal = false, .err = 0 };
    pr.len = empties.len;
    var rng = std.Random.DefaultPrng.init(7);
    var hits: [6]usize = @splat(0);
    for (0..6000) |_| hits[pr.sampleMidmix(rng.random())] += 1;
    for (hits) |h| try std.testing.expect(h > 0);
}

test "parallel training batch matches a serial run of the same game seeds" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 4;
    var init_rng = std.Random.DefaultPrng.init(77);
    var base = Net.init(init_rng.random());

    // Serial reference: game i uses seed (batch seed + i) from the same starting net,
    // then the batch result is net_start + mean(net_after_i - net_start), summed in game order.
    var units: [games]Net = undefined;
    for (&units, 0..) |*u, i| {
        u.* = base;
        var rng = std.Random.DefaultPrng.init(77 + i);
        _ = trainGame(u, rng.random(), 0.001, null);
    }
    const flat_len = @sizeOf(Net) / @sizeOf(f64);
    var sums: [flat_len]f64 = @splat(0);
    const start = flat(&base);
    for (&units) |*u| {
        for (flat(u), start, 0..) |v, s, k| sums[k] += v - s;
    }
    var serial = base;
    for (flat(&serial), start, sums) |*w, s, sum| w.* = s + sum / games;

    var par = base;
    const par_out = trainBatch(&par, 77, 0, games, 0.001, null);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&serial), std.mem.asBytes(&par));
    // : the batch reports the moves it played, so the `par` CLI line never prints `0 moves`.
    var serial_moves: u64 = 0;
    for (0..games) |i| {
        var rng = std.Random.DefaultPrng.init(77 + i);
        var u = base;
        serial_moves += (try trainGameOpts(&u, rng.random(), 0.001, null, .{})).moves;
    }
    try std.testing.expectEqual(serial_moves, par_out.moves);
}

test "batch_scale multiplies the merged delta so a batch does the work of its games (#137)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 4;
    var init_rng = std.Random.DefaultPrng.init(77);
    var base = Net.init(init_rng.random());

    // Same reconstruction as the plain batch test: the merged delta is the mean of the per-game
    // deltas, so `batch_scale = games` turns one batch step into the sum, which is what
    // `games` serial games' worth of update is (the linear scaling rule).
    var units: [games]Net = undefined;
    for (&units, 0..) |*u, i| {
        u.* = base;
        var rng = std.Random.DefaultPrng.init(77 + i);
        _ = trainGame(u, rng.random(), 0.001, null);
    }
    const flat_len = @sizeOf(Net) / @sizeOf(f64);
    var sums: [flat_len]f64 = @splat(0);
    const start = flat(&base);
    for (&units) |*u| {
        for (flat(u), start, 0..) |v, s, k| sums[k] += v - s;
    }

    var scaled_serial = base;
    for (flat(&scaled_serial), start, sums) |*w, s, sum| w.* = s + sum;

    var par = base;
    _ = try trainBatchOpts(&par, 77, 0, games, 0.001, null, .{ .batch_scale = @floatFromInt(games) });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&scaled_serial), std.mem.asBytes(&par));

    // 1.0 stays the historical mean (the plain batch test above pins the bytes).
    var unscaled = base;
    _ = try trainBatchOpts(&unscaled, 77, 0, games, 0.001, null, .{ .batch_scale = 1.0 });
    var mean_serial = base;
    for (flat(&mean_serial), start, sums) |*w, s, sum| w.* = s + sum / games;
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&mean_serial), std.mem.asBytes(&unscaled));
}

test "clamp bounds the TD target (#234)" {
    try std.testing.expectEqual(@as(f64, 7), clampTarget(7, .{}));
    try std.testing.expectEqual(@as(f64, -7), clampTarget(-7, .{ .clamp = 200 }));
    try std.testing.expectEqual(@as(f64, 200), clampTarget(1e6, .{ .clamp = 200 }));
    try std.testing.expectEqual(@as(f64, -200), clampTarget(-1.685e86, .{ .clamp = 200 }));
}

test "a zero step ceiling leaves serial and batch training bit-identical (#246)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(246);
    const base = Net.init(init_rng.random());
    var cache: Cache = .{};
    _ = forward(&base, sample_board, &cache);

    var plain = base;
    train(&plain, &cache, 0.7, 0.01);
    var zero = base;
    trainCapped(&zero, &cache, 0.7, 0.01, 0);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain), std.mem.asBytes(&zero));

    // the default Options carry both zeros, so an explicit zero batch is the historical batch
    const games = 3;
    var history = base;
    _ = trainBatch(&history, 246, 0, games, 0.003, null);
    var explicit = base;
    _ = try trainBatchOpts(&explicit, 246, 0, games, 0.003, null, .{ .grad_clip = 0, .step_clip = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&history), std.mem.asBytes(&explicit));
}

test "step clip cuts every element of one SGD step (#246)" {
    var init_rng = std.Random.DefaultPrng.init(246);
    const base = Net.init(init_rng.random());
    var cache: Cache = .{};
    _ = forward(&base, sample_board, &cache);
    // lr 1 makes the step equal to the gradient, so the clip threshold is a real fraction of it
    var plain = base;
    train(&plain, &cache, 1.0, 1.0);
    var biggest: f64 = 0;
    for (flat(&plain), flatConst(&base)) |a, b| biggest = @max(biggest, @abs(a - b));
    try std.testing.expect(biggest > 1e-6);

    const cap = biggest / 2;
    var capped = base;
    trainCapped(&capped, &cache, 1.0, 1.0, cap);
    var hit = false;
    for (flat(&capped), flatConst(&base)) |a, b| {
        const moved = @abs(a - b);
        try std.testing.expect(moved <= cap * (1 + 1e-12));
        if (moved >= cap * (1 - 1e-9)) hit = true;
    }
    try std.testing.expect(hit);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain), std.mem.asBytes(&capped)));
}

test "grad clip bounds the L2 norm of one merged batch step (#246)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 4;
    var init_rng = std.Random.DefaultPrng.init(246);
    const base = Net.init(init_rng.random());

    var plain = base;
    _ = try trainBatchOpts(&plain, 246, 0, games, 0.003, null, .{});
    var norm2: f64 = 0;
    for (flat(&plain), flatConst(&base)) |a, b| norm2 += (a - b) * (a - b);
    const norm = @sqrt(norm2);
    try std.testing.expect(norm > 0);

    const cap = norm / 4;
    var capped = base;
    _ = try trainBatchOpts(&capped, 246, 0, games, 0.003, null, .{ .grad_clip = cap });
    const scale = cap / norm;
    var capped_norm2: f64 = 0;
    for (flat(&capped), flat(&plain), flatConst(&base)) |a, p, b| {
        // one scale for the whole delta: the direction of the step survives
        try std.testing.expectApproxEqRel((p - b) * scale, a - b, 1e-6);
        capped_norm2 += (a - b) * (a - b);
    }
    try std.testing.expectApproxEqRel(cap, @sqrt(capped_norm2), 1e-6);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain), std.mem.asBytes(&capped)));
}

test "weightsFinite rejects a diverged net (#234)" {
    var prng = std.Random.DefaultPrng.init(31);
    var net = Net.init(prng.random());
    try std.testing.expect(weightsFinite(&net));
    net.b3 = std.math.nan(f64);
    try std.testing.expect(!weightsFinite(&net));
    net.b3 = 2e6; // finite but far past anything training reaches (the shipped net's max is 27)
    try std.testing.expect(!weightsFinite(&net));
    net.b3 = 1.685e86; // what seed 2 actually reached (#234)
    try std.testing.expect(!weightsFinite(&net));
    net.b3 = 0;
    try std.testing.expect(weightsFinite(&net));
}

test "Adam normalises a unit gradient to one lr-sized step per weight (#231)" {
    var init_rng = std.Random.DefaultPrng.init(91);
    var net = Net.init(init_rng.random());
    const before = net;
    var g = std.mem.zeroes(Net);
    for (flat(&g)) |*x| x.* = 1;
    var state: Adam = .{};
    applyAdam(&net, &g, &state, 0.01);
    // first step: `m_hat = v_hat = 1` for a unit gradient, so every weight moves by exactly `lr`
    for (flat(&net), flatConst(&before)) |w, s| try std.testing.expectApproxEqAbs(s - 0.01, w, 1e-6);
    // a zero gradient then moves on the momentum alone, by less than the last step but not nothing
    var zero = std.mem.zeroes(Net);
    const before2 = net;
    applyAdam(&net, &zero, &state, 0.01);
    for (flat(&net), flatConst(&before2)) |w, s| try std.testing.expectApproxEqAbs(s - 0.0067, w, 1e-3);
}

test "par + Adam takes one bounded step per batch (#231)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 2;
    var init_rng = std.Random.DefaultPrng.init(77);
    var base = Net.init(init_rng.random());
    var state: Adam = .{};
    var net = base;
    _ = try trainBatchOpts(&net, 77, 0, games, 0.003, null, .{ .adam = &state, .adam_lr = 0.01 });
    try std.testing.expectEqual(@as(u64, 1), state.t);
    for (flat(&net)) |w| try std.testing.expect(std.math.isFinite(w));
    // the gradient scale never enters the step: |dw| <= adam_lr whatever the batch delta was
    for (flat(&net), flatConst(&base)) |w, s| try std.testing.expect(@abs(w - s) <= 0.0101);
}

test "parallel training batch with variant opts matches a serial run of the same game seeds" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 4;
    // symmetric per-move updates and the post-game replay pass take different branches of trainGameOpts
    const variants = [_]Options{ .{ .symmetry = true }, .{ .replay_k = 4 } };
    for (variants) |opts| {
        var init_rng = std.Random.DefaultPrng.init(77);
        var base = Net.init(init_rng.random());

        // Serial reference (same reconstruction as the plain batch test), one trainGameOpts per game.
        var units: [games]Net = undefined;
        for (&units, 0..) |*u, i| {
            u.* = base;
            var rng = std.Random.DefaultPrng.init(77 + i);
            _ = try trainGameOpts(u, rng.random(), 0.001, null, opts);
        }
        const flat_len = @sizeOf(Net) / @sizeOf(f64);
        var sums: [flat_len]f64 = @splat(0);
        const start = flat(&base);
        for (&units) |*u| {
            for (flat(u), start, 0..) |v, s, k| sums[k] += v - s;
        }
        var serial = base;
        for (flat(&serial), start, sums) |*w, s, sum| w.* = s + sum / games;

        var par = base;
        _ = try trainBatchOpts(&par, 77, 0, games, 0.001, null, opts);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&serial), std.mem.asBytes(&par));
    }
}

test "move cap leaves a normal game bit-identical and stops a long one (#242)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(242);
    const start = Net.init(init_rng.random());

    // The cap is on by default but 100,000 is far past a real game: nothing about a normal run moves,
    // including the rng draws left after the game (same spawn tape, same training, same `moves`).
    var plain_net = start;
    var plain_rng = std.Random.DefaultPrng.init(243);
    const plain = try trainGameOpts(&plain_net, plain_rng.random(), 0.003, null, .{});
    try std.testing.expect(!plain.capped);
    try std.testing.expect(plain.moves < default_move_cap);

    var same_net = start;
    var same_rng = std.Random.DefaultPrng.init(243);
    const same = try trainGameOpts(&same_net, same_rng.random(), 0.003, null, .{ .move_cap = default_move_cap });
    try std.testing.expectEqual(plain.score, same.score);
    try std.testing.expectEqual(plain.moves, same.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain_net), std.mem.asBytes(&same_net));
    try std.testing.expectEqual(plain_rng.random().uintLessThan(u64, 1_000_000), same_rng.random().uintLessThan(u64, 1_000_000));

    // A frozen (lr = 0) game has a fixed policy, so half its length is a cap the same spawn tape must hit.
    const frozen = frozenGame(&start, 243, .{});
    const cap: u32 = @max(1, frozen.moves / 2);
    var short_net = start;
    var short_rng = std.Random.DefaultPrng.init(243);
    const short = try trainGameOpts(&short_net, short_rng.random(), 0, null, .{ .move_cap = cap });
    try std.testing.expect(short.capped);
    try std.testing.expectEqual(cap, short.moves);
    // 0 disables the cap, so the same game runs to its normal end
    const uncapped = frozenGame(&start, 243, .{ .move_cap = 0 });
    try std.testing.expect(!uncapped.capped);
    try std.testing.expectEqual(frozen.moves, uncapped.moves);
}

test "the parallel batch reports and aggregates the capped game (#242)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 4;
    const cap: u32 = 20;
    var init_rng = std.Random.DefaultPrng.init(242);
    const base = Net.init(init_rng.random());

    var par = base;
    const out = try trainBatchOpts(&par, 242, 0, games, 0.003, null, .{ .move_cap = cap });

    // Same games serially: one capped game must not hide the other games' moves from the batch.
    var serial_moves: u64 = 0;
    var capped_game: ?usize = null;
    var capped_moves: u32 = 0;
    for (0..games) |i| {
        var unit = base;
        var rng = std.Random.DefaultPrng.init(trainingSeed(242, i));
        const game = try trainGameOpts(&unit, rng.random(), 0.003, null, .{ .move_cap = cap });
        serial_moves += game.moves;
        if (game.capped and capped_game == null) {
            capped_game = i;
            capped_moves = game.moves;
        }
    }
    try std.testing.expect(capped_game != null);
    try std.testing.expectEqual(capped_game, out.capped_game);
    try std.testing.expectEqual(capped_moves, out.capped_moves);
    try std.testing.expectEqual(serial_moves, out.moves);
    try std.testing.expectEqual(cap, out.capped_moves);
}

test "game time limits are deterministic at their boundaries (#336)" {
    const start = clockNow();
    try std.testing.expect(!gameTimeLimitReached(start, null));
    try std.testing.expect(gameTimeLimitReached(start, 0));
    try std.testing.expect(!gameTimeLimitReached(start, std.math.maxInt(u64)));
}

test "a zero time limit rejects an unfinished game on every training path (#336)" {
    var init_rng = std.Random.DefaultPrng.init(336);
    const start = Net.init(init_rng.random());
    inline for (.{ @as(Options, .{}), @as(Options, .{ .backward = true }), @as(Options, .{ .replay_k = 3 }), @as(Options, .{ .nstep = 4 }) }) |variant| {
        var net = start;
        var opts: Options = variant;
        opts.game_time_limit_ns = 0;
        var rng = std.Random.DefaultPrng.init(337);
        const game = try trainGameOpts(&net, rng.random(), 0.003, null, opts);
        try std.testing.expect(game.capped);
        try std.testing.expect(game.timed_out);
        try std.testing.expectEqual(@as(u32, 0), game.moves);
        try std.testing.expectEqual(@as(u32, 0), game.score);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&start), std.mem.asBytes(&net));
    }
}

test "a zero time limit rejects unfinished evaluation games (#336)" {
    var init_rng = std.Random.DefaultPrng.init(336);
    const net = Net.init(init_rng.random());
    const opts: Options = .{ .game_time_limit_ns = 0 };
    var rng = std.Random.DefaultPrng.init(337);
    const plain = playValueOpts(.{ .single = &net }, rng.random(), .greedy, opts);
    try std.testing.expect(plain.timed_out);
    try std.testing.expectEqual(@as(u32, 0), plain.moves);
    try std.testing.expect(plain.max_tile <= 4);

    var spawns: [100]Spawn = undefined;
    var tape = Tape.initRecord(&spawns);
    const taped = playTapedValueOpts(.{ .single = &net }, rng.random(), .greedy, &tape, opts);
    try std.testing.expect(taped.timed_out);
    try std.testing.expectEqual(@as(u32, 0), taped.moves);
    try std.testing.expect(taped.max_tile <= 4);
}

test "a timed-out parallel batch does not merge unfinished games (#336)" {
    var init_rng = std.Random.DefaultPrng.init(336);
    const start = Net.init(init_rng.random());
    var net = start;
    const out = try trainBatchOpts(&net, 337, 0, 4, 0.003, null, .{ .game_time_limit_ns = 0 });
    try std.testing.expectEqual(@as(?usize, 0), out.timed_out_game);
    try std.testing.expectEqual(@as(?usize, null), out.capped_game);
    try std.testing.expectEqual(@as(u32, 0), out.timed_out_score);
    try std.testing.expectEqual(@as(u32, 0), out.timed_out_moves);
    try std.testing.expectEqual(@as(u64, 0), out.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&start), std.mem.asBytes(&net));
}

test "a zero time limit also applies to staged training (#336)" {
    var rng = std.Random.DefaultPrng.init(336);
    var staged = Staged.init(rng.random());
    const before = staged;
    const game = trainGameStagedOpts(&staged, rng.random(), 0.003, .{ .game_time_limit_ns = 0 });
    try std.testing.expect(game.capped);
    try std.testing.expect(game.timed_out);
    try std.testing.expectEqual(@as(u32, 0), game.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&staged));
}

test "the 8 dihedral transforms are distinct for a non-symmetric board and compose back to identity" {
    const board: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 2, 0, 4, 0 },
        .{ 0, 8, 0, 16 },
    };
    const boards: [symmetry_count]Board = .{ transformBoard(0, board), transformBoard(1, board), transformBoard(2, board), transformBoard(3, board), transformBoard(4, board), transformBoard(5, board), transformBoard(6, board), transformBoard(7, board) };
    for (boards, 0..) |a, i| for (boards[i + 1 ..]) |b| {
        try std.testing.expect(!std.meta.eql(a, b));
    };
    // every transform has order 1, 2 or 4, so applying it 8 times is the identity
    for (0..symmetry_count) |k| {
        var b = board;
        for (0..8) |_| b = transformBoard(k, b);
        try std.testing.expectEqual(board, b);
    }
    // composition of two transforms is again one of the 8 (cell-index permutation group)
    for (0..symmetry_count) |a| for (0..symmetry_count) |b| for (0..core.BoardCellCount) |cell| {
        const composed = symmetryMap(a, symmetryMap(b, cell));
        var found = false;
        for (0..symmetry_count) |c| if (symmetryMap(c, cell) == composed) {
            found = true;
        };
        try std.testing.expect(found);
    };
}

test "forward on a transformed board equals forward with transformed weights" {
    var init_rng = std.Random.DefaultPrng.init(5);
    const net = Net.init(init_rng.random());
    var base_cache: Cache = .{};
    const base_value = forward(&net, sample_board, &base_cache);
    for (0..symmetry_count) |k| {
        // rotate the input rows the same way, then the two views of one position must agree
        var turned = net;
        for (0..core.BoardCellCount) |cell| for (0..levels) |level| {
            const to = symmetryMap(k, cell);
            turned.w1[to * levels + level] = net.w1[cell * levels + level];
        };
        var cache: Cache = .{};
        // same sum, different accumulation order: equal up to float noise
        try std.testing.expectApproxEqAbs(base_value, forward(&turned, transformBoard(k, sample_board), &cache), 1e-9);
    }
}

test "symmetry=false and lambda=0 reproduce the recorded checksum (5 games, seeds 11/12)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(12);
    for (0..5) |_| _ = try trainGameOpts(&net, rng.random(), 0.003, null, .{ .symmetry = false, .lambda = 0 });
    try std.testing.expectEqual(@as(u32, 3042928886), std.hash.Crc32.hash(std.mem.asBytes(&net)));
}

test "rsym rotates only the input and base draws nothing (#271)" {
    const board: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 2, 0, 4, 0 },
        .{ 0, 8, 0, 16 },
    };
    var prng = std.Random.DefaultPrng.init(271);
    const rng = prng.random();
    var seen: [symmetry_count]bool = @splat(false);
    for (0..256) |_| {
        const sym = updateSym(rng, .{ .rsym = true });
        try std.testing.expectEqual(@as(?usize, null), sym.next);
        const trained = symBoard(board, sym.after);
        var found = false;
        for (0..symmetry_count) |k| if (std.meta.eql(trained, transformBoard(k, board))) {
            seen[k] = true;
            found = true;
        };
        try std.testing.expect(found);
    }
    // 256 uniform draws over 8 transforms hit every orientation (no realistic chance of a miss)
    for (seen) |hit| try std.testing.expect(hit);
    // base returns the stored afterstate and consumes no rng, so two identical streams stay aligned
    var a = std.Random.DefaultPrng.init(271);
    var b = std.Random.DefaultPrng.init(271);
    try std.testing.expectEqual(board, symBoard(board, updateSym(a.random(), .{}).after));
    _ = updateSym(b.random(), .{});
    try std.testing.expectEqual(a.random().int(u64), b.random().int(u64));
}

test "rtgt trains the same transform's afterstate and next board (#273)" {
    const board: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 2, 0, 4, 0 },
        .{ 0, 8, 0, 16 },
    };
    const next_board: Board = .{
        .{ 4, 2, 0, 8 },
        .{ 16, 8, 4, 2 },
        .{ 0, 0, 16, 4 },
        .{ 2, 8, 0, 2 },
    };
    const FixedK = struct {
        k: usize,

        fn fill(self: *@This(), buf: []u8) void {
            // `uintLessThan(usize, 8)` is `(x * 8) >> 64`, so `k << 61` selects exactly k.
            const x: u64 = @as(u64, @intCast(self.k)) << 61;
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, x, .little);
            for (buf, 0..) |*byte, i| byte.* = bytes[i % bytes.len];
        }
    };
    const NoCall = struct {
        fn fill(_: *@This(), _: []u8) void {
            @panic("rng should not be called");
        }
    };
    var init_rng = std.Random.DefaultPrng.init(273);
    const start = Net.init(init_rng.random());
    const reward = 12 / reward_scale;
    const spawned: GameState = .{ .board = next_board, .score = 0 };
    const next: Pick = .{ .dir = .left, .after = next_board, .reward = 0, .value = valueOf(&start, next_board) };
    const p: Pick = .{ .dir = .up, .after = board, .reward = reward, .value = 0 };
    const lr = 0.003;

    for (0..symmetry_count) |k| {
        var source = FixedK{ .k = k };
        var got = start;
        var got_cache: Cache = .{};
        var got_trace: Net = std.mem.zeroes(Net);
        const got_target = tdUpdate(&got, &got_cache, p, next, spawned, lr, std.Random.init(&source, FixedK.fill), .{ .rtgt = true }, &got_trace);

        const want_target = reward + valueOf(&start, transformBoard(k, next_board));
        var want = start;
        var want_cache: Cache = .{};
        var want_trace: Net = std.mem.zeroes(Net);
        trainStep(&want, &want_cache, transformBoard(k, board), want_target, lr, .{}, &want_trace);

        try std.testing.expectEqual(want_target, got_target);
        try std.testing.expectEqual(want, got);
    }

    // k = 0 is the untransformed update, i.e. base's one step; base must not draw
    var zero = FixedK{ .k = 0 };
    var got = start;
    var got_cache: Cache = .{};
    var got_trace: Net = std.mem.zeroes(Net);
    const got_target = tdUpdate(&got, &got_cache, p, next, spawned, lr, std.Random.init(&zero, FixedK.fill), .{ .rtgt = true }, &got_trace);
    var none = NoCall{};
    var base = start;
    var base_cache: Cache = .{};
    var base_trace: Net = std.mem.zeroes(Net);
    const base_target = tdUpdate(&base, &base_cache, p, next, spawned, lr, std.Random.init(&none, NoCall.fill), .{}, &base_trace);
    try std.testing.expectEqual(base_target, got_target);
    try std.testing.expectEqual(base, got);
}

test "rtgt rotates a terminal input but leaves its target at the reward (#273)" {
    const board: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 2, 0, 4, 0 },
        .{ 0, 8, 0, 16 },
    };
    const FixedK = struct {
        k: usize,

        fn fill(self: *@This(), buf: []u8) void {
            const x: u64 = @as(u64, @intCast(self.k)) << 61;
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, x, .little);
            for (buf, 0..) |*byte, i| byte.* = bytes[i % bytes.len];
        }
    };
    var init_rng = std.Random.DefaultPrng.init(273);
    const start = Net.init(init_rng.random());
    const reward = 12 / reward_scale;
    const p: Pick = .{ .dir = .up, .after = board, .reward = reward, .value = 0 };
    const spawned: GameState = .{ .board = board, .score = 0 };
    const lr = 0.003;

    for (0..symmetry_count) |k| {
        var source = FixedK{ .k = k };
        var got = start;
        var got_cache: Cache = .{};
        var got_trace: Net = std.mem.zeroes(Net);
        const got_target = tdTerminalUpdate(&got, &got_cache, p, spawned, lr, std.Random.init(&source, FixedK.fill), .{ .rtgt = true }, &got_trace);

        var want = start;
        var want_cache: Cache = .{};
        var want_trace: Net = std.mem.zeroes(Net);
        trainStep(&want, &want_cache, transformBoard(k, board), reward, lr, .{}, &want_trace);

        try std.testing.expectEqual(reward, got_target);
        try std.testing.expectEqual(want, got);
    }
}

test "sym_leaf=false keeps the single-forward leaf and its own cache entry (#178)" {
    var prng = std.Random.DefaultPrng.init(178);
    const net = Net.init(prng.random());
    var scratch: Cache = .{};
    const plain = forward(&net, sample_board, &scratch);
    var cache: ValueCache = .{};
    // the default search is the pre-#178 one: one forward pass, the same value as before
    var search: Search = .{ .vfn = .{ .single = &net }, .cache = &cache, .ply = 2 };
    try std.testing.expectEqual(plain, search.leaf(sample_board));
    try std.testing.expectEqual(@as(u64, 1), search.stats.leaves);
    // the averaged leaf is another value: a miss, not a hit on the single-forward entry
    search.sym_leaf = true;
    _ = search.leaf(sample_board);
    try std.testing.expectEqual(@as(u64, 2), search.stats.leaves);
    try std.testing.expectEqual(@as(u64, 0), search.stats.hits);
    // and the plain entry is still returned as a hit
    search.sym_leaf = false;
    try std.testing.expectEqual(plain, search.leaf(sample_board));
    try std.testing.expectEqual(@as(u64, 2), search.stats.leaves);
    try std.testing.expectEqual(@as(u64, 1), search.stats.hits);
    // the option-carrying entry points default to the same move as the pre-#178 ones
    var stats_a: Stats = .{};
    var stats_b: Stats = .{};
    var cache_a: ValueCache = .{};
    var cache_b: ValueCache = .{};
    const plain_move = bestMoveWithCache(&net, mid_board, .deep, &stats_a, &cache_a);
    const opts_move = bestMoveWithCacheValueOpts(.{ .single = &net }, mid_board, .deep, &stats_b, &cache_b, .{});
    try std.testing.expectEqual(plain_move, opts_move);
    try std.testing.expectEqual(stats_a.leaves, stats_b.leaves);
}

test "sym_leaf averages the 8 dihedral transforms of the leaf (#178)" {
    var prng = std.Random.DefaultPrng.init(1178);
    const net = Net.init(prng.random());
    var cache: ValueCache = .{};
    var search: Search = .{ .vfn = .{ .single = &net }, .cache = &cache, .ply = 2, .sym_leaf = true };
    const got = search.leaf(mid_board.board);
    // expected: the 8 transformed boards, forwarded one by one
    var total: f64 = 0;
    for (0..symmetry_count) |k| {
        var c: Cache = .{};
        total += forward(&net, transformBoard(k, mid_board.board), &c);
    }
    try std.testing.expectApproxEqAbs(total / @as(f64, @floatFromInt(symmetry_count)), got, 1e-12);
    // one leaf, not eight: the averaging is inside the leaf counter
    try std.testing.expectEqual(@as(u64, 1), search.stats.leaves);
}

test "sym_leaf of a dihedrally symmetric board equals one forward (#178)" {
    var prng = std.Random.DefaultPrng.init(2178);
    const net = Net.init(prng.random());
    // constant on each D4 orbit (corners, edges, inner 2x2), so all 8 transforms leave it alone
    const symmetric: Board = .{
        .{ 2, 4, 4, 2 },
        .{ 4, 8, 8, 4 },
        .{ 4, 8, 8, 4 },
        .{ 2, 4, 4, 2 },
    };
    for (0..symmetry_count) |k| try std.testing.expectEqual(symmetric, transformBoard(k, symmetric));
    var cache: ValueCache = .{};
    var search: Search = .{ .vfn = .{ .single = &net }, .cache = &cache, .ply = 2, .sym_leaf = true };
    var scratch: Cache = .{};
    try std.testing.expectApproxEqAbs(forward(&net, symmetric, &scratch), search.leaf(symmetric), 1e-12);
}

/// one short game under an `id_budget`, with its move sequence and its leaf count. The value
/// cache lives for the whole game, exactly as in `playWithOptsValue`.
const IdGame = struct {
    moves: [64]Direction = undefined,
    len: usize = 0,
    leaves: u64 = 0,
};

fn playIdGame(vfn: Value, seed: u64, budget: usize, max_moves: usize) IdGame {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var state = GameState.gameStart(rng);
    var cache: ValueCache = .{};
    var out: IdGame = .{};
    while (out.len < max_moves) {
        var stats: Stats = .{};
        const dir = bestMoveWithCacheValueOpts(vfn, state, .deep, &stats, &cache, .{ .id_budget = budget }) orelse break;
        out.moves[out.len] = dir;
        out.len += 1;
        out.leaves += stats.leaves;
        _ = state.move(dir, rng);
    }
    return out;
}

test "id_budget picks the same moves at the same seed and budget (#183)" {
    var prng = std.Random.DefaultPrng.init(1183);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    // same seed, same budget, twice: the move order and the leaf counts match exactly (no wall clock,
    // no rng inside the search)
    const a = playIdGame(vfn, 183, 5_000, 6);
    const b = playIdGame(vfn, 183, 5_000, 6);
    try std.testing.expect(a.len > 0);
    try std.testing.expectEqualSlices(Direction, a.moves[0..a.len], b.moves[0..b.len]);
    try std.testing.expectEqual(a.leaves, b.leaves);
}

test "id_budget 0 still returns a legal move (#183)" {
    var prng = std.Random.DefaultPrng.init(2183);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var stats: Stats = .{};
    var cache: ValueCache = .{};
    const dir = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &stats, &cache, .{ .id_budget = 0 }) orelse
        return error.TestUnexpectedResult;
    // the fallback spends nothing and the move is legal
    try std.testing.expectEqual(@as(u64, 0), stats.leaves);
    var next = mid_board;
    try std.testing.expect(next.slide(dir).changed);
    // and a dead board still answers null, budget or not
    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    var dead_cache: ValueCache = .{};
    try std.testing.expectEqual(@as(?Direction, null), bestMoveWithCacheValueOpts(vfn, dead, .deep, &stats, &dead_cache, .{ .id_budget = 0 }));
}

test "id_budget adopts a monotonically deeper ply as the budget grows (#183)" {
    var prng = std.Random.DefaultPrng.init(3183);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var previous: u8 = 0;
    var deepest: u8 = 0;
    // a fresh cache per budget: what a budget buys must not depend on an earlier run's entries
    for ([_]usize{ 0, 200, 2_000, 20_000 }) |budget| {
        var stats: Stats = .{};
        var cache: ValueCache = .{};
        const got = deepMoveId(vfn, mid_board, &stats, &cache, null, phaseDepth(.deep, mid_board.board), budget, false, null, 0, null);
        try std.testing.expect(got.ply >= previous);
        try std.testing.expectEqual(got.leaves, stats.leaves);
        previous = got.ply;
        deepest = @max(deepest, got.ply);
    }
    // 20k leaves is more than one 2-ply tree here, so at least one iteration completed
    try std.testing.expect(deepest >= 2);
}

test "id_margin stops the deepening when the start depth's move is not a near tie (#188)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(3191);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    // a huge budget and a zero margin: any positive gap at ply 2 ends the search at ply 2
    var tight_stats: Stats = .{};
    var tight_cache: ValueCache = .{};
    const tight = deepMoveId(vfn, mid_board, &tight_stats, &tight_cache, null, phaseDepth(.deep, mid_board.board), 5_000_000, false, null, 0, 0);
    // without the margin the same budget goes deeper, so the cut is what saved the leaves
    var open_stats: Stats = .{};
    var open_cache: ValueCache = .{};
    const open = deepMoveId(vfn, mid_board, &open_stats, &open_cache, null, phaseDepth(.deep, mid_board.board), 5_000_000, false, null, 0, null);
    try std.testing.expect(tight.ply == 2);
    try std.testing.expect(open.ply > tight.ply);
    try std.testing.expect(tight.leaves < open.leaves);
}

test "id_margin = null and a wide margin keep every iteration (#188)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(3193);
    const net = Net.init(prng.random());
    const vfn: Value = .{ .single = &net };
    var plain_stats: Stats = .{};
    var plain_cache: ValueCache = .{};
    const plain = deepMoveId(vfn, mid_board, &plain_stats, &plain_cache, null, phaseDepth(.deep, mid_board.board), 200_000, false, null, 0, null);
    // a margin no root gap can exceed never triggers the cut, so the search is the same one
    var wide_stats: Stats = .{};
    var wide_cache: ValueCache = .{};
    const wide = deepMoveId(vfn, mid_board, &wide_stats, &wide_cache, null, phaseDepth(.deep, mid_board.board), 200_000, false, null, 0, 1e9);
    try std.testing.expectEqual(plain.dir, wide.dir);
    try std.testing.expectEqual(plain.ply, wide.ply);
    try std.testing.expectEqual(plain.leaves, wide.leaves);
}

test "TD(lambda) trace on a two-step sequence applies the eligibility-weighted update" {
    var init_rng = std.Random.DefaultPrng.init(13);
    const start_net = Net.init(init_rng.random());
    const board1 = sample_board;
    const board2: Board = .{
        .{ 0, 2, 0, 0 },
        .{ 4, 0, 0, 8 },
        .{ 0, 0, 16, 0 },
        .{ 0, 0, 0, 32 },
    };
    const target1: f64 = 0.5;
    const target2: f64 = 1.0;
    const lr: f64 = 0.01;
    const lambda: f64 = 0.7;

    var net = start_net;
    var trace: Net = std.mem.zeroes(Net);
    var cache: Cache = .{};

    var g1: Net = undefined;
    var d1: f64 = undefined;
    _ = forward(&net, board1, &cache);
    g1 = gradOf(&net, &cache);
    d1 = cache.out - target1;
    trainTrace(&net, &cache, target1, lr, &trace, lambda);

    // second step: eligibility is lambda * grad1 + grad2 (gamma = 1)
    var g2: Net = undefined;
    var d2: f64 = undefined;
    _ = forward(&net, board2, &cache);
    g2 = gradOf(&net, &cache);
    d2 = cache.out - target2;
    trainTrace(&net, &cache, target2, lr, &trace, lambda);

    const w0 = flat(@constCast(&start_net));
    const w = flat(&net);
    const p1 = flat(&g1);
    const p2 = flat(&g2);
    for (w0, w, p1, p2) |start, got, grad1, grad2| {
        const expected = start - lr * d1 * grad1 - lr * d2 * (lambda * grad1 + grad2);
        try std.testing.expectApproxEqAbs(expected, got, 1e-12);
    }
}

test "lambda=0 trace update matches the plain TD(0) step" {
    var init_rng = std.Random.DefaultPrng.init(14);
    const start_net = Net.init(init_rng.random());
    var a = start_net;
    var b = start_net;
    var trace: Net = std.mem.zeroes(Net);
    var cache_a: Cache = .{};
    var cache_b: Cache = .{};
    _ = forward(&a, sample_board, &cache_a);
    _ = forward(&b, sample_board, &cache_b);
    train(&a, &cache_a, 0.7, 0.01);
    trainTrace(&b, &cache_b, 0.7, 0.01, &trace, 0);
    // Same update, but `train` applies it once per active row while the trace accumulates duplicates,
    // so the two differ only by float associativity.
    for (flat(&a), flat(&b)) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-15);
}

test "one symmetry step equals eight separate transformed TD(0) steps" {
    var init_rng = std.Random.DefaultPrng.init(15);
    const start_net = Net.init(init_rng.random());
    var together = start_net;
    var manual = start_net;
    var cache: Cache = .{};
    var trace: Net = std.mem.zeroes(Net);
    trainStep(&together, &cache, sample_board, 0.9, 0.001, .{ .symmetry = true }, &trace);
    for (0..symmetry_count) |k| {
        var c: Cache = .{};
        _ = forward(&manual, transformBoard(k, sample_board), &c);
        train(&manual, &c, 0.9, 0.001);
    }
    for (flat(&manual), flat(&together)) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-15);
}

test "symmetry and symlam change the weights (smoke)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var plain = base;
    var sym = base;
    var symlam = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = try trainGameOpts(&plain, rng_a.random(), 0.003, null, .{});
    _ = try trainGameOpts(&sym, rng_b.random(), 0.003, null, .{ .symmetry = true });
    _ = try trainGameOpts(&symlam, rng_c.random(), 0.003, null, .{ .symmetry = true, .lambda = 0.7 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain), std.mem.asBytes(&sym)));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&sym), std.mem.asBytes(&symlam)));
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&symlam))) |w| try std.testing.expect(std.math.isFinite(w));
}

/// Unclipped averaged symmetry gradient of one afterstate: mean over the 8 transforms of
/// (V(k) - target) * grad V(k), every transform evaluated on the same (unmodified) `net`.
/// Test-only, deliberately not shared with `trainSymBatch` so the tests compare two computations.
fn avgSymGrad(net: *const Net, after: Board, target: f64) Net {
    var g = std.mem.zeroes(Net);
    var cache: Cache = .{};
    for (0..symmetry_count) |k| {
        const v = forward(net, transformBoard(k, after), &cache);
        const step = gradOf(net, &cache);
        for (flat(&g), flat(@constCast(&step))) |*acc, x| acc.* += (v - target) * x;
    }
    for (flat(&g)) |*acc| acc.* /= @as(f64, @floatFromInt(symmetry_count));
    return g;
}

test "one symbatch step equals the averaged symmetry gradient applied once" {
    var init_rng = std.Random.DefaultPrng.init(16);
    var start = Net.init(init_rng.random());
    var batched = start;
    var cache: Cache = .{};
    var trace: Net = std.mem.zeroes(Net);
    trainStep(&batched, &cache, sample_board, 0.4, 0.01, .{ .sym_batch = true }, &trace);

    var g = avgSymGrad(&start, sample_board, 0.4);
    for (flat(&batched), flat(&start), flat(&g)) |got, w0, grad| try std.testing.expectApproxEqAbs(w0 - 0.01 * grad, got, 1e-15);
}

test "one symbatch step differs from the eight sequential symmetry steps" {
    var init_rng = std.Random.DefaultPrng.init(17);
    const start = Net.init(init_rng.random());
    var batched = start;
    var sequential = start;
    var cache: Cache = .{};
    var trace: Net = std.mem.zeroes(Net);
    trainStep(&batched, &cache, sample_board, 0.4, 0.01, .{ .sym_batch = true }, &trace);
    trainStep(&sequential, &cache, sample_board, 0.4, 0.01, .{ .symmetry = true }, &trace);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&batched), std.mem.asBytes(&sequential)));
    for (flat(&batched)) |w| try std.testing.expect(std.math.isFinite(w));
}

test "a large averaged gradient is clipped to the module norm" {
    var init_rng = std.Random.DefaultPrng.init(18);
    var start = Net.init(init_rng.random());
    var batched = start;
    var cache: Cache = .{};
    var trace: Net = std.mem.zeroes(Net);
    const target: f64 = 100;
    trainStep(&batched, &cache, sample_board, target, 0.001, .{ .sym_batch = true }, &trace);

    var g = avgSymGrad(&start, sample_board, target);
    var g2: f64 = 0;
    for (flat(&g)) |x| g2 += x * x;
    try std.testing.expect(@sqrt(g2) > 10 * sym_batch_clip);

    var d2: f64 = 0;
    for (flat(&start), flat(&batched)) |w0, w| {
        const delta = w - w0;
        d2 += delta * delta;
    }
    try std.testing.expectApproxEqRel(0.001 * sym_batch_clip, @sqrt(d2), 1e-12);
}

test "scaleInit multiplies the initial weights by the scale (#107)" {
    var rng_a = std.Random.DefaultPrng.init(7);
    const plain = Net.init(rng_a.random());
    var rng_b = std.Random.DefaultPrng.init(7);
    var optimistic = Net.init(rng_b.random());
    scaleInit(&optimistic, 4);
    try std.testing.expectEqual(plain.w1[3][5] * 4, optimistic.w1[3][5]);
    try std.testing.expectEqual(plain.w2[0][1] * 4, optimistic.w2[0][1]);
    try std.testing.expectEqual(plain.w3[2] * 4, optimistic.w3[2]);
    try std.testing.expect(optimistic.b1[0] == plain.b1[0]);
}

test "stageOfEmpties maps empty-cell counts to the 7 stages" {
    const cases = [_]struct { empties: usize, stage: usize }{
        .{ .empties = 0, .stage = 0 },
        .{ .empties = 2, .stage = 0 },
        .{ .empties = 3, .stage = 1 },
        .{ .empties = 4, .stage = 1 },
        .{ .empties = 5, .stage = 2 },
        .{ .empties = 6, .stage = 2 },
        .{ .empties = 7, .stage = 3 },
        .{ .empties = 8, .stage = 3 },
        .{ .empties = 9, .stage = 4 },
        .{ .empties = 10, .stage = 4 },
        .{ .empties = 11, .stage = 5 },
        .{ .empties = 12, .stage = 5 },
        .{ .empties = 13, .stage = 6 },
        .{ .empties = 16, .stage = 6 },
    };
    for (cases) |c| try std.testing.expectEqual(c.stage, stageOfEmpties(c.empties));
    try std.testing.expectEqual(@as(usize, 7), stage_count);
    // out-of-range empties clamp to the last stage instead of indexing past `nets`
    try std.testing.expectEqual(@as(usize, 6), stageOfEmpties(17));
}

/// `Staged` is `stage_count` (7) nets — ~2 MB at the 128/32 default of — so it has to live on the
/// heap: the test runner and `main` stacks overflowed once the shipped width doubled.
fn heapStaged(allocator: std.mem.Allocator, rng: std.Random) !*Staged {
    const self = try allocator.create(Staged);
    self.* = Staged.init(rng);
    return self;
}

test "Staged.forward reads the net of the board's own stage" {
    var prng = std.Random.DefaultPrng.init(11);
    const staged = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(staged);
    const full = boardWithEmpties(16); // stage 6
    const dense = boardWithEmpties(1); // stage 0
    const full_before = staged.forward(full);
    const dense_before = staged.forward(dense);
    // b3 is added straight to the output, so bumping one stage's b3 shifts exactly that stage's values
    staged.nets[0].b3 += 1;
    try std.testing.expectEqual(full_before, staged.forward(full));
    try std.testing.expectEqual(dense_before + 1, staged.forward(dense));
}

test "Staged.train updates only the destination afterstate's stage" {
    var prng = std.Random.DefaultPrng.init(13);
    const staged = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(staged);
    const after = boardWithEmpties(16); // stage 6, the net that gets the step
    const next = boardWithEmpties(1); // stage 0, read for the target only
    // the 7-net copy also has to be heap allocated (14 MB of stack otherwise)
    const pristine = try std.testing.allocator.create(Staged);
    defer std.testing.allocator.destroy(pristine);
    pristine.* = staged.*;
    const before = staged.forward(after);
    const err = staged.train(after, next, 0.5, 0.01);
    const target = 0.5 + pristine.forward(next);
    try std.testing.expectApproxEqAbs(target - before, err, 1e-12);
    for (staged.nets, pristine.nets, 0..) |a, b, k| {
        const changed = !std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
        try std.testing.expectEqual(k == 6, changed);
    }
    // the step moves V(after) toward the target
    try std.testing.expect(@abs(staged.forward(after) - target) < @abs(before - target));
}

test "Staged.train with a terminal step uses the reward alone" {
    var prng = std.Random.DefaultPrng.init(15);
    const staged = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(staged);
    const after = boardWithEmpties(4); // stage 1
    const before = staged.forward(after);
    const err = staged.train(after, null, -1, 0.01);
    try std.testing.expectApproxEqAbs(-1 - before, err, 1e-12);
}

test "staged weights round trip and stay separate from the single-net file" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/mstd.bin", .{tmp.sub_path});
    defer std.testing.allocator.free(path);

    var prng = std.Random.DefaultPrng.init(17);
    const staged = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(staged);
    try staged.saveFile(io, std.testing.allocator, path);
    const loaded = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(loaded);
    try loaded.loadFile(io, std.testing.allocator, path);
    for (staged.nets, loaded.nets) |a, b| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));

    // the staged file is its own format, so the single-net reader still rejects it
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .limited(staged_file_len + 1));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(@as(usize, staged_file_len), bytes.len);
    try std.testing.expectEqualStrings("2048-MS\x00", bytes[0..8]);
    try std.testing.expectError(error.WrongWeightFileSize, parse(bytes));

    // and the staged reader rejects a single-net file
    const single_path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/net.bin", .{tmp.sub_path});
    defer std.testing.allocator.free(single_path);
    try save(&Net.init(prng.random()), io, single_path);
    try std.testing.expectError(error.WrongWeightFileSize, loaded.loadFile(io, std.testing.allocator, single_path));
}

test "staged self-play training is deterministic for a fixed seed" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_a = std.Random.DefaultPrng.init(19);
    var init_b = std.Random.DefaultPrng.init(19);
    const staged_a = try heapStaged(std.testing.allocator, init_a.random());
    defer std.testing.allocator.destroy(staged_a);
    const staged_b = try heapStaged(std.testing.allocator, init_b.random());
    defer std.testing.allocator.destroy(staged_b);
    const before = try std.testing.allocator.create(Staged);
    defer std.testing.allocator.destroy(before);
    before.* = staged_a.*;
    var rng_a = std.Random.DefaultPrng.init(23);
    var rng_b = std.Random.DefaultPrng.init(23);
    const game_a = trainGameStaged(staged_a, rng_a.random(), 0.003);
    const game_b = trainGameStaged(staged_b, rng_b.random(), 0.003);
    try std.testing.expectEqual(game_a, game_b);
    for (staged_a.nets, staged_b.nets) |a, b| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&before.nets), std.mem.asBytes(&staged_a.nets)));
}

test "bestMove and play accept the staged value" {
    var prng = std.Random.DefaultPrng.init(29);
    const staged = try heapStaged(std.testing.allocator, prng.random());
    defer std.testing.allocator.destroy(staged);
    const value: Value = .{ .staged = staged };
    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), bestMoveValue(value, dead, .greedy));
    var rng = std.Random.DefaultPrng.init(31);
    const result = playValue(value, rng.random(), .greedy);
    try std.testing.expect(result.moves > 0);
}

/// One recorded game's tape: 8192 spawns × 8 bytes per `Spawn` = 64 KiB since .
const tape_capacity = 8192;

test "spawn tape records and replays the same game" {
    var prng = std.Random.DefaultPrng.init(37);
    const net = Net.init(prng.random());
    const value: Value = .{ .single = &net };

    // recording draws a uniform number per spawn instead of an index out of the hole count, so it
    // no longer reproduces `playValue`'s game byte for byte; what has to hold is that the tape *is* the
    // game, i.e. a replay of it is identical whatever rng the replay starts from.
    var buf: [tape_capacity]Spawn = undefined;
    var rec_rng = std.Random.DefaultPrng.init(1234);
    var tape = Tape.initRecord(&buf);
    const recorded = playTapedValue(value, rec_rng.random(), .greedy, &tape);
    try std.testing.expect(recorded.moves > 0);
    // one spawn per move plus the two start tiles (one less when the board fills on the last move)
    try std.testing.expect(tape.len <= recorded.moves + 2 and tape.len >= recorded.moves + 1);

    // and a replay from a *different* rng seed is the same game: the tape, not the rng, drives the spawns
    var replay_rng = std.Random.DefaultPrng.init(99);
    var replay_tape = Tape.initReplay(buf[0..tape.len]);
    const replayed = playTapedValue(value, replay_rng.random(), .greedy, &replay_tape);
    try std.testing.expectEqual(recorded.score, replayed.score);
    try std.testing.expectEqual(recorded.moves, replayed.moves);
    try std.testing.expectEqual(recorded.board, replayed.board);
    try std.testing.expectEqual(tape.len, replay_tape.cursor);
}

test "tape replay takes the recorded slot among the empty cells (#235, #237)" {
    // A board whose first empty cell in row-major order is (0,2): the old code placed the replayed tile
    // there whenever the recorded cell was taken, which is the bias remove.
    var state = core.GameState.initEmpty();
    state.board[0][0] = 2;
    state.board[0][1] = 2;
    state.board[1][0] = 4;
    // 13 holes left; ask for the 4th (index 3)
    var spawns = [_]Spawn{Spawn.at(Spawn.pickFor(3, 13), 4)};
    var tape = Tape.initReplay(&spawns);
    var prng = std.Random.DefaultPrng.init(7);
    try std.testing.expect(tape.spawn(&state, prng.random()));
    // empties are (0,2), (0,3), (1,1), (1,2), ... so slot 3 is (1,2)
    try std.testing.expectEqual(@as(core.Tile, 0), state.board[0][2]);
    try std.testing.expectEqual(@as(core.Tile, 0), state.board[0][3]);
    try std.testing.expectEqual(@as(core.Tile, 0), state.board[1][1]);
    try std.testing.expectEqual(@as(core.Tile, 4), state.board[1][2]);
    // the same number on a board with one hole left still lands, in that hole
    var spawns2 = [_]Spawn{Spawn.at(Spawn.pickFor(3, 13), 2)};
    var tape2 = Tape.initReplay(&spawns2);
    var small = core.GameState.initEmpty();
    for (0..core.BoardSize) |row| for (0..core.BoardSize) |col| {
        if (!(row == 2 and col == 3)) small.board[row][col] = 2;
    };
    try std.testing.expect(tape2.spawn(&small, prng.random()));
    try std.testing.expectEqual(@as(core.Tile, 2), small.board[2][3]);
}

test "common random picks give each board an almost-uniform hole (#237)" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Spawn));
    // For floor(p*n/2^32), hole k owns integer picks [ceil(k*2^32/n), ceil((k+1)*2^32/n)).
    // Check each bucket's endpoints and size for every possible hole count on a 4x4 board.
    const universe: u64 = @as(u64, 1) << 32;
    var n: usize = 1;
    while (n <= 16) : (n += 1) {
        var total: u64 = 0;
        var min_width: u64 = universe;
        var max_width: u64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            const count: u64 = @intCast(n);
            const lo = ((@as(u64, @intCast(k)) * universe) + count - 1) / count;
            const hi = ((@as(u64, @intCast(k + 1)) * universe) + count - 1) / count;
            try std.testing.expectEqual(k, Spawn.slotFor(@intCast(lo), n));
            try std.testing.expectEqual(k, Spawn.slotFor(@intCast(hi - 1), n));
            const bucket_width = hi - lo;
            total += bucket_width;
            min_width = @min(min_width, bucket_width);
            max_width = @max(max_width, bucket_width);
        }
        try std.testing.expectEqual(universe, total);
        try std.testing.expect(max_width - min_width <= 1);
    }

    // If A has 4 holes and B has 2, the four equal-sized A buckets pair into two equal-sized B buckets.
    var lower: usize = 0;
    var upper: usize = 0;
    for (0..4) |a_slot| {
        const pick = Spawn.pickFor(a_slot, 4);
        switch (Spawn.slotFor(pick, 2)) {
            0 => lower += 1,
            1 => upper += 1,
            else => return error.TestUnexpectedResult,
        }
    }
    try std.testing.expectEqual(@as(usize, 2), lower);
    try std.testing.expectEqual(@as(usize, 2), upper);
}

test "recording with no tape capacity does not change the game or rng (#237)" {
    var init_rng = std.Random.DefaultPrng.init(37);
    const net = Net.init(init_rng.random());
    const value: Value = .{ .single = &net };

    var empty: [0]Spawn = .{};
    var enough: [tape_capacity]Spawn = undefined;
    var no_capacity_rng = std.Random.DefaultPrng.init(77);
    var enough_capacity_rng = std.Random.DefaultPrng.init(77);
    var no_capacity = Tape.initRecord(&empty);
    var enough_capacity = Tape.initRecord(&enough);
    const short = playTapedValue(value, no_capacity_rng.random(), .greedy, &no_capacity);
    const recorded = playTapedValue(value, enough_capacity_rng.random(), .greedy, &enough_capacity);

    try std.testing.expectEqual(recorded, short);
    try std.testing.expectEqual(@as(usize, 0), no_capacity.len);
    try std.testing.expect(enough_capacity.len > 0);
    try std.testing.expectEqual(no_capacity_rng.random().int(u64), enough_capacity_rng.random().int(u64));
}

test "recording preserves the raw common pick across different hole counts (#237)" {
    const ByteStream = struct {
        bytes: [5]u8,
        cursor: usize = 0,

        fn fill(self: *@This(), output: []u8) void {
            for (output) |*byte| {
                byte.* = self.bytes[self.cursor];
                self.cursor += 1;
            }
        }
    };

    const picks = [_]u32{ 0x7000_0000, 0x9000_0000 };
    var a_slots: [2]usize = undefined;
    var b_slots: [2]usize = undefined;
    for (picks, 0..) |pick, i| {
        // A has 3 holes and B has 2. These picks select the same A hole but different B holes.
        var state_a = GameState.initEmpty();
        for (0..core.BoardSize) |row| for (0..core.BoardSize) |col| {
            if (!(row == 0 and col < 3)) state_a.board[row][col] = 2;
        };
        const bytes = [_]u8{
            @truncate(pick),
            @truncate(pick >> 8),
            @truncate(pick >> 16),
            @truncate(pick >> 24),
            27, // bounded tile-value draw returns 1 and spawns a 2
        };
        var source = ByteStream{ .bytes = bytes };
        const rng = std.Random.init(&source, ByteStream.fill);
        var spawns: [1]Spawn = undefined;
        var record = Tape.initRecord(&spawns);
        try std.testing.expect(record.spawn(&state_a, rng));
        try std.testing.expectEqual(@as(usize, 1), record.len);
        try std.testing.expectEqual(pick, spawns[0].pick);
        a_slots[i] = Spawn.slotFor(spawns[0].pick, 3);

        var state_b = GameState.initEmpty();
        for (0..core.BoardSize) |row| for (0..core.BoardSize) |col| {
            if (!(row == 0 and col < 2)) state_b.board[row][col] = 2;
        };
        var replay = Tape.initReplay(&spawns);
        var replay_rng = std.Random.DefaultPrng.init(9);
        try std.testing.expect(replay.spawn(&state_b, replay_rng.random()));
        b_slots[i] = Spawn.slotFor(spawns[0].pick, 2);
        try std.testing.expectEqual(@as(core.Tile, 2), state_b.board[0][b_slots[i]]);
    }
    try std.testing.expectEqual(a_slots[0], a_slots[1]);
    try std.testing.expect(a_slots[0] == 1);
    try std.testing.expect(b_slots[0] != b_slots[1]);
}

test "pairStats gives the paired mean and standard error over games" {
    const diffs = [_]f64{ 100, 200, 300 };
    const r = pairStats(&diffs);
    try std.testing.expectEqual(@as(usize, 3), r.games);
    try std.testing.expectApproxEqAbs(@as(f64, 200), r.mean, 1e-12);
    // sample variance 10,000 over 3 games: stderr = sqrt(10,000 / 3)
    try std.testing.expectApproxEqAbs(@sqrt(10000.0 / 3.0), r.stderr, 1e-12);
    try std.testing.expectEqual(@as(usize, 3), r.better);
    try std.testing.expectEqual(@as(usize, 0), r.worse);

    // signs: B wins on one game, loses on two, ties on one and does not count it
    const signed = [_]f64{ -5000, 2500, -1000, 0 };
    const s = pairStats(&signed);
    try std.testing.expectApproxEqAbs(@as(f64, -875), s.mean, 1e-12);
    try std.testing.expectEqual(@as(usize, 1), s.better);
    try std.testing.expectEqual(@as(usize, 2), s.worse);

    // an empty run has no statistics and must not divide by zero
    const none = pairStats(&.{});
    try std.testing.expectEqual(@as(usize, 0), none.games);
    try std.testing.expectEqual(@as(f64, 0), none.mean);
    try std.testing.expectEqual(@as(f64, 0), none.stderr);
}

test "Teachers round-trips boards and targets through the file format (#175)" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/teachers.tch", .{tmp.sub_path});
    defer std.testing.allocator.free(path);

    var board_a: Board = @splat(@splat(0));
    board_a[0][0] = 4096;
    board_a[3][3] = 512;
    var board_b: Board = @splat(@splat(0));
    board_b[1][2] = 8192;
    board_b[1][3] = 2;
    const teachers = try std.testing.allocator.create(Teachers);
    defer std.testing.allocator.destroy(teachers);
    teachers.* = .{};
    teachers.append(board_a, 1.5);
    teachers.append(board_b, -0.25);
    try teachers.save(io, std.testing.allocator, path);

    var back = try std.testing.allocator.create(Teachers);
    defer std.testing.allocator.destroy(back);
    back.* = .{};
    try back.load(io, std.testing.allocator, path);
    try std.testing.expectEqual(@as(usize, 2), back.len);
    try std.testing.expectEqual(board_a, back.boardAt(0));
    try std.testing.expectEqual(board_b, back.boardAt(1));
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), back.targetAt(0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), back.targetAt(1), 1e-6);

    // header, then 20 bytes per entry
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .limited(1 << 20));
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(teacher_header_len + 2 * teacher_entry_len, bytes.len);
    try std.testing.expectEqualStrings(teacher_magic, bytes[0..8]);
}

test "Teachers encodes the fixed boards and targets to the same bytes as before the refactor" {
    const teachers = try std.testing.allocator.create(Teachers);
    defer std.testing.allocator.destroy(teachers);
    teachers.* = .{};
    var board_a: Board = @splat(@splat(0));
    board_a[0][0] = 4096;
    board_a[3][3] = 512;
    var board_b: Board = @splat(@splat(0));
    board_b[1][2] = 8192;
    board_b[1][3] = 2;
    teachers.append(board_a, 1.5);
    teachers.append(board_b, -0.25);

    // magic, count 2, then each 16-byte log2 board with the f32 target after it — recorded before the
    // cursor refactor (1.5 = 0x3fc00000, -0.25 = 0xbe800000).
    const golden = [_]u8{
        0x32, 0x30, 0x34, 0x38, 0x2d, 0x54, 0x43, 0x48, 0x02, 0x00, 0x00, 0x00,
        0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x0d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0xbe,
    };
    var bytes: [golden.len]u8 = undefined;
    teachers.encodeInto(&bytes);
    try std.testing.expectEqualSlices(u8, &golden, &bytes);
}

/// `Teachers` is ~40 MB (32 dumps), so tests heap allocate it, like `nn_main` does.
fn testTeachers() !*Teachers {
    const t = try std.testing.allocator.create(Teachers);
    t.* = .{};
    return t;
}

test "Teachers parse rejects a wrong magic, a truncated file and an oversized count (#175)" {
    const buffer = try testTeachers();
    defer std.testing.allocator.destroy(buffer);
    var bytes: [teacher_header_len + teacher_entry_len]u8 = @splat(0);
    @memcpy(bytes[0..8], "2048-XXX");
    try std.testing.expectError(error.BadTeacherMagic, buffer.parse(&bytes));
    @memcpy(bytes[0..8], teacher_magic);
    std.mem.writeInt(u32, bytes[8..12], 1, .little);
    try std.testing.expectError(error.WrongTeacherFileSize, buffer.parse(bytes[0 .. bytes.len - 1]));
    std.mem.writeInt(u32, bytes[8..12], @intCast(teacher_cap + 1), .little);
    try std.testing.expectError(error.TeacherTooLarge, buffer.parse(&bytes));
}

test "computeTeachers prices each stored afterstate with one spawn (#175)" {
    var prng = std.Random.DefaultPrng.init(71);
    const rng = prng.random();
    const net = Net.init(rng);
    var dump: Dump = .{};
    var deep: Board = @splat(@splat(0));
    deep[0][0] = 4096;
    deep[0][1] = 4096;
    dump.append(deep);
    var pass_a = std.Random.DefaultPrng.init(73);
    const buffer = try testTeachers();
    defer std.testing.allocator.destroy(buffer);
    computeTeachers(&net, buffer, &dump, pass_a.random(), .deep3);
    try std.testing.expectEqual(@as(usize, 1), buffer.len);
    // the sample's input is the stored afterstate itself: `V` is only ever consumed on afterstates, so
    // the spawn belongs to the target's search and not to the board the gradient sees
    try std.testing.expectEqual(deep, buffer.boardAt(0));
    // ... and the target is priced one spawn deeper than the input board
    var priced = GameState{ .board = deep, .score = 0 };
    var pass_c = std.Random.DefaultPrng.init(73);
    _ = priced.spawnRandomTile(pass_c.random());
    try std.testing.expectApproxEqAbs(
        teacherValue(.{ .single = &net }, priced, .deep3, false),
        buffer.targetAt(0),
        1e-6,
    );
    // the same seed reproduces the same pass
    const again = try testTeachers();
    defer std.testing.allocator.destroy(again);
    var pass_b = std.Random.DefaultPrng.init(73);
    computeTeachers(&net, again, &dump, pass_b.random(), .deep3);
    try std.testing.expectEqual(buffer.boardAt(0), again.boardAt(0));
    try std.testing.expectApproxEqAbs(buffer.targetAt(0), again.targetAt(0), 1e-12);
}

test "dist takes one regression step per self-play step on average (#175)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(97);
    const rng = prng.random();
    const net = Net.init(rng);
    const buffer = try testTeachers();
    defer std.testing.allocator.destroy(buffer);
    var board: Board = @splat(@splat(0));
    board[0][0] = 4096;
    buffer.append(board, 2.0);
    buffer.append(board, 0.5);
    const opts: Options = .{ .distill = buffer };
    var rng_a = std.Random.DefaultPrng.init(11);
    var rng_b = std.Random.DefaultPrng.init(11);
    var a = net;
    var b = net;
    const game = try trainGameOpts(&a, rng_a.random(), 0.001, null, opts);
    const again = try trainGameOpts(&b, rng_b.random(), 0.001, null, opts);
    try std.testing.expectEqual(game.moves, again.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    // the mix is observable and is not the plain path: draws > 0 and the net moved
    try std.testing.expect(buffer.draws > 0);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&net), std.mem.asBytes(&a)));
    // one regression step per move, and the drawn board is the afterstate the dump stored (not the
    // post-spawn completion), so `dist` labels `V` on the same domain the policy and the TD half use
    try std.testing.expectEqual(game.moves * 2, buffer.draws);
    // without the buffer the path is the plain one (the games differ: `dist` draws one extra rng value
    // per move for the teacher step, so only the net is comparable, not the move count)
    var plain = net;
    var rng_c = std.Random.DefaultPrng.init(11);
    _ = try trainGameOpts(&plain, rng_c.random(), 0.001, null, .{});
    try std.testing.expectEqual(game.moves * 2, buffer.draws);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&plain)));
}

test "distill_every thins the teacher mix and distill_lr does not change the TD half (#206)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(98);
    const rng = prng.random();
    const net = Net.init(rng);
    var board: Board = @splat(@splat(0));
    board[0][0] = 4096;
    // `distill_every = 4`: one regression per four self-play steps, so the draws are about moves/4
    const thinned = try testTeachers();
    defer std.testing.allocator.destroy(thinned);
    thinned.append(board, 2.0);
    var thinned_net = net;
    var rng_a = std.Random.DefaultPrng.init(11);
    const thinned_game = try trainGameOpts(&thinned_net, rng_a.random(), 0.001, null, .{ .distill = thinned, .distill_every = 4 });
    try std.testing.expect(thinned.draws > 0);
    try std.testing.expect(thinned.draws <= thinned_game.moves);
    // every = 1 is the dense mix, so the same game draws strictly more teacher steps than every = 4
    const dense = try testTeachers();
    defer std.testing.allocator.destroy(dense);
    dense.append(board, 2.0);
    var dense_net = net;
    var rng_b = std.Random.DefaultPrng.init(11);
    const dense_game = try trainGameOpts(&dense_net, rng_b.random(), 0.001, null, .{ .distill = dense, .distill_every = 1 });
    try std.testing.expectEqual(dense_game.moves, dense.draws);
    try std.testing.expect(dense.draws > thinned.draws);
    // `distill_lr` only rescales the teacher step: with a teacher buffer that is never drawn the TD half
    // is bit-identical (the rate is read only inside `distStep`)
    const empty = try testTeachers();
    defer std.testing.allocator.destroy(empty);
    var a = net;
    var b = net;
    var rng_c = std.Random.DefaultPrng.init(11);
    var rng_d = std.Random.DefaultPrng.init(11);
    _ = try trainGameOpts(&a, rng_c.random(), 0.001, null, .{ .distill = empty });
    _ = try trainGameOpts(&b, rng_d.random(), 0.001, null, .{ .distill = empty, .distill_lr = 1e-5 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

/// test scorers: +1 for an afterstate that kept a tile in the top-left cell (the `left` move of
/// `post_spawn_board`), 0 otherwise.
fn rerankPrefersLeft(board: Board) f64 {
    return if (board[0][0] != 0) 1 else 0;
}

/// test scorer: +1 for an afterstate whose top-left cell is empty (the `right` move).
fn rerankPrefersRight(board: Board) f64 {
    return if (board[0][0] == 0) 1 else 0;
}

test "rerank = null keeps the plain 2-ply argmax (#189)" {
    const net = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &net };
    var plain_stats: Stats = .{};
    var plain_cache: ValueCache = .{};
    const plain = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &plain_stats, &plain_cache, .{});
    try std.testing.expect(plain != null);
    // the explicit null and the plain default are the same move, with the same search work
    var null_stats: Stats = .{};
    var null_cache: ValueCache = .{};
    const nulled = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &null_stats, &null_cache, .{ .rerank = null });
    try std.testing.expectEqual(plain, nulled);
    try std.testing.expectEqual(plain_stats.leaves, null_stats.leaves);
    try std.testing.expectEqual(bestMove(&net, mid_board, .deep), nulled);
}

test "rerank with margin 0 and weight 0 keeps the plain 2-ply argmax (#189)" {
    const net = try parse(@embedFile("nn_weights.bin"));
    const vfn: Value = .{ .single = &net };
    var plain_stats: Stats = .{};
    var plain_cache: ValueCache = .{};
    const plain = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &plain_stats, &plain_cache, .{});
    // margin 0 is only the exact tie, weight 0 is inert: neither may change the move
    var zero_stats: Stats = .{};
    var zero_cache: ValueCache = .{};
    const zero = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &zero_stats, &zero_cache, .{
        .rerank = .{ .margin = 0, .weight = 0, .score = &rerankPrefersRight },
    });
    try std.testing.expectEqual(plain, zero);
    // a margin that covers every move is still inert while the weight is 0
    var wide_stats: Stats = .{};
    var wide_cache: ValueCache = .{};
    const wide = bestMoveWithCacheValueOpts(vfn, mid_board, .deep, &wide_stats, &wide_cache, .{
        .rerank = .{ .margin = 1e9, .weight = 0, .score = &rerankPrefersRight },
    });
    try std.testing.expectEqual(plain, wide);
    // and no extra search work: the rerank re-scores afterstates, it never touches the tree
    try std.testing.expectEqual(plain_stats.leaves, wide_stats.leaves);
}

test "rerank flips a near tie when the foreign score prefers the second move (#189)" {
    // zero net: every afterstate is worth 0, so the 2-ply value of a move is its reward plus the
    // rewards of the tree below it. `post_spawn_board` is full and its `left` and `right` afterstates
    // are mirror images, so the two values tie (up to float noise) — a gap far inside any margin.
    const net = std.mem.zeroes(Net);
    const vfn: Value = .{ .single = &net };
    const state: GameState = .{ .board = post_spawn_board };
    var gap_cache: ValueCache = .{};
    var gap_search: Search = .{ .vfn = vfn, .cache = &gap_cache, .ply = 2 };
    var left_state = state;
    const left_moved = left_state.slide(.left);
    var right_state = state;
    const right_moved = right_state.slide(.right);
    try std.testing.expect(left_moved.changed and right_moved.changed);
    const left_value = @as(f64, @floatFromInt(left_moved.score_delta)) / reward_scale + gap_search.chance(left_state, 2, 1);
    const right_value = @as(f64, @floatFromInt(right_moved.score_delta)) / reward_scale + gap_search.chance(right_state, 2, 1);
    const margin: f64 = 0.05;
    try std.testing.expect(@abs(left_value - right_value) <= margin);

    // the plain search picks one of the two; the rerank then lets the foreign score decide, so each
    // scorer wins its own move and the two reranks disagree (the NN alone cannot)
    var base_cache: ValueCache = .{};
    var base_stats: Stats = .{};
    const base = bestMoveWithCacheValueOpts(vfn, state, .deep, &base_stats, &base_cache, .{});
    try std.testing.expect(base == .left or base == .right);
    var left_cache: ValueCache = .{};
    var left_stats: Stats = .{};
    const flipped_left = bestMoveWithCacheValueOpts(vfn, state, .deep, &left_stats, &left_cache, .{
        .rerank = .{ .margin = margin, .weight = 1, .score = &rerankPrefersLeft },
    });
    var right_cache: ValueCache = .{};
    var right_stats: Stats = .{};
    const flipped_right = bestMoveWithCacheValueOpts(vfn, state, .deep, &right_stats, &right_cache, .{
        .rerank = .{ .margin = margin, .weight = 1, .score = &rerankPrefersRight },
    });
    try std.testing.expectEqual(@as(?Direction, .left), flipped_left);
    try std.testing.expectEqual(@as(?Direction, .right), flipped_right);
    // a margin strictly below the gap never reranks: the plain pick survives both scorers
    const tight_margin = @abs(left_value - right_value) - 1;
    var tight_left_cache: ValueCache = .{};
    var tight_left_stats: Stats = .{};
    const tight_left = bestMoveWithCacheValueOpts(vfn, state, .deep, &tight_left_stats, &tight_left_cache, .{
        .rerank = .{ .margin = tight_margin, .weight = 1, .score = &rerankPrefersLeft },
    });
    var tight_right_cache: ValueCache = .{};
    var tight_right_stats: Stats = .{};
    const tight_right = bestMoveWithCacheValueOpts(vfn, state, .deep, &tight_right_stats, &tight_right_cache, .{
        .rerank = .{ .margin = tight_margin, .weight = 1, .score = &rerankPrefersRight },
    });
    try std.testing.expectEqual(base, tight_left);
    try std.testing.expectEqual(base, tight_right);
}

test "the teacher cap holds 32 dumps and appendBuffer merges two passes (#207)" {
    try std.testing.expectEqual(32 * dump_cap, teacher_cap);
    const gpa = std.testing.allocator;
    const a = try gpa.create(Teachers);
    defer gpa.destroy(a);
    const b = try gpa.create(Teachers);
    defer gpa.destroy(b);
    a.len = 2;
    a.targets[0] = 1.5;
    a.targets[1] = -2.0;
    b.len = 3;
    b.targets[0] = 10;
    b.targets[1] = 20;
    b.targets[2] = 30;
    try std.testing.expectEqual(@as(usize, 0), a.appendBuffer(b));
    try std.testing.expectEqual(@as(usize, 5), a.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10), a.targets[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 30), a.targets[4], 1e-6);
    // overflowing the cap keeps the first `teacher_cap` samples and reports the rest
    a.len = teacher_cap - 1;
    try std.testing.expectEqual(@as(usize, 2), a.appendBuffer(b));
    try std.testing.expectEqual(teacher_cap, a.len);
}

test "a teacher file may hold more than one dump worth of samples (#207)" {
    const count = dump_cap + 10;
    const len = teacher_header_len + count * teacher_entry_len;
    const bytes = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(bytes);
    @memset(bytes, 0);
    @memcpy(bytes[0..8], teacher_magic);
    std.mem.writeInt(u32, bytes[8..12], @intCast(count), .little);
    const t = try std.testing.allocator.create(Teachers);
    defer std.testing.allocator.destroy(t);
    try t.parse(bytes);
    try std.testing.expectEqual(count, t.len);
}

test "global features are normalised and describe the board (#226)" {
    const empty: Board = std.mem.zeroes(Board);
    const f0 = featureValues(empty);
    try std.testing.expectEqual(@as(f64, 1), f0[0]);
    try std.testing.expectEqual(@as(f64, 0), f0[1]);
    try std.testing.expectEqual(@as(f64, 1), f0[2]);
    try std.testing.expectEqual(@as(f64, 1), f0[3]);
    try std.testing.expectEqual(@as(f64, 0), f0[4]);
    try std.testing.expectEqual(@as(f64, 0), f0[5]);

    // the max tile in a corner is the one signal the one-hot input cannot place cheaply
    var corner: Board = std.mem.zeroes(Board);
    corner[0][0] = 2048;
    const f1 = featureValues(corner);
    try std.testing.expectEqual(@as(f64, 1), f1[4]);
    try std.testing.expect(f1[0] < f0[0]);
    try std.testing.expect(f1[1] > 0.7);

    var middle: Board = std.mem.zeroes(Board);
    middle[1][1] = 2048;
    try std.testing.expectEqual(@as(f64, 0), featureValues(middle)[4]);

    // a monotone row is smoother than an alternating one
    var mono: Board = std.mem.zeroes(Board);
    mono[0] = .{ 2, 4, 8, 16 };
    var rough: Board = std.mem.zeroes(Board);
    rough[0] = .{ 2, 1024, 4, 2048 };
    try std.testing.expect(featureValues(mono)[3] > featureValues(rough)[3]);

    // every feature stays inside 0..1 for an arbitrary mid-game board
    var mid: Board = std.mem.zeroes(Board);
    mid[0] = .{ 4, 8, 2, 2 };
    mid[1] = .{ 512, 16, 4, 0 };
    mid[2] = .{ 64, 8, 128, 2 };
    mid[3] = .{ 1024, 32, 2, 4 };
    for (featureValues(mid)) |f| try std.testing.expect(f >= 0 and f <= 1);
}

test "empty-cell potential has the #277 endpoints and scale" {
    try std.testing.expectEqual(@as(f64, 0), shapePotential(boardWithEmpties(0), 1));
    try std.testing.expectEqual(@as(f64, 1), shapePotential(boardWithEmpties(15), 1));
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), shapePotential(boardWithEmpties(3), 1), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), shapePotential(boardWithEmpties(3), 1.25), 1e-15);
}

test "shaped non-terminal and terminal targets follow the potential formula (#277)" {
    var init_rng = std.Random.DefaultPrng.init(277);
    var net = Net.init(init_rng.random());
    var cache: Cache = .{};
    var trace = net;
    var rng = std.Random.DefaultPrng.init(278);
    const c = 0.75;
    const p: Pick = .{ .dir = .left, .after = boardWithEmpties(8), .reward = 0.25, .value = 1 };
    const next: Pick = .{ .dir = .right, .after = boardWithEmpties(5), .reward = 0.5, .value = 3.5 };
    const spawned: GameState = .{ .board = boardWithEmpties(6) };

    // : reward + Phi(next.after) - Phi(p.after) + V(next.after), with V = next.value on td1/greedy1.
    const want_nonterminal = p.reward + shapePotential(next.after, c) - shapePotential(p.after, c) + next.value;
    const got_nonterminal = tdUpdate(&net, &cache, p, next, spawned, 0, rng.random(), .{ .shape_c = c }, &trace);
    try std.testing.expectApproxEqAbs(want_nonterminal, got_nonterminal, 1e-12);

    // The terminal afterstate has no next board: its potential is 0, leaving reward - Phi(p.after).
    const want_terminal = p.reward - shapePotential(p.after, c);
    const got_terminal = tdTerminalUpdate(&net, &cache, p, spawned, 0, rng.random(), .{ .shape_c = c }, &trace);
    try std.testing.expectApproxEqAbs(want_terminal, got_terminal, 1e-12);
}

test "shape_c zero keeps the raw reward bit-identical (#277)" {
    var init_rng = std.Random.DefaultPrng.init(277);
    var net = Net.init(init_rng.random());
    var cache: Cache = .{};
    var trace = net;
    var rng = std.Random.DefaultPrng.init(279);
    const p: Pick = .{ .dir = .left, .after = boardWithEmpties(8), .reward = 0.25, .value = 1 };
    const next: Pick = .{ .dir = .right, .after = boardWithEmpties(5), .reward = 0.5, .value = 3.5 };
    const spawned: GameState = .{ .board = boardWithEmpties(6) };

    const plain = tdUpdate(&net, &cache, p, next, spawned, 0, rng.random(), .{}, &trace);
    const zero = tdUpdate(&net, &cache, p, next, spawned, 0, rng.random(), .{ .shape_c = 0 }, &trace);
    try std.testing.expectEqual(plain, zero);
    try std.testing.expectEqual(p.reward + next.value, zero);

    const plain_terminal = tdTerminalUpdate(&net, &cache, p, spawned, 0, rng.random(), .{}, &trace);
    const zero_terminal = tdTerminalUpdate(&net, &cache, p, spawned, 0, rng.random(), .{ .shape_c = 0 }, &trace);
    try std.testing.expectEqual(plain_terminal, zero_terminal);
    try std.testing.expectEqual(p.reward, zero_terminal);
}

test "par accepts shape and matches the serial games (#277)" {
    if (!width.slow_tests) return error.SkipZigTest;
    const games = 2;
    const opts = Options{ .shape_c = 0.5 };
    var init_rng = std.Random.DefaultPrng.init(277);
    var base = Net.init(init_rng.random());

    var units: [games]Net = undefined;
    for (&units, 0..) |*u, i| {
        u.* = base;
        var rng = std.Random.DefaultPrng.init(277 + i);
        _ = try trainGameOpts(u, rng.random(), 0.001, null, opts);
    }
    const flat_len = @sizeOf(Net) / @sizeOf(f64);
    var sums: [flat_len]f64 = @splat(0);
    const start = flat(&base);
    for (&units) |*u| {
        for (flat(u), start, 0..) |v, s, k| sums[k] += v - s;
    }
    var serial = base;
    for (flat(&serial), start, sums) |*w, s, sum| w.* = s + sum / games;

    var par = base;
    _ = try trainBatchOpts(&par, 277, 0, games, 0.001, null, opts);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&serial), std.mem.asBytes(&par));
}

test "explicit shape_c zero trains exactly like base, serial and par (#277)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(277);
    const base = Net.init(init_rng.random());

    var plain = base;
    var zero = base;
    for (0..2) |i| {
        var plain_rng = std.Random.DefaultPrng.init(277 + i);
        var zero_rng = std.Random.DefaultPrng.init(277 + i);
        _ = trainGame(&plain, plain_rng.random(), 0.001, null);
        _ = try trainGameOpts(&zero, zero_rng.random(), 0.001, null, .{ .shape_c = 0 });
    }
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain), std.mem.asBytes(&zero));

    var plain_par = base;
    var zero_par = base;
    _ = trainBatch(&plain_par, 277, 0, 3, 0.001, null);
    _ = try trainBatchOpts(&zero_par, 277, 0, 3, 0.001, null, .{ .shape_c = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&plain_par), std.mem.asBytes(&zero_par));
}

test "PopArt refuses mixed training variants and stays opt-in (#289)" {
    var state: PopArtState = .{};
    const adam = try std.testing.allocator.create(Adam);
    defer std.testing.allocator.destroy(adam);
    adam.* = .{};
    try validateOptions(.{});
    try validateOptions(.{ .popart = &state });
    try std.testing.expectError(error.PopArtRequiresPlainTd1, validateOptions(.{ .popart = &state, .lambda = 0.7 }));
    try std.testing.expectError(error.PopArtRequiresPlainTd1, validateOptions(.{ .popart = &state, .step_clip = 1 }));
    try std.testing.expectError(error.PopArtRequiresPlainTd1, validateOptions(.{ .popart = &state, .adam = adam }));
    // a search-only field (the spawn sampler) is inert on this path, so it is refused, not ignored
    try std.testing.expectError(error.PopArtRequiresPlainTd1, validateOptions(.{ .popart = &state, .sample = 2 }));
    // Per-game safety bounds do not alter an update, and stay allowed with PopArt.
    try validateOptions(.{ .popart = &state, .move_cap = 500, .game_time_limit_ns = 0 });
    try std.testing.expectError(error.PopArtRequiresPlainTd1, validateBatchOptions(.{ .popart = &state }));
}

test "shape refuses variants that would drop the shaped reward (#277)" {
    if (!width.slow_tests) return error.SkipZigTest;
    var init_rng = std.Random.DefaultPrng.init(277);
    try validateOptions(.{});
    try validateOptions(.{ .shape_c = 0, .replay_k = 30, .backward = true });
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .replay_k = 30 }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .nstep = 4 }));
    var target_net = Net.init(init_rng.random());
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .target_net = &target_net }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .teacher = .deep }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .backward = true }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .death = true }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .rsym = true }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .lambda = 0.7 }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .target = .search2 }));
    try std.testing.expectError(error.ShapeRequiresPlainTd1, validateOptions(.{ .shape_c = 1, .selfplay = .deep2 }));

    var net = Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(280);
    try std.testing.expectError(error.ShapeRequiresPlainTd1, trainGameOpts(&net, rng.random(), 0.001, null, .{ .shape_c = 1, .replay_k = 30 }));
}

/// `V(board)` = number of empty cells: one hidden unit sums the level-0 input rows and the output reads
/// that unit alone, so the expected-spawn values below are hand-computable integers.
fn emptyCountNet() Net {
    var net = std.mem.zeroes(Net);
    for (0..core.BoardCellCount) |cell| net.w1[cell * levels][0] = 1;
    net.w2[0][0] = 1;
    net.w3[0] = 1;
    return net;
}

test "expected spawn prices both tiles of the greedy 1-ply reply (#321)" {
    const net = emptyCountNet();
    // One empty cell at (0,0), where a 2 spawn merges with (1,0) (up or down, 4 points) and a 4 spawn
    // merges with (0,1) (left or right, 8 points). No other pair merges, and V counts empty cells, so
    // the greedy reply is `4/1000 + 1` and `8/1000 + 1` and the cell is certain.
    const after = Board{
        .{ 0, 4, 8, 16 },
        .{ 2, 32, 64, 128 },
        .{ 8, 64, 128, 256 },
        .{ 16, 128, 256, 512 },
    };
    try std.testing.expectApproxEqAbs(@as(f64, 0.9 * (0.004 + 1) + 0.1 * (0.008 + 1)), expectedSpawnValue(&net, after), 1e-12);

    // The pending reward is added once, on top of the expectation, and the sampled continuation is
    // ignored: `next` is a board no move of `after` reaches.
    var trainable = net;
    var cache: Cache = .{};
    var trace = std.mem.zeroes(Net);
    const p: Pick = .{ .dir = .left, .after = after, .reward = 0.25, .value = 0 };
    const next: Pick = .{ .dir = .up, .after = boardWithEmpties(0), .reward = 9, .value = 9 };
    var rng = std.Random.DefaultPrng.init(321);
    const target = tdUpdate(&trainable, &cache, p, next, .{ .board = boardWithEmpties(9) }, 0, rng.random(), .{ .expected_spawn = true }, &trace);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25 + 1.0044), target, 1e-12);
}

test "expected spawn prices the alive spawns a terminal move still had (#321)" {
    const net = emptyCountNet();
    // The game died on the 2 spawn: the board below with a 2 at (3,3) is full and has no pair, so the
    // sampled terminal target is the reward alone. The 4 spawn merges with (3,2) for 8 points and leaves
    // one empty cell behind (V = 1), so its 0.1 share is `0.008 + 1`.
    const after = Board{
        .{ 8, 16, 32, 64 },
        .{ 128, 256, 512, 1024 },
        .{ 8, 16, 32, 64 },
        .{ 128, 256, 4, 0 },
    };
    try std.testing.expectApproxEqAbs(@as(f64, 0.9 * 0 + 0.1 * (0.008 + 1)), expectedSpawnValue(&net, after), 1e-12);

    var cache: Cache = .{};
    var trace = std.mem.zeroes(Net);
    const p: Pick = .{ .dir = .up, .after = after, .reward = 0.016, .value = 0 };
    var rng = std.Random.DefaultPrng.init(321);
    var trainable = net;
    const expected = tdTerminalUpdate(&trainable, &cache, p, .{ .board = after }, 0, rng.random(), .{ .expected_spawn = true }, &trace);
    try std.testing.expectApproxEqAbs(@as(f64, 0.016 + 0.1008), expected, 1e-12);
    // the sampled path is unchanged: the reward alone
    var plain = net;
    try std.testing.expectEqual(@as(f64, 0.016), tdTerminalUpdate(&plain, &cache, p, .{ .board = after }, 0, rng.random(), .{}, &trace));
}

test "expected spawn ignores the sampled continuation and draws no tile (#321)" {
    const net = emptyCountNet();
    const after = Board{
        .{ 0, 4, 8, 16 },
        .{ 2, 32, 64, 128 },
        .{ 8, 64, 128, 256 },
        .{ 16, 128, 256, 512 },
    };
    const p: Pick = .{ .dir = .left, .after = after, .reward = 0.004, .value = 0 };
    var cache: Cache = .{};
    var trace = std.mem.zeroes(Net);
    var used = std.Random.DefaultPrng.init(11);
    var fresh = std.Random.DefaultPrng.init(11);
    var a = net;
    var b = net;
    // Two targets of the same step: only the sampled continuation and its board differ.
    const left: Pick = .{ .dir = .left, .after = boardWithEmpties(4), .reward = 0.5, .value = 2 };
    const right: Pick = .{ .dir = .right, .after = boardWithEmpties(6), .reward = -1, .value = -3 };
    const first = tdUpdate(&a, &cache, p, left, .{ .board = boardWithEmpties(4) }, 0, used.random(), .{ .expected_spawn = true }, &trace);
    const second = tdUpdate(&b, &cache, p, right, .{ .board = boardWithEmpties(6) }, 0, used.random(), .{ .expected_spawn = true }, &trace);
    try std.testing.expectEqual(first, second);
    // The enumeration read the net only, so the game's spawn rng is exactly where the seed left it.
    try std.testing.expectEqual(fresh.random().int(u64), used.random().int(u64));
}

/// The greedy 1-ply reply of a post-spawn board, re-derived from the policy the sampled target uses.
fn greedyReplyValue(net: *const Net, state: GameState) f64 {
    const pick = pickMove(net, state, false) orelse return 0;
    return pick.reward + pick.value;
}

test "expected spawn averages every spawn outcome with weights that sum to one (#321)" {
    var init_rng = std.Random.DefaultPrng.init(321);
    const net = Net.init(init_rng.random());
    const board = Board{
        .{ 2, 4, 8, 0 },
        .{ 16, 32, 64, 0 },
        .{ 128, 256, 512, 1024 },
        .{ 4, 8, 16, 32 },
    };
    // Reference: two cells at 1/2 each, the tile at 0.9/0.1, the greedy reply as the branch value.
    // Any weight that does not normalise to 1 (a missing 1/empties, a doubled cell) moves the sum.
    var total: f64 = 0;
    var empties: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (board[r][c] != 0) continue;
            empties += 1;
            var child = GameState{ .board = board };
            child.board[r][c] = 2;
            total += 0.9 * greedyReplyValue(&net, child);
            child.board[r][c] = 4;
            total += 0.1 * greedyReplyValue(&net, child);
        }
    }
    try std.testing.expect(empties == 2);
    try std.testing.expectApproxEqRel(total / empties, expectedSpawnValue(&net, board), 1e-12);
}

test "expected spawn target equals the sampled target when the tile cannot matter (#321)" {
    const net = emptyCountNet();
    // One empty cell at (2,0) and no 2 or 4 anywhere, so the drawn tile can only fill that cell and no
    // move can tell the two tiles apart: both branches score the same 8+8 merge in row 0, 0.016 + 1.
    const after = Board{
        .{ 8, 8, 16, 32 },
        .{ 64, 128, 256, 512 },
        .{ 0, 8, 16, 32 },
        .{ 128, 256, 512, 1024 },
    };
    try std.testing.expectApproxEqAbs(@as(f64, 0.016 + 1), expectedSpawnValue(&net, after), 1e-12);

    // The sampled path: the game draws the tile, plays the best reply, and bootstraps reward + V.
    var spawned = GameState{ .board = after };
    var rng = std.Random.DefaultPrng.init(321);
    try std.testing.expect(spawned.spawnRandomTile(rng.random()));
    const next = pickMove(&net, spawned, false).?;
    const p: Pick = .{ .dir = .up, .after = after, .reward = 0.004, .value = 0 };
    var trainable = net;
    var cache: Cache = .{};
    var trace = std.mem.zeroes(Net);
    const expected = tdUpdate(&trainable, &cache, p, next, spawned, 0, rng.random(), .{ .expected_spawn = true }, &trace);
    try std.testing.expectApproxEqAbs(@as(f64, 0.004) + next.reward + next.value, expected, 1e-12);
}

test "expected spawn is opt-in and refuses variants it would silently ignore (#321)" {
    var pr: Priority = .{};
    var target_net = emptyCountNet();
    try validateOptions(.{});
    // a search-only field (the spawn sampler) is inert on this path, so it is refused, not ignored
    try validateOptions(.{ .expected_spawn = true });
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .sample = 2 }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .backward = true }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .replay_k = 30 }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .priority = &pr }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .target_net = &target_net }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .nstep = 4 }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .target = .search2 }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .selfplay = .deep2 }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .death = true }));
    try std.testing.expectError(error.ExpectedSpawnRequiresPlainTd1, validateOptions(.{ .expected_spawn = true, .lambda = 0.7 }));
    // Per-game safety bounds do not alter an update, and the target is stateless per game, so `par`
    // batches it like `base`.
    try validateOptions(.{ .expected_spawn = true, .move_cap = 500, .game_time_limit_ns = 0 });
    try validateBatchOptions(.{ .expected_spawn = true });
}
