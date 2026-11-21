const std = @import("std");
const core = @import("core");
const blob = @import("blob");
const game_limit = @import("game_limit");

const Board = core.Board;

/// Same cell encoding as `nn.zig`: log2 tile level, 0 = empty; tiles >= 2^15 share the top level.
pub const levels = 16;
/// Score deltas are divided by this before they become rewards (same scale as `nn.reward_scale`).
pub const reward_scale: f64 = 1000;

pub fn levelOf(value: core.Tile) usize {
    if (value == 0) return 0;
    return @min(levels - 1, std.math.log2_int(core.Tile, value));
}

pub const Pattern = struct {
    len: u8,
    cells: [6]u8,
};

/// Base patterns; the 8 dihedral transforms of each one are the full pattern set.
pub const patterns = [_]Pattern{
    .{ .len = 4, .cells = .{ 0, 1, 2, 3, 0, 0 } }, // row 0
    .{ .len = 4, .cells = .{ 4, 5, 6, 7, 0, 0 } }, // row 1
    .{ .len = 4, .cells = .{ 0, 1, 4, 5, 0, 0 } }, // 2x2 top-left
    .{ .len = 4, .cells = .{ 1, 2, 5, 6, 0, 0 } }, // 2x2 middle column
    .{ .len = 4, .cells = .{ 0, 5, 10, 15, 0, 0 } }, // main diagonal
    .{ .len = 6, .cells = .{ 0, 1, 2, 4, 5, 6 } }, // 2x3 top-left
    // : the 2x3 rectangles split into two translation orbits under the 8 dihedral transforms
    // (8 positions around the edge, 4 positions through the middle). The first 6-tuple above is the
    // `min` index of orbit 0 for every board, so orbit 1 needs exactly this one more table.
    .{ .len = 6, .cells = .{ 4, 5, 6, 8, 9, 10 } }, // 2x3 middle-left (orbit 1)
    // (#323) One non-rectangular 6-tuple: an L with arms 3 and 4 on a corner, top row cols 0-2 plus
    // left column rows 0-3. Its 8 transforms are the 4 corners x 2 arm orientations, an orbit no
    // rectangular tuple shares, so it is one more table reachable only through `Set.lshape`.
    .{ .len = 6, .cells = .{ 0, 1, 2, 4, 8, 12 } }, // L, arms 3x4 on the top-left corner
};
pub const pattern_count = patterns.len;

/// The 8 dihedral transforms of a cell index: rotations and mirrors of the 4x4 board.
pub fn transformCell(t: usize, cell: usize) usize {
    const r = cell / core.BoardSize;
    const c = cell % core.BoardSize;
    const s = core.BoardSize - 1;
    return switch (t) {
        0 => r * core.BoardSize + c,
        1 => c * core.BoardSize + (s - r),
        2 => (s - r) * core.BoardSize + (s - c),
        3 => (s - c) * core.BoardSize + r,
        4 => r * core.BoardSize + (s - c),
        5 => (s - r) * core.BoardSize + c,
        6 => c * core.BoardSize + r,
        else => (s - c) * core.BoardSize + (s - r),
    };
}

pub fn transformBoard(board: Board, t: usize) Board {
    var out: Board = undefined;
    for (0..core.BoardSize) |r| for (0..core.BoardSize) |c| {
        const dst = transformCell(t, r * core.BoardSize + c);
        out[dst / core.BoardSize][dst % core.BoardSize] = board[r][c];
    };
    return out;
}

/// One `(accumulator, shift)` pair: cell level `<< shift` goes into accumulator `acc`.
/// There is one accumulator per (base pattern, symmetry transform); the canonical read-out takes
/// their minimum, the opt-in summed read-out (#322) uses all of them.
pub const Slot = struct { acc: u8, shift: u5 };

const max_slots = 32;

const SlotTable = struct {
    items: [core.BoardCellCount][max_slots]Slot,
    counts: [core.BoardCellCount]u8,
};

/// cell -> (tuple, shift) lookup, built once at compile time so a board needs one 16-cell scan.
pub const slot_table = blk: {
    @setEvalBranchQuota(200_000);
    var t: SlotTable = undefined;
    for (&t.counts) |*c| c.* = 0;
    for (patterns, 0..) |pat, p| for (0..8) |o| for (0..pat.len) |j| {
        const cell = transformCell(o, pat.cells[j]);
        if (t.counts[cell] >= max_slots) @compileError("ntuple slot table overflow");
        t.items[cell][t.counts[cell]] = .{ .acc = @intCast(p * 8 + o), .shift = @intCast(4 * j) };
        t.counts[cell] += 1;
    };
    break :blk t;
};

/// One canonical table index per base pattern (shared by all 8 transforms of that pattern).
pub const Indices = [pattern_count]u32;

/// The 8 transform indices of every base pattern, in transform order.
pub const TransformIndices = [pattern_count][8]u32;

/// How the 8 dihedral transforms of a base pattern are read out of the shared table.
pub const Symmetry = enum {
    /// The smallest of the 8 transform indices, one entry per pattern (default; all shipped tables).
    canonical,
    /// All 8 transform indices, values summed; a duplicated index counts once per transform (#322).
    summed,
};

/// Both read-outs of one board, derived from the same 16-cell scan so a runner can pick one at
/// start-up without recomputing indices per move. Values come from `TupleNet.symmetry`.
pub const Entries = struct {
    canonical: Indices,
    summed: TransformIndices,
};

fn transformIndices(board: Board) TransformIndices {
    var acc: [pattern_count * 8]u32 = @splat(0);
    for (0..core.BoardCellCount) |cell| {
        const level: u32 = @intCast(levelOf(board[cell / core.BoardSize][cell % core.BoardSize]));
        for (slot_table.items[cell][0..slot_table.counts[cell]]) |slot| {
            acc[slot.acc] |= level << slot.shift;
        }
    }
    var out: TransformIndices = undefined;
    for (0..pattern_count) |p| out[p] = acc[p * 8 ..][0..8].*;
    return out;
}

pub fn entriesFor(board: Board) Entries {
    const summed = transformIndices(board);
    var canonical: Indices = undefined;
    for (0..pattern_count) |p| {
        var best = summed[p][0];
        for (summed[p][1..]) |idx| best = @min(best, idx);
        canonical[p] = best;
    }
    return .{ .canonical = canonical, .summed = summed };
}

pub fn indices(board: Board) Indices {
    return entriesFor(board).canonical;
}

pub fn tableLen(p: usize) usize {
    return @as(usize, 1) << @intCast(4 * patterns[p].len);
}

pub const total_entries = blk: {
    var n: usize = 0;
    for (patterns) |p| n += @as(usize, 1) << @intCast(4 * p.len);
    break :blk n;
};

pub const offsets = blk: {
    var off: [pattern_count]usize = @splat(0);
    for (patterns, 0..) |_, p| {
        if (p > 0) off[p] = off[p - 1] + tableLen(p - 1);
    }
    break :blk off;
};

/// Table set: `.light` keeps only the five 4-tuples, `.full` adds the first 2x3 6-tuple (the
/// committed `ntuple_weights.bin` and every existing policy), `.wide` adds both 2x3 orbits, and
/// `.lshape` adds the single L-shaped 6-tuple on top of `.wide` (#323).
pub const Set = enum { light, full, wide, lshape };

pub const light_count = 5;
pub const full_count = 6;
/// Both 2x3 orbits (#106). The L shape is appended after them so `.wide` tables keep their layout.
pub const wide_count = 7;

pub fn countOf(set: Set) usize {
    return switch (set) {
        .light => light_count,
        .full => full_count,
        .wide => wide_count,
        .lshape => pattern_count,
    };
}

fn entriesOf(n: usize) usize {
    var entries: usize = 0;
    for (0..n) |p| entries += tableLen(p);
    return entries;
}

pub fn entryCount(set: Set) usize {
    return entriesOf(countOf(set));
}

pub const TupleNet = struct {
    storage: []f32,
    /// Number of active patterns (a prefix of `patterns`); `.light` stops before the 6-tuple.
    count: usize,
    /// Read-out of the 8 dihedral transforms; `.canonical` is the shipped default (#322 opt-in).
    symmetry: Symmetry = .canonical,

    pub fn init(gpa: std.mem.Allocator) !TupleNet {
        return initWith(gpa, .full);
    }

    pub fn initWith(gpa: std.mem.Allocator, set: Set) !TupleNet {
        return initCount(gpa, countOf(set));
    }

    fn initCount(gpa: std.mem.Allocator, n: usize) !TupleNet {
        const storage = try gpa.alloc(f32, entriesOf(n));
        @memset(storage, 0);
        return .{ .storage = storage, .count = n };
    }

    pub fn deinit(self: *TupleNet, gpa: std.mem.Allocator) void {
        gpa.free(self.storage);
    }

    pub fn table(self: *TupleNet, p: usize) []f32 {
        return self.storage[offsets[p]..][0..tableLen(p)];
    }

    pub fn value(self: *const TupleNet, idx: Indices) f64 {
        var v: f64 = 0;
        for (0..self.count) |p| v += self.storage[offsets[p] + idx[p]];
        return v;
    }

    /// Summed read-out: every transform index of every active pattern, duplicates summed each time.
    pub fn valueSummed(self: *const TupleNet, all: TransformIndices) f64 {
        var v: f64 = 0;
        for (0..self.count) |p| {
            for (all[p]) |idx| v += self.storage[offsets[p] + idx];
        }
        return v;
    }

    /// Value of `entries` under this net's read-out (`symmetry`).
    pub fn valueOf(self: *const TupleNet, e: Entries) f64 {
        return switch (self.symmetry) {
            .canonical => self.value(e.canonical),
            .summed => self.valueSummed(e.summed),
        };
    }

    pub fn eval(self: *const TupleNet, board: Board) f64 {
        return self.valueOf(entriesFor(board));
    }

    /// Apply a precomputed TD error equally across each active tuple's touched weight.
    fn applyDelta(self: *TupleNet, idx: Indices, delta: f64, lr: f64) void {
        const step: f32 = @floatCast(lr * delta / @as(f64, @floatFromInt(self.count)));
        for (0..self.count) |p| self.storage[offsets[p] + idx[p]] += step;
    }

    /// Summed-read-out TD step: a gradient step on `valueSummed` with alpha = lr / (8 * n_tuples).
    /// Each of the 8 transform indices of a pattern moves by `lr * delta / (8 * n_tuples)`, so an
    /// index duplicated k times moves k times as far -- the multiplicity of the forward sum. A
    /// pattern moves by `lr * delta / n_tuples` in total (the canonical per-pattern rate) and the
    /// value by `lr * delta` per pattern when its 8 indices are distinct, so `lr` is not rescaled.
    fn applySummedDelta(self: *TupleNet, all: TransformIndices, delta: f64, lr: f64) void {
        const step: f32 = @floatCast(lr * delta / (@as(f64, @floatFromInt(self.count)) * 8));
        for (0..self.count) |p| {
            for (all[p]) |idx| self.storage[offsets[p] + idx] += step;
        }
    }

    /// TD step: move every touched entry by `lr * (target - V) / n_tuples`; returns the TD error.
    pub fn train(self: *TupleNet, idx: Indices, target: f64, lr: f64) f64 {
        const delta = target - self.value(idx);
        self.applyDelta(idx, delta, lr);
        return delta;
    }

    /// `train` on `entries`, with the update that matches this net's read-out.
    pub fn trainEntries(self: *TupleNet, e: Entries, target: f64, lr: f64) f64 {
        const delta = target - self.valueOf(e);
        switch (self.symmetry) {
            .canonical => self.applyDelta(e.canonical, delta, lr),
            .summed => self.applySummedDelta(e.summed, delta, lr),
        }
        return delta;
    }
};

/// Paper defaults for the opt-in delayed TD(lambda) learner.
pub const delayed_td_horizon: usize = 3;
pub const delayed_td_lambda: f64 = 0.5;

/// Buffer up to h+1 afterstates and their TD errors, then apply one aggregated update per state.
/// This is the delayed TD(lambda) algorithm: the default TD(0) training path remains separate.
pub const DelayedTDLambda = struct {
    net: *TupleNet,
    lr: f64,
    states: [delayed_td_horizon + 1]Indices = undefined,
    deltas: [delayed_td_horizon + 1]f64 = undefined,
    len: usize = 0,

    pub fn init(net: *TupleNet, lr: f64) DelayedTDLambda {
        return .{ .net = net, .lr = lr };
    }

    /// Add one afterstate's TD error; once h future errors are available, update it and discard it.
    pub fn observe(self: *DelayedTDLambda, idx: Indices, delta: f64) void {
        self.states[self.len] = idx;
        self.deltas[self.len] = delta;
        self.len += 1;
        if (self.len == delayed_td_horizon + 1) {
            self.applyOldest();
            self.dropOldest();
        }
    }

    /// Apply the remaining truncated traces at episode end, oldest state first.
    pub fn finish(self: *DelayedTDLambda) void {
        while (self.len > 0) {
            self.applyOldest();
            self.dropOldest();
        }
    }

    fn applyOldest(self: *DelayedTDLambda) void {
        var delta_sum: f64 = 0;
        var decay: f64 = 1;
        for (self.deltas[0..self.len]) |delta| {
            delta_sum += delta * decay;
            decay *= delayed_td_lambda;
        }
        self.net.applyDelta(self.states[0], delta_sum, self.lr);
    }

    fn dropOldest(self: *DelayedTDLambda) void {
        for (1..self.len) |i| {
            self.states[i - 1] = self.states[i];
            self.deltas[i - 1] = self.deltas[i];
        }
        self.len -= 1;
    }
};

pub fn delayedTDLambdaExtraMemoryBytes() usize {
    return @sizeOf(DelayedTDLambda);
}

/// TC(λ) horizon and trace decay from the paper's informal parameter search; β uses the existing learning-rate schedule.
pub const tc_horizon: usize = 3;
pub const tc_lambda: f64 = 0.5;

/// Per-weight signed/absolute TD-error accumulators plus the current episode's h-step trace.
pub const TemporalCoherence = struct {
    net: *TupleNet,
    errors: []f32,
    abs_errors: []f32,
    trace: [tc_horizon + 1]Indices = undefined,
    trace_len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, net: *TupleNet) !TemporalCoherence {
        const errors = try gpa.alloc(f32, net.storage.len);
        errdefer gpa.free(errors);
        const abs_errors = try gpa.alloc(f32, net.storage.len);
        errdefer gpa.free(abs_errors);
        @memset(errors, 0);
        @memset(abs_errors, 0);
        return .{ .net = net, .errors = errors, .abs_errors = abs_errors };
    }

    pub fn deinit(self: *TemporalCoherence, gpa: std.mem.Allocator) void {
        gpa.free(self.errors);
        gpa.free(self.abs_errors);
    }

    pub fn extraMemoryBytes(self: *const TemporalCoherence) usize {
        return (self.errors.len + self.abs_errors.len) * @sizeOf(f32) + @sizeOf(TemporalCoherence);
    }

    /// Reset episode-local trace only; per-weight coherence statistics persist through training.
    pub fn beginEpisode(self: *TemporalCoherence) void {
        self.trace_len = 0;
    }

    /// Apply one TD error to the current afterstate and up to `tc_horizon` preceding afterstates (Algorithm 2).
    pub fn observe(self: *TemporalCoherence, idx: Indices, delta: f64, beta: f64) void {
        // Drop before appending so `trace_len` never exceeds `tc_horizon` when it indexes `trace`.
        if (self.trace_len > tc_horizon) self.dropOldest();
        self.trace[self.trace_len] = idx;
        self.trace_len += 1;

        var decay: f64 = 1;
        for (0..self.trace_len) |age| {
            const trace_idx = self.trace[self.trace_len - 1 - age];
            self.updateState(trace_idx, delta * decay, delta, beta);
            decay *= tc_lambda;
        }
    }

    fn updateState(self: *TemporalCoherence, idx: Indices, weighted_delta: f64, delta: f64, beta: f64) void {
        const tuple_count: f64 = @floatFromInt(self.net.count);
        for (0..self.net.count) |p| {
            const entry = offsets[p] + idx[p];
            const accumulated = self.errors[entry];
            const accumulated_abs = self.abs_errors[entry];
            const alpha = if (accumulated_abs != 0)
                @as(f64, @floatCast(@abs(accumulated))) / @as(f64, @floatCast(accumulated_abs))
            else
                1;
            const step: f32 = @floatCast(beta * alpha * weighted_delta / tuple_count);
            self.net.storage[entry] += step;
            self.errors[entry] += @floatCast(delta);
            self.abs_errors[entry] += @floatCast(@abs(delta));
        }
    }

    fn dropOldest(self: *TemporalCoherence) void {
        for (1..self.trace_len) |i| self.trace[i - 1] = self.trace[i];
        self.trace_len -= 1;
    }
};

/// int16 view of a table: one scale per pattern, so the 4-tuples can use a coarser step than the
/// 6-tuples. `q * scale` reproduces the f32 value to within half a step (129 MiB -> 64.6 MiB).
pub const Quantized = struct {
    storage: []i16,
    /// Value = `storage[i] * scales[p]` for pattern `p`; 0 for an all-zero table.
    scales: [pattern_count]f64 = @splat(0),
    count: usize,

    pub fn initWith(gpa: std.mem.Allocator, set: Set) !Quantized {
        return initCount(gpa, countOf(set));
    }

    fn initCount(gpa: std.mem.Allocator, n: usize) !Quantized {
        const storage = try gpa.alloc(i16, entriesOf(n));
        @memset(storage, 0);
        return .{ .storage = storage, .count = n };
    }

    pub fn deinit(self: *Quantized, gpa: std.mem.Allocator) void {
        gpa.free(self.storage);
    }

    pub fn value(self: *const Quantized, idx: Indices) f64 {
        var v: f64 = 0;
        for (0..self.count) |p| v += @as(f64, @floatFromInt(self.storage[offsets[p] + idx[p]])) * self.scales[p];
        return v;
    }

    pub fn eval(self: *const Quantized, board: Board) f64 {
        return self.value(indices(board));
    }
};

/// Largest absolute value of each table; the quantizer step is `max / 32767` per pattern.
fn tableScales(net: *const TupleNet) [pattern_count]f64 {
    var scales: [pattern_count]f64 = @splat(0);
    for (0..net.count) |p| {
        var max: f32 = 0;
        for (net.storage[offsets[p]..][0..tableLen(p)]) |w| max = @max(max, @abs(w));
        scales[p] = if (max == 0) 0 else @as(f64, max) / 32767;
    }
    return scales;
}

/// Round one table into `out` with `scale` (value / scale); a zero scale leaves zeros.
fn quantizeValues(values: []const f32, scale: f64, out: []i16) void {
    if (scale == 0) return @memset(out, 0);
    for (values, out) |w, *q| {
        const scaled = @as(f64, w) / scale;
        q.* = @intFromFloat(@round(std.math.clamp(scaled, -32767, 32767)));
    }
}

/// Quantize every active table into a fresh int16 buffer.
pub fn quantize(gpa: std.mem.Allocator, net: *const TupleNet) !Quantized {
    var q = try Quantized.initCount(gpa, net.count);
    errdefer q.deinit(gpa);
    q.scales = tableScales(net);
    for (0..net.count) |p| {
        quantizeValues(net.storage[offsets[p]..][0..tableLen(p)], q.scales[p], q.storage[offsets[p]..][0..tableLen(p)]);
    }
    return q;
}

/// Write the quantized values back into an f32 net, so the rounding can be A/B'd with the same
/// evaluation code that reads float tables.
pub fn dequantizeInto(q: *const Quantized, net: *TupleNet) void {
    std.debug.assert(q.count == net.count);
    for (0..q.count) |p| {
        const scale: f32 = @floatCast(q.scales[p]);
        for (q.storage[offsets[p]..][0..tableLen(p)], net.storage[offsets[p]..][0..tableLen(p)]) |v, *w| {
            w.* = @as(f32, @floatFromInt(v)) * scale;
        }
    }
}

pub const quantized_magic = "2048-NQ\x00";

/// File length for `n` quantized tables.
pub fn quantizedFileLen(n: usize) usize {
    return header_len + n * 4 + n * 8 + entriesOf(n) * 2;
}

/// File: magic, version, endianness marker, pattern count, per-table lengths (u32 LE), per-table
/// scales (f64 LE), then the i16 LE payload.
pub fn encodeQuantized(gpa: std.mem.Allocator, q: *const Quantized) ![]u8 {
    const bytes = try gpa.alloc(u8, quantizedFileLen(q.count));
    var w = blob.Writer.init(bytes);
    blob.writeHeader(&w, quantized_magic, file_version);
    w.writeU32(@intCast(q.count));
    for (0..q.count) |p| w.writeU32(@intCast(tableLen(p)));
    for (0..q.count) |p| w.writeF64(q.scales[p]);
    for (q.storage) |value| w.writeI16(value);
    return bytes;
}

pub fn parseQuantized(gpa: std.mem.Allocator, bytes: []const u8) !Quantized {
    if (bytes.len < header_len) return error.WrongTableFileSize;
    var r = blob.Reader.init(bytes);
    try blob.readHeader(&r, quantized_magic, file_version, .{
        .magic = error.BadTableMagic,
        .version = error.UnsupportedTableVersion,
        .endian = error.WrongTableEndianness,
    });
    const n: usize = try r.readU32();
    if (n == 0 or n > pattern_count) return error.WrongPatternCount;
    if (bytes.len != quantizedFileLen(n)) return error.WrongTableFileSize;
    for (0..n) |p| {
        if (try r.readU32() != tableLen(p)) return error.WrongTableLength;
    }
    var q = try Quantized.initCount(gpa, n);
    errdefer q.deinit(gpa);
    for (0..n) |p| q.scales[p] = try r.readF64();
    for (q.storage) |*value| value.* = try r.readI16();
    return q;
}

pub fn saveQuantized(q: *const Quantized, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try encodeQuantized(gpa, q);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn loadQuantized(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !Quantized {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(quantizedFileLen(pattern_count) + 1));
    defer gpa.free(bytes);
    return parseQuantized(gpa, bytes);
}

pub const file_magic = "2048-NT\x00";
pub const file_version: u32 = 1;
const header_len: usize = 8 + 4 + 4 + 4;
/// File length for `n` tables; `.wide` is `fileLen(wide_count)`, `.lshape` is `fileLen(pattern_count)`.
pub fn fileLen(n: usize) usize {
    return header_len + n * 4 + entriesOf(n) * 4;
}

/// File: magic, version, endianness marker, pattern count, per-table lengths (u32 LE), then f32 LE payload.
pub fn encode(gpa: std.mem.Allocator, net: *const TupleNet) ![]u8 {
    const bytes = try gpa.alloc(u8, fileLen(net.count));
    var w = blob.Writer.init(bytes);
    blob.writeHeader(&w, file_magic, file_version);
    w.writeU32(@intCast(net.count));
    for (0..net.count) |p| w.writeU32(@intCast(tableLen(p)));
    for (net.storage) |value| w.writeF32(value);
    return bytes;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !TupleNet {
    if (bytes.len < header_len) return error.WrongTableFileSize;
    var r = blob.Reader.init(bytes);
    try blob.readHeader(&r, file_magic, file_version, .{
        .magic = error.BadTableMagic,
        .version = error.UnsupportedTableVersion,
        .endian = error.WrongTableEndianness,
    });
    const n: usize = try r.readU32();
    if (n == 0 or n > pattern_count) return error.WrongPatternCount;
    if (bytes.len != fileLen(n)) return error.WrongTableFileSize;
    for (0..n) |p| {
        if (try r.readU32() != tableLen(p)) return error.WrongTableLength;
    }
    var net = try TupleNet.initCount(gpa, n);
    errdefer net.deinit(gpa);
    for (net.storage) |*value| value.* = try r.readF32();
    return net;
}

pub fn save(net: *const TupleNet, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try encode(gpa, net);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub fn load(net: *TupleNet, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(fileLen(pattern_count) + 1));
    defer gpa.free(bytes);
    // A table from the other `Set` has a different length; refuse it instead of overrunning.
    if (bytes.len != fileLen(net.count)) return error.TableSizeMismatch;
    var parsed = try parse(gpa, bytes);
    defer parsed.deinit(gpa);
    @memcpy(net.storage, parsed.storage);
}

pub const Mode = enum {
    /// 1-ply over afterstates: argmax of reward + V(afterstate).
    greedy,
    /// greedy plus one spawn of chance-node lookahead.
    chance,
    /// 2-ply: one more own move and spawn layer below `chance`.
    deep,
};

/// Where the TD target comes from: the afterstate the game actually reached, or the best move's
/// afterstate on the post-spawn board (`r + max_a V(after(s', a))`, the two-stage target).
pub const Target = enum { on_policy, max_backup };

/// Learning algorithm selected by the runner; TD(0) stays the default.
pub const Learner = enum { td_zero, delayed_td_lambda, temporal_coherence };

/// Runner variant: policy mode, table set, learner, read-out, TD target, and whether to load
/// `table_path` first.
pub const Variant = struct {
    mode: Mode = .greedy,
    set: Set = .full,
    learner: Learner = .td_zero,
    target: Target = .on_policy,
    /// `.summed` is the #322 opt-in; every shipped table and the default remain `.canonical`.
    symmetry: Symmetry = .canonical,
    resume_from_file: bool = false,
};

pub fn parseVariant(name: []const u8) ?Variant {
    if (std.mem.eql(u8, name, "base")) return .{};
    if (std.mem.eql(u8, name, "chance")) return .{ .mode = .chance };
    if (std.mem.eql(u8, name, "deep")) return .{ .mode = .deep };
    if (std.mem.eql(u8, name, "light")) return .{ .set = .light };
    if (std.mem.eql(u8, name, "lightdeep")) return .{ .mode = .deep, .set = .light };
    if (std.mem.eql(u8, name, "resume")) return .{ .resume_from_file = true };
    if (std.mem.eql(u8, name, "lightresume")) return .{ .resume_from_file = true, .set = .light };
    if (std.mem.eql(u8, name, "wide")) return .{ .set = .wide };
    if (std.mem.eql(u8, name, "summed")) return .{ .symmetry = .summed };
    if (std.mem.eql(u8, name, "widesum")) return .{ .set = .wide, .symmetry = .summed };
    if (std.mem.eql(u8, name, "widedelay")) return .{ .set = .wide, .learner = .delayed_td_lambda };
    if (std.mem.eql(u8, name, "widetc")) return .{ .set = .wide, .learner = .temporal_coherence };
    if (std.mem.eql(u8, name, "widedeep_tc")) return .{ .mode = .deep, .set = .wide, .learner = .temporal_coherence };
    if (std.mem.eql(u8, name, "maxb")) return .{ .set = .wide, .target = .max_backup };
    if (std.mem.eql(u8, name, "maxbchance")) return .{ .set = .wide, .mode = .chance, .target = .max_backup };
    if (std.mem.eql(u8, name, "maxbdeep")) return .{ .set = .wide, .mode = .deep, .target = .max_backup };
    if (std.mem.eql(u8, name, "widedeep")) return .{ .mode = .deep, .set = .wide };
    if (std.mem.eql(u8, name, "lshape")) return .{ .set = .lshape };
    if (std.mem.eql(u8, name, "lshaperesume")) return .{ .resume_from_file = true, .set = .lshape };
    return null;
}

pub const PlayResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
    /// `net.eval` leaf calls summed over the game (0 for modes that do not search).
    leaves: u64 = 0,
    /// (#336) The game hit its time cap before game over; it is not a completed evaluation row.
    timed_out: bool = false,
};
pub const GameResult = struct {
    score: u32,
    moves: u32,
    max_tile: core.Tile = 0,
    capped: bool = false,
    timed_out: bool = false,
};
/// Null preserves the library API's unlimited default; the experiment runner supplies a cap.
pub const GameOptions = struct { game_time_limit_ns: ?u64 = null };

fn timeLimitReached(start: std.Io.Timestamp, limit_ns: ?u64) bool {
    return game_limit.reached(start, limit_ns);
}

fn gameResult(state: core.GameState, moves: u32, timed_out: bool) GameResult {
    return .{ .score = state.score, .moves = moves, .max_tile = core.maxTile(state.board), .timed_out = timed_out, .capped = timed_out };
}

/// Afterstate chosen by a policy, with its scaled reward, cached table entries and value estimate.
const Pick = struct {
    dir: core.Direction,
    entries: Entries,
    reward: f64,
    value: f64,
};

/// Best legal move by reward + V(afterstate); null when no move changes the board.
fn pickMove(net: *const TupleNet, state: core.GameState) ?Pick {
    var best: ?Pick = null;
    for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        const entries = entriesFor(next.board);
        const pick: Pick = .{
            .dir = d,
            .entries = entries,
            .reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale,
            .value = net.valueOf(entries),
        };
        if (best == null or pick.reward + pick.value > best.?.reward + best.?.value) best = pick;
    }
    return best;
}

/// Value of a post-spawn board for the chance node: best reward + V over the legal moves (0 if none).
fn bestAfterValue(net: *const TupleNet, state: core.GameState) f64 {
    const pick = pickMove(net, state) orelse return 0;
    return pick.reward + pick.value;
}

/// Expectimax with one spawn of lookahead, returning the best move's own afterstate pick.
fn chanceMove(net: *const TupleNet, state: core.GameState) ?Pick {
    var best: ?Pick = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
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
                    total += spawn.p * bestAfterValue(net, child);
                }
            }
        }
        const chance = if (empties == 0) bestAfterValue(net, moved) else total / empties;
        const value = @as(f64, @floatFromInt(result.score_delta)) / reward_scale + chance;
        if (best == null or value > best_value) {
            const entries = entriesFor(moved.board);
            best = .{
                .dir = d,
                .entries = entries,
                .reward = @as(f64, @floatFromInt(result.score_delta)) / reward_scale,
                .value = net.valueOf(entries),
            };
            best_value = value;
        }
    }
    return best;
}

/// Search counters for one move: afterstate evaluations (`net.eval` calls).
pub const Stats = struct { leaves: u64 = 0 };

/// Depth-limited expectimax over the tables: our moves and every spawn (2 at 0.9, 4 at 0.1) exactly.
const Search = struct {
    net: *const TupleNet,
    stats: Stats = .{},

    /// Max over legal moves of reward + expected value after the spawn; 0 when no move changes the board.
    fn value(self: *Search, state: core.GameState, depth: u8) f64 {
        var best: f64 = -std.math.inf(f64);
        for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
            var next = state;
            const moved = next.slide(d);
            if (!moved.changed) continue;
            const v = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + self.chance(next, depth - 1);
            best = @max(best, v);
        }
        return if (best == -std.math.inf(f64)) 0 else best;
    }

    /// Expected value of a post-move board after spawning; depth 0 is the value leaf (the afterstate itself).
    fn chance(self: *Search, state: core.GameState, depth: u8) f64 {
        if (depth == 0) {
            self.stats.leaves += 1;
            return self.net.eval(state.board);
        }
        var total: f64 = 0;
        var empties: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] != 0) continue;
                empties += 1;
                for ([_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } }) |spawn| {
                    var child = state;
                    child.board[r][c] = spawn.value;
                    total += spawn.p * self.value(child, depth);
                }
            }
        }
        return if (empties == 0) self.value(state, depth) else total / empties;
    }
};

/// 2-ply: my move -> spawn -> my move -> spawn -> afterstate value + score_delta/reward_scale.
fn deepMove(net: *const TupleNet, state: core.GameState, stats: *Stats) ?Pick {
    var search: Search = .{ .net = net };
    defer stats.* = search.stats;
    var best: ?Pick = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        const v = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + search.chance(next, 2);
        if (best == null or v > best_value) {
            const entries = entriesFor(next.board);
            best = .{
                .dir = d,
                .entries = entries,
                .reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale,
                .value = net.valueOf(entries),
            };
            best_value = v;
        }
    }
    return best;
}

fn chooseMove(net: *const TupleNet, state: core.GameState, mode: Mode, stats: *Stats) ?Pick {
    return switch (mode) {
        .greedy => pickMove(net, state),
        .chance => chanceMove(net, state),
        .deep => deepMove(net, state, stats),
    };
}

pub fn bestMove(net: *const TupleNet, state: core.GameState, mode: Mode) ?core.Direction {
    var stats: Stats = .{};
    return bestMoveWithStats(net, state, mode, &stats);
}

/// `bestMove` plus search counters (`leaves` is 0 for modes that do not search).
pub fn bestMoveWithStats(net: *const TupleNet, state: core.GameState, mode: Mode, stats: *Stats) ?core.Direction {
    stats.* = .{};
    return if (chooseMove(net, state, mode, stats)) |p| p.dir else null;
}

/// 1-ply greedy move from a quantized table; this is what `algo=ntuple` plays.
pub fn bestMoveQuantized(q: *const Quantized, state: core.GameState) ?core.Direction {
    var best: ?core.Direction = null;
    var best_value: f64 = 0;
    for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        const value = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + q.eval(next.board);
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

/// Play one game to game over with the table policy.
pub fn play(net: *const TupleNet, rng: std.Random, mode: Mode) PlayResult {
    var stats: Stats = .{};
    return playWithStats(net, rng, mode, &stats);
}

pub fn playWithStats(net: *const TupleNet, rng: std.Random, mode: Mode, stats: *Stats) PlayResult {
    return playWithStatsOpts(net, rng, mode, stats, .{});
}

pub const gameStartTime = game_limit.start;

pub fn playWithStatsOpts(net: *const TupleNet, rng: std.Random, mode: Mode, stats: *Stats, opts: GameOptions) PlayResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    var state = core.GameState.gameStart(rng);
    var moves: u32 = 0;
    var move_stats: Stats = .{};
    while (true) {
        if (timeLimitReached(start, opts.game_time_limit_ns)) return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .leaves = stats.leaves, .timed_out = true };
        move_stats = .{};
        const pick = chooseMove(net, state, mode, &move_stats) orelse break;
        stats.leaves += move_stats.leaves;
        _ = state.move(pick.dir, rng);
        moves += 1;
    }
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .leaves = stats.leaves };
}

/// TD(0) on afterstates over one self-play game.
pub fn trainGame(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode) GameResult {
    return trainGameWith(net, rng, lr, mode, .on_policy);
}

/// `trainGame` with a selectable TD target. `.max_backup` replaces the value of the move the
/// game actually played with the max over the four moves available on the post-spawn board, so the
/// table learns the afterstate value of the *best* continuation rather than of the sampled one.
pub fn trainGameWith(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode, target: Target) GameResult {
    return trainGameWithOpts(net, rng, lr, mode, target, .{});
}

pub fn trainGameWithOpts(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode, target: Target, opts: GameOptions) GameResult {
    const start = gameStartTime(opts.game_time_limit_ns);
    var state = core.GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    var stats: Stats = .{};
    while (true) {
        if (timeLimitReached(start, opts.game_time_limit_ns)) return gameResult(state, moves, true);
        const pick = chooseMove(net, state, mode, &stats) orelse break;
        moves += 1;
        if (pending) |p| {
            const next_value = switch (target) {
                .on_policy => pick.value,
                .max_backup => bestAfterValue(net, state),
            };
            _ = net.trainEntries(p.entries, p.reward + next_value, lr);
        }
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| _ = net.trainEntries(p.entries, p.reward, lr);
    return gameResult(state, moves, false);
}

/// Opt-in delayed TD(lambda) self-play, with the on-policy afterstate target only.
pub fn trainGameDelayedTDLambda(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode) GameResult {
    return trainGameDelayedTDLambdaOpts(net, rng, lr, mode, .{});
}

pub fn trainGameDelayedTDLambdaOpts(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode, opts: GameOptions) GameResult {
    // #322's summed read-out is TD(0)-only; #290 keeps the canonical index path (do not mix).
    std.debug.assert(net.symmetry == .canonical);
    const start = gameStartTime(opts.game_time_limit_ns);
    var state = core.GameState.gameStart(rng);
    var learner = DelayedTDLambda.init(net, lr);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    var stats: Stats = .{};
    while (true) {
        if (timeLimitReached(start, opts.game_time_limit_ns)) return gameResult(state, moves, true);
        const pick = chooseMove(net, state, mode, &stats) orelse break;
        moves += 1;
        if (pending) |p| {
            // Re-evaluate the pending afterstate: older delayed updates may have touched its weights.
            const delta = p.reward + pick.value - net.value(p.entries.canonical);
            learner.observe(p.entries.canonical, delta);
        }
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| learner.observe(p.entries.canonical, p.reward - net.value(p.entries.canonical));
    learner.finish();
    return gameResult(state, moves, false);
}

/// Opt-in on-policy TD(λ) with per-weight Temporal Coherence; the TD(0) path is unchanged.
pub fn trainGameTemporalCoherence(tc: *TemporalCoherence, rng: std.Random, beta: f64, mode: Mode) GameResult {
    return trainGameTemporalCoherenceOpts(tc, rng, beta, mode, .{});
}

pub fn trainGameTemporalCoherenceOpts(tc: *TemporalCoherence, rng: std.Random, beta: f64, mode: Mode, opts: GameOptions) GameResult {
    const net = tc.net;
    // #322's summed read-out is TD(0)-only; #288 keeps the canonical index path (do not mix).
    std.debug.assert(net.symmetry == .canonical);
    const start = gameStartTime(opts.game_time_limit_ns);
    tc.beginEpisode();
    var state = core.GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    var stats: Stats = .{};
    while (true) {
        if (timeLimitReached(start, opts.game_time_limit_ns)) return gameResult(state, moves, true);
        const pick = chooseMove(net, state, mode, &stats) orelse break;
        moves += 1;
        if (pending) |p| {
            // Re-evaluate the pending afterstate because its weights may have changed through the trace.
            const delta = p.reward + pick.value - net.value(p.entries.canonical);
            tc.observe(p.entries.canonical, delta, beta);
        }
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| tc.observe(p.entries.canonical, p.reward - net.value(p.entries.canonical), beta);
    return gameResult(state, moves, false);
}

// ---------------------------------------------------------------------------

fn testNet() !TupleNet {
    return TupleNet.init(std.testing.allocator);
}

const sample_board: Board = .{
    .{ 2, 4, 0, 8 },
    .{ 0, 16, 32, 0 },
    .{ 64, 0, 0, 128 },
    .{ 0, 256, 0, 2 },
};

fn directIndex(board: Board, p: usize, t: usize) u32 {
    var idx: u32 = 0;
    for (0..patterns[p].len) |j| {
        const cell = transformCell(t, patterns[p].cells[j]);
        idx |= @as(u32, @intCast(levelOf(board[cell / core.BoardSize][cell % core.BoardSize]))) << @intCast(4 * j);
    }
    return idx;
}

fn canonicalIndex(board: Board, p: usize) u32 {
    var best: u32 = std.math.maxInt(u32);
    for (0..8) |t| best = @min(best, directIndex(board, p, t));
    return best;
}

/// True when the unordered cell set `cells` is one of the 8 dihedral transforms of `base`,
/// i.e. both patterns would index the same table.
fn sameOrbit(base: []const u8, cells: []const u8) bool {
    if (base.len != cells.len) return false;
    var want: [6]u8 = undefined;
    @memcpy(want[0..cells.len], cells);
    std.mem.sort(u8, want[0..cells.len], {}, std.sort.asc(u8));
    for (0..8) |t| {
        var transformed: [6]u8 = undefined;
        for (base, 0..) |cell, j| transformed[j] = @intCast(transformCell(t, cell));
        std.mem.sort(u8, transformed[0..base.len], {}, std.sort.asc(u8));
        if (std.mem.eql(u8, transformed[0..base.len], want[0..cells.len])) return true;
    }
    return false;
}

/// A distinct weight per table, so a round-trip can tell the tables apart.
fn tableProbe(p: usize) f32 {
    return @floatCast(@as(f64, @floatFromInt(p)) + 0.75);
}

test "levelOf maps tiles to log2 levels and clamps large tiles" {
    try std.testing.expectEqual(@as(usize, 0), levelOf(0));
    try std.testing.expectEqual(@as(usize, 1), levelOf(2));
    try std.testing.expectEqual(@as(usize, 11), levelOf(2048));
    try std.testing.expectEqual(@as(usize, levels - 1), levelOf(65536));
}

test "pattern index is invariant under the 8 symmetry transforms" {
    const base = indices(sample_board);
    for (0..8) |t| {
        const rotated = indices(transformBoard(sample_board, t));
        try std.testing.expectEqualSlices(u32, &base, &rotated);
    }
}

test "fast slot indices match the direct canonical index" {
    const fast = indices(sample_board);
    for (0..pattern_count) |p| {
        try std.testing.expectEqual(canonicalIndex(sample_board, p), fast[p]);
    }
}

test "one TD step changes only the touched table entries" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    const idx = indices(sample_board);
    for (0..net.count) |p| net.storage[offsets[p] + idx[p]] = 0.5;

    const before = try std.testing.allocator.dupe(f32, net.storage);
    defer std.testing.allocator.free(before);
    _ = net.train(idx, 1.0, 0.1);

    var changed: usize = 0;
    for (net.storage, before) |now, old| {
        if (now != old) changed += 1;
    }
    try std.testing.expectEqual(net.count, changed);
}

test "save and load round-trip exactly" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    const idx = indices(sample_board);
    for (0..net.count) |p| net.storage[offsets[p] + idx[p]] = @floatCast(@as(f64, @floatFromInt(p)) + 0.25);

    const bytes = try encode(std.testing.allocator, &net);
    defer std.testing.allocator.free(bytes);
    var back = try parse(std.testing.allocator, bytes);
    defer back.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(net.storage), std.mem.sliceAsBytes(back.storage));
}

test "empty and dead boards do not panic" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(f64, 0), net.eval(std.mem.zeroes(Board)));
    const dead: Board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    };
    _ = net.eval(dead);
    try std.testing.expectEqual(@as(?core.Direction, null), bestMove(&net, .{ .board = dead }, .greedy));
}

test "training is deterministic for a fixed seed" {
    var a = try testNet();
    defer a.deinit(std.testing.allocator);
    var b = try testNet();
    defer b.deinit(std.testing.allocator);
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&a, rng_a.random(), 0.01, .greedy);
    _ = trainGame(&b, rng_b.random(), 0.01, .greedy);
    try std.testing.expectEqualSlices(f32, a.storage, b.storage);
}

test "deep returns only legal moves and null on a dead board" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    const dead: Board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    };
    try std.testing.expectEqual(@as(?core.Direction, null), bestMove(&net, .{ .board = dead }, .deep));
    var state: core.GameState = .{ .board = sample_board };
    try std.testing.expect(state.slide(bestMove(&net, state, .deep).?).changed);
}

test "deep is deterministic and counts leaf evaluations" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    const state: core.GameState = .{ .board = sample_board };
    var a: Stats = .{};
    var b: Stats = .{};
    const first = bestMoveWithStats(&net, state, .deep, &a);
    const second = bestMoveWithStats(&net, state, .deep, &b);
    try std.testing.expect(first != null);
    try std.testing.expectEqual(first, second);
    try std.testing.expect(a.leaves > 0);
    try std.testing.expectEqual(a.leaves, b.leaves);
}

test "the wide set adds the second 2x3 orbit and keeps light/full unchanged (#106)" {
    try std.testing.expectEqual(@as(usize, 5), light_count);
    try std.testing.expectEqual(@as(usize, 6), full_count);
    try std.testing.expectEqual(@as(usize, 7), wide_count);
    try std.testing.expectEqual(@as(usize, 5 * 65536), entryCount(.light));
    try std.testing.expectEqual(@as(usize, 5 * 65536 + 16777216), entryCount(.full));
    try std.testing.expectEqual(entryCount(.full) + 16777216, entryCount(.wide));

    var wide = try TupleNet.initWith(std.testing.allocator, .wide);
    defer wide.deinit(std.testing.allocator);
    try std.testing.expectEqual(wide_count, wide.count);
    try std.testing.expectEqual(entryCount(.wide), wide.storage.len);
}

test "the second 6-tuple is not a transform of the first (#106)" {
    var same_orbit = false;
    for (0..8) |t| {
        var cells: [6]u8 = undefined;
        for (patterns[5].cells, 0..) |cell, j| cells[j] = @intCast(transformCell(t, cell));
        std.mem.sort(u8, &cells, {}, std.sort.asc(u8));
        var want = patterns[6].cells[0..6].*;
        std.mem.sort(u8, &want, {}, std.sort.asc(u8));
        if (std.mem.eql(u8, &cells, &want)) same_orbit = true;
    }
    try std.testing.expect(!same_orbit);
    // ... and it is a 2x3 rectangle as well.
    try std.testing.expectEqual(@as(u8, 6), patterns[6].len);
}

test "the L-shaped 6-tuple is one D4 orbit that no existing pattern shares (#323)" {
    const lshape = patterns[pattern_count - 1];
    try std.testing.expectEqual(@as(u8, 6), lshape.len);
    // The table is built from these cells: top row cols 0-2 plus left column rows 0-3.
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 2, 4, 8, 12 }, lshape.cells[0..6]);

    // Arms 3 and 4 are unequal, so no transform fixes the set: the orbit is exactly 8 placements
    // (4 corners x 2 arm orientations) and needs one table.
    var orbit: [8][6]u8 = undefined;
    var orbit_len: usize = 0;
    for (0..8) |t| {
        var cells: [6]u8 = undefined;
        for (lshape.cells[0..6], 0..) |cell, j| cells[j] = @intCast(transformCell(t, cell));
        std.mem.sort(u8, &cells, {}, std.sort.asc(u8));
        var seen = false;
        for (orbit[0..orbit_len]) |placed| seen = seen or std.mem.eql(u8, &placed, &cells);
        if (!seen) {
            orbit[orbit_len] = cells;
            orbit_len += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 8), orbit_len);

    // Distinct from every existing pattern orbit, in particular both 2x3 rectangles and every
    // 4-tuple: the addition cannot reuse a table the `min` index already covers.
    for (patterns[0 .. pattern_count - 1]) |pat| {
        try std.testing.expect(!sameOrbit(pat.cells[0..pat.len], lshape.cells[0..6]));
    }
    try std.testing.expect(sameOrbit(lshape.cells[0..6], &[_]u8{ 3, 7, 11, 13, 14, 15 }));
}

test "the lshape set adds exactly one 6-tuple table over wide (#323)" {
    try std.testing.expectEqual(@as(usize, wide_count + 1), pattern_count);
    try std.testing.expectEqual(total_entries, entryCount(.lshape));
    try std.testing.expectEqual(entryCount(.wide) + 16777216, entryCount(.lshape));
    // `.wide` keeps the #106 layout, so tables written for it stay loadable.
    try std.testing.expectEqual(@as(usize, 33_882_112), entryCount(.wide));
    try std.testing.expectEqual(@as(usize, 135_528_496), fileLen(wide_count));

    var lshape = try TupleNet.initWith(std.testing.allocator, .lshape);
    defer lshape.deinit(std.testing.allocator);
    try std.testing.expectEqual(pattern_count, lshape.count);
    try std.testing.expectEqual(entryCount(.lshape), lshape.storage.len);
    // The new table is appended after the seven existing ones.
    try std.testing.expectEqual(entryCount(.wide), offsets[wide_count]);
    try std.testing.expectEqual(tableLen(wide_count), lshape.table(wide_count).len);
}

test "the L-shape table stays inside the 200 MiB f32 ceiling (#323)" {
    const mib = 1024 * 1024;
    // 16^6 f32 entries: 64 MiB for the one new table, and nothing else grows.
    try std.testing.expectEqual(@as(usize, 64 * mib), tableLen(pattern_count - 1) * @sizeOf(f32));
    try std.testing.expectEqual(@as(usize, 32 * mib), tableLen(pattern_count - 1) * @sizeOf(i16));
    try std.testing.expectEqual(@as(usize, 64 * mib), (entryCount(.lshape) - entryCount(.wide)) * @sizeOf(f32));

    // The whole opt-in set: 193.25 MiB f32 (129.25 MiB for `.wide`), under the prototype ceiling.
    try std.testing.expectEqual(@as(usize, 202_637_312), entryCount(.lshape) * @sizeOf(f32));
    try std.testing.expect(entryCount(.lshape) * @sizeOf(f32) <= 200 * mib);
    try std.testing.expectEqual(@as(usize, 202_637_364), fileLen(pattern_count));
    // The int16 form a checkpoint or eval table would ship: ~96.6 MiB.
    try std.testing.expectEqual(@as(usize, 101_318_772), quantizedFileLen(pattern_count));
    try std.testing.expect(quantizedFileLen(pattern_count) <= 100 * mib);
}

test "light tables keep the five base tuples and drop the 6-tuple" {
    try std.testing.expectEqual(@as(usize, 5), light_count);
    try std.testing.expectEqual(@as(usize, 5 * 65536), entryCount(.light));
    try std.testing.expectEqual(entryCount(.lshape), total_entries);
    try std.testing.expect(fileLen(light_count) * 17 <= fileLen(pattern_count) * 1);

    var light = try TupleNet.initWith(std.testing.allocator, .light);
    defer light.deinit(std.testing.allocator);
    try std.testing.expectEqual(light_count, light.count);
    try std.testing.expectEqual(entryCount(.light), light.storage.len);

    const idx = indices(sample_board);
    var expected: f64 = 0;
    for (0..light_count) |p| {
        light.storage[offsets[p] + idx[p]] = 0.5;
        expected += 0.5;
    }
    try std.testing.expectEqual(expected, light.value(idx));
}

test "light table file round-trips with its own pattern count" {
    var light = try TupleNet.initWith(std.testing.allocator, .light);
    defer light.deinit(std.testing.allocator);
    const idx = indices(sample_board);
    for (0..light_count) |p| light.storage[offsets[p] + idx[p]] = @floatCast(@as(f64, @floatFromInt(p)) + 0.5);

    const bytes = try encode(std.testing.allocator, &light);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(fileLen(light_count), bytes.len);
    var back = try parse(std.testing.allocator, bytes);
    defer back.deinit(std.testing.allocator);
    try std.testing.expectEqual(light_count, back.count);
    try std.testing.expectEqualSlices(f32, light.storage, back.storage);
}

test "the lshape table round-trips through the f32 blob format (#323)" {
    const gpa = std.testing.allocator;
    const idx = indices(sample_board);
    var net = try TupleNet.initWith(gpa, .lshape);
    const count = net.count;
    for (0..count) |p| net.storage[offsets[p] + idx[p]] = tableProbe(p);
    const bytes = encode(gpa, &net) catch |err| {
        net.deinit(gpa);
        return err;
    };
    // Free the 193.25 MiB source before the parser allocates its own copy.
    net.deinit(gpa);
    defer gpa.free(bytes);

    // Same magic and version as a `.wide` file; only the pattern count and payload grow.
    try std.testing.expectEqual(fileLen(pattern_count), bytes.len);
    try std.testing.expectEqual(file_version, std.mem.readInt(u32, bytes[8..12], .little));
    try std.testing.expectEqual(@as(u32, @intCast(count)), std.mem.readInt(u32, bytes[16..20], .little));

    var back = try parse(gpa, bytes);
    defer back.deinit(gpa);
    try std.testing.expectEqual(count, back.count);
    try std.testing.expectEqualSlices(u8, bytes[header_len + count * 4 ..], std.mem.sliceAsBytes(back.storage));
    for (0..count) |p| try std.testing.expectEqual(tableProbe(p), back.storage[offsets[p] + idx[p]]);
    // Every table, the new one included, contributes to the value: sum(0.75 + p) for p in 0..8.
    try std.testing.expectEqual(@as(f64, 34), back.eval(sample_board));
}

test "the lshape quantized table round-trips inside the int16 ceiling (#323)" {
    const gpa = std.testing.allocator;
    var q = try Quantized.initWith(gpa, .lshape);
    const count = q.count;
    for (0..count) |p| q.scales[p] = tableProbe(p);
    const scales = q.scales;
    q.storage[offsets[pattern_count - 1] + 3] = 1234;

    const bytes = try encodeQuantized(gpa, &q);
    q.deinit(gpa);
    defer gpa.free(bytes);

    try std.testing.expectEqual(quantizedFileLen(pattern_count), bytes.len);
    try std.testing.expect(bytes.len <= 100 * 1024 * 1024);
    var back = try parseQuantized(gpa, bytes);
    defer back.deinit(gpa);
    try std.testing.expectEqual(count, back.count);
    try std.testing.expectEqualSlices(f64, scales[0..count], back.scales[0..count]);
    try std.testing.expectEqual(@as(i16, 1234), back.storage[offsets[pattern_count - 1] + 3]);
}

test "a 7-table wide file written before the L-shape still loads (#323)" {
    const gpa = std.testing.allocator;
    const idx = indices(sample_board);
    var wide = try TupleNet.initWith(gpa, .wide);
    const count = wide.count;
    for (0..count) |p| wide.storage[offsets[p] + idx[p]] = tableProbe(p);
    const bytes = encode(gpa, &wide) catch |err| {
        wide.deinit(gpa);
        return err;
    };
    wide.deinit(gpa);
    defer gpa.free(bytes);

    // The pre-#323 header is unchanged: version 1, count 7, the same first table length.
    try std.testing.expectEqual(fileLen(wide_count), bytes.len);
    try std.testing.expectEqual(file_version, std.mem.readInt(u32, bytes[8..12], .little));
    try std.testing.expectEqual(@as(u32, wide_count), std.mem.readInt(u32, bytes[16..20], .little));
    try std.testing.expectEqual(@as(u32, @intCast(tableLen(0))), std.mem.readInt(u32, bytes[20..24], .little));

    var back = try parse(gpa, bytes);
    defer back.deinit(gpa);
    try std.testing.expectEqual(@as(usize, wide_count), back.count);
    // The seven tables of the old file are read back and the L table is not part of the sum:
    // sum(0.75 + p) for p in 0..7.
    try std.testing.expectEqual(@as(f64, 26.25), back.eval(sample_board));
    // The L shape is opt-in: its file is longer, so `load` refuses to mix the two sets.
    try std.testing.expect(fileLen(wide_count) < fileLen(pattern_count));
}

test "encode writes the f32 table layout byte for byte (#302)" {
    const gpa = std.testing.allocator;
    var net = try TupleNet.initCount(gpa, 1);
    defer net.deinit(gpa);
    for (net.storage, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast(i % 11)) - 5);

    const bytes = try encode(gpa, &net);
    defer gpa.free(bytes);
    // magic, version, endianness marker, pattern count, then the length of the one 4-tuple table
    try std.testing.expectEqualSlices(u8, &[_]u8{
        '2', '0', '4', '8', '-', 'N', 'T', 0,
        1,   0,   0,   0,   4,   3,   2,   1,
        1,   0,   0,   0,   0,   0,   1,   0,
    }, bytes[0 .. header_len + 4]);
    // the payload starts with -5.0, then -4.0, as f32 little endian
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x00, 0x00, 0xa0, 0xc0, 0x00, 0x00, 0x80, 0xc0 }, bytes[header_len + 4 ..][0..8]);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings("378f24e3dca98db88df93c86e20b4fb6727b0990140658edccc15eee93eb7819", &hex);
}

test "encodeQuantized writes the int16 table layout byte for byte (#302)" {
    const gpa = std.testing.allocator;
    var q = try Quantized.initCount(gpa, 1);
    defer q.deinit(gpa);
    q.scales[0] = 1.0;
    for (q.storage, 0..) |*value, i| value.* = @intCast(@as(i32, @intCast(i % 3001)) - 1500);

    const bytes = try encodeQuantized(gpa, &q);
    defer gpa.free(bytes);
    // ... and the 8-byte scale (1.0) follows the table length, which starts right after the header
    try std.testing.expectEqualSlices(u8, &[_]u8{
        '2', '0', '4', '8', '-', 'N', 'Q',  0,
        1,   0,   0,   0,   4,   3,   2,    1,
        1,   0,   0,   0,   0,   0,   1,    0,
        0,   0,   0,   0,   0,   0,   0xf0, 0x3f,
    }, bytes[0 .. header_len + 12]);
    // the payload starts with -1500, then -1499, as i16 little endian
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x24, 0xfa, 0x25, 0xfa, 0x26, 0xfa, 0x27, 0xfa }, bytes[header_len + 12 ..][0..8]);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings("bdc28ee4d1918a0ccdc3a231c148208b76ccdd2734c1c7e2b06c9b90ded0f936", &hex);
}

test "parse rejects a foreign magic, another version, a wrong length and a wrong size (#302)" {
    const gpa = std.testing.allocator;
    var net = try TupleNet.initCount(gpa, 1);
    defer net.deinit(gpa);
    const full = try encode(gpa, &net);
    defer gpa.free(full);

    try std.testing.expectError(error.WrongTableFileSize, parse(gpa, full[0 .. header_len - 1]));
    try std.testing.expectError(error.WrongTableFileSize, parse(gpa, full[0 .. full.len - 1]));
    const oversized = try gpa.alloc(u8, full.len + 1);
    defer gpa.free(oversized);
    @memcpy(oversized[0..full.len], full);
    oversized[full.len] = 0;
    try std.testing.expectError(error.WrongTableFileSize, parse(gpa, oversized));

    const bytes = try gpa.dupe(u8, full);
    defer gpa.free(bytes);
    bytes[0] ^= 0xff;
    try std.testing.expectError(error.BadTableMagic, parse(gpa, bytes));
    bytes[0] ^= 0xff;
    std.mem.writeInt(u32, bytes[8..12], file_version + 1, .little);
    try std.testing.expectError(error.UnsupportedTableVersion, parse(gpa, bytes));
    std.mem.writeInt(u32, bytes[8..12], file_version, .little);
    std.mem.writeInt(u32, bytes[12..16], 0x04030201, .little);
    try std.testing.expectError(error.WrongTableEndianness, parse(gpa, bytes));
    std.mem.writeInt(u32, bytes[12..16], blob.endian_marker, .little);
    std.mem.writeInt(u32, bytes[16..20], 0, .little);
    try std.testing.expectError(error.WrongPatternCount, parse(gpa, bytes));
    std.mem.writeInt(u32, bytes[16..20], @intCast(pattern_count + 1), .little);
    try std.testing.expectError(error.WrongPatternCount, parse(gpa, bytes));
    std.mem.writeInt(u32, bytes[16..20], 1, .little);
    std.mem.writeInt(u32, bytes[20..24], @intCast(tableLen(0) + 1), .little);
    try std.testing.expectError(error.WrongTableLength, parse(gpa, bytes));
}

test "parseQuantized rejects the same broken headers as parse (#302)" {
    const gpa = std.testing.allocator;
    var q = try Quantized.initCount(gpa, 1);
    defer q.deinit(gpa);
    q.scales[0] = 1.0;
    const full = try encodeQuantized(gpa, &q);
    defer gpa.free(full);

    try std.testing.expectError(error.WrongTableFileSize, parseQuantized(gpa, full[0 .. header_len - 1]));
    try std.testing.expectError(error.WrongTableFileSize, parseQuantized(gpa, full[0 .. full.len - 1]));

    const bytes = try gpa.dupe(u8, full);
    defer gpa.free(bytes);
    bytes[0] ^= 0xff;
    try std.testing.expectError(error.BadTableMagic, parseQuantized(gpa, bytes));
    bytes[0] ^= 0xff;
    std.mem.writeInt(u32, bytes[8..12], file_version + 1, .little);
    try std.testing.expectError(error.UnsupportedTableVersion, parseQuantized(gpa, bytes));
    std.mem.writeInt(u32, bytes[8..12], file_version, .little);
    std.mem.writeInt(u32, bytes[12..16], 0x04030201, .little);
    try std.testing.expectError(error.WrongTableEndianness, parseQuantized(gpa, bytes));
    std.mem.writeInt(u32, bytes[12..16], blob.endian_marker, .little);
    std.mem.writeInt(u32, bytes[16..20], 0, .little);
    try std.testing.expectError(error.WrongPatternCount, parseQuantized(gpa, bytes));
    std.mem.writeInt(u32, bytes[16..20], 1, .little);
    std.mem.writeInt(u32, bytes[20..24], @intCast(tableLen(0) + 1), .little);
    try std.testing.expectError(error.WrongTableLength, parseQuantized(gpa, bytes));
}

test "resume keeps learned values instead of reinitialising the table" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    var rng = std.Random.DefaultPrng.init(4);
    _ = trainGame(&net, rng.random(), 0.05, .greedy);

    var trained = false;
    for (net.storage) |w| {
        if (w != 0) trained = true;
    }
    try std.testing.expect(trained);

    const bytes = try encode(std.testing.allocator, &net);
    defer std.testing.allocator.free(bytes);

    var fresh = try testNet();
    defer fresh.deinit(std.testing.allocator);
    var resumed = try testNet();
    defer resumed.deinit(std.testing.allocator);
    var parsed = try parse(std.testing.allocator, bytes);
    defer parsed.deinit(std.testing.allocator);
    @memcpy(resumed.storage, parsed.storage);
    try std.testing.expectEqualSlices(f32, net.storage, resumed.storage);
    try std.testing.expect(!std.mem.eql(f32, fresh.storage, resumed.storage));

    const before = try std.testing.allocator.dupe(f32, resumed.storage);
    defer std.testing.allocator.free(before);
    const idx = indices(sample_board);
    _ = resumed.train(idx, 1.0, 0.05);
    try std.testing.expect(!std.mem.eql(f32, before, resumed.storage));
}

test "variant names select the mode, the table set and resume" {
    try std.testing.expectEqual(Variant{}, parseVariant("base").?);
    try std.testing.expectEqual(Mode.chance, parseVariant("chance").?.mode);
    try std.testing.expectEqual(Mode.deep, parseVariant("deep").?.mode);
    try std.testing.expectEqual(Set.light, parseVariant("light").?.set);
    const lightdeep = parseVariant("lightdeep").?;
    try std.testing.expectEqual(Mode.deep, lightdeep.mode);
    try std.testing.expectEqual(Set.light, lightdeep.set);
    try std.testing.expect(parseVariant("resume").?.resume_from_file);
    const lightresume = parseVariant("lightresume").?;
    try std.testing.expect(lightresume.resume_from_file);
    try std.testing.expectEqual(Set.light, lightresume.set);
    try std.testing.expectEqual(Set.wide, parseVariant("wide").?.set);
    const widedelay = parseVariant("widedelay").?;
    try std.testing.expectEqual(Set.wide, widedelay.set);
    try std.testing.expectEqual(Learner.delayed_td_lambda, widedelay.learner);
    try std.testing.expectEqual(Target.on_policy, widedelay.target);
    const widetc = parseVariant("widetc").?;
    try std.testing.expectEqual(Learner.temporal_coherence, widetc.learner);
    const widedeep_tc = parseVariant("widedeep_tc").?;
    try std.testing.expectEqual(Mode.deep, widedeep_tc.mode);
    try std.testing.expectEqual(Learner.temporal_coherence, widedeep_tc.learner);
    try std.testing.expectEqual(Mode.deep, parseVariant("widedeep").?.mode);
    try std.testing.expectEqual(Set.wide, parseVariant("widedeep").?.set);
    const lshape = parseVariant("lshape").?;
    try std.testing.expectEqual(Set.lshape, lshape.set);
    try std.testing.expectEqual(Mode.greedy, lshape.mode);
    try std.testing.expectEqual(Learner.td_zero, lshape.learner);
    try std.testing.expect(!lshape.resume_from_file);
    const lshaperesume = parseVariant("lshaperesume").?;
    try std.testing.expect(lshaperesume.resume_from_file);
    try std.testing.expectEqual(Set.lshape, lshaperesume.set);
    try std.testing.expectEqual(@as(?Variant, null), parseVariant("lightresumex"));
}

test "quantizeValues rounds to the nearest step and clamps (#129)" {
    const values = [_]f32{ 0, 0.5, -0.5, 1.0, 3.0, -3.0 };
    var q: [6]i16 = undefined;
    quantizeValues(&values, 1.0, &q);
    try std.testing.expectEqualSlices(i16, &[_]i16{ 0, 1, -1, 1, 3, -3 }, &q);

    // a zero scale (all-zero table) stays zero instead of dividing by zero
    var zeros: [3]i16 = @splat(7);
    quantizeValues(&[_]f32{ 1, 2, 3 }, 0, &zeros);
    try std.testing.expectEqualSlices(i16, &[_]i16{ 0, 0, 0 }, &zeros);

    // out of range saturates instead of wrapping
    var big: [2]i16 = undefined;
    quantizeValues(&[_]f32{ 1e9, -1e9 }, 1.0, &big);
    try std.testing.expectEqualSlices(i16, &[_]i16{ 32767, -32767 }, &big);
}

test "quantized table round-trips through the file format (#129)" {
    const gpa = std.testing.allocator;
    var q = try Quantized.initWith(gpa, .light);
    defer q.deinit(gpa);
    for (0..q.count) |p| q.scales[p] = @floatFromInt(p + 1);
    q.storage[0] = 1234;
    q.storage[offsets[1] + 7] = -4321;

    const bytes = try encodeQuantized(gpa, &q);
    defer gpa.free(bytes);
    try std.testing.expectEqual(quantizedFileLen(q.count), bytes.len);

    var back = try parseQuantized(gpa, bytes);
    defer back.deinit(gpa);
    try std.testing.expectEqual(q.count, back.count);
    try std.testing.expectEqualSlices(f64, q.scales[0..q.count], back.scales[0..back.count]);
    try std.testing.expectEqual(q.storage[0], back.storage[0]);
    try std.testing.expectEqual(q.storage[offsets[1] + 7], back.storage[offsets[1] + 7]);
}

test "parseQuantized rejects short buffers and the float format (#129)" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.WrongTableFileSize, parseQuantized(gpa, &[_]u8{ 1, 2, 3 }));
    var f32_header: [header_len]u8 = @splat(0);
    @memcpy(f32_header[0..8], file_magic);
    try std.testing.expectError(error.BadTableMagic, parseQuantized(gpa, &f32_header));
}

test "quantized eval reproduces the float table it came from (#129)" {
    const gpa = std.testing.allocator;
    var net = try TupleNet.initCount(gpa, 1);
    defer net.deinit(gpa);
    for (net.storage, 0..) |*w, i| w.* = @as(f32, @floatFromInt(i % 977)) - 300;

    var q = try quantize(gpa, &net);
    defer q.deinit(gpa);

    var idx: Indices = @splat(0);
    idx[0] = indices(sample_board)[0];
    // one table only: the value is that table's entry, so the tolerance is a single quantization step
    try std.testing.expect(@abs(net.value(idx) - q.value(idx)) < 0.02);
    try std.testing.expect(@abs(net.eval(sample_board) - q.eval(sample_board)) < 0.02);
}

test "delayed TD(lambda) matches the naive bounded trace including the episode tail (#290)" {
    const gpa = std.testing.allocator;
    var reference = try TupleNet.initWith(gpa, .light);
    defer reference.deinit(gpa);
    var delayed = try TupleNet.initWith(gpa, .light);
    defer delayed.deinit(gpa);

    var second = sample_board;
    second[0][0] = 0;
    second[3][3] = 4;
    var third = sample_board;
    third[1][1] = 0;
    third[2][2] = 16;
    const states = [_]Indices{
        indices(sample_board),
        indices(second),
        indices(third),
        indices(sample_board),
        indices(second),
        indices(third),
    };
    const deltas = [_]f64{ 1.25, -0.5, 2.0, -3.0, 0.25, 4.0 };
    const lr = 0.05;

    trainTdLambdaReference(&reference, &states, &deltas, lr, delayed_td_lambda, delayed_td_horizon);
    var learner = DelayedTDLambda.init(&delayed, lr);
    for (states, deltas) |idx, delta| learner.observe(idx, delta);
    learner.finish();

    for (reference.storage, delayed.storage) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.00001);
    }
}

/// Straightforward per-error implementation used only as a test oracle for the delayed update.
fn trainTdLambdaReference(net: *TupleNet, states: []const Indices, deltas: []const f64, lr: f64, lambda: f64, horizon: usize) void {
    std.debug.assert(states.len == deltas.len);
    for (deltas, 0..) |delta, t| {
        const first = if (t > horizon) t - horizon else 0;
        var k = t + 1;
        while (k > first) {
            k -= 1;
            var decay: f64 = 1;
            for (0..t - k) |_| decay *= lambda;
            net.applyDelta(states[k], delta * decay, lr);
        }
    }
}

test "temporal coherence scales updates by accumulated signed versus absolute error (#288)" {
    const gpa = std.testing.allocator;
    var net = try TupleNet.initCount(gpa, 1);
    defer net.deinit(gpa);
    var tc = try TemporalCoherence.init(gpa, &net);
    defer tc.deinit(gpa);

    var first: Indices = @splat(0);
    first[0] = 1;
    var second: Indices = @splat(0);
    second[0] = 2;

    tc.beginEpisode();
    tc.observe(first, 2.0, 1.0);
    tc.observe(second, -1.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), net.storage[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tc.errors[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), tc.abs_errors[1], 0.00001);

    tc.beginEpisode();
    tc.observe(first, 3.0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), net.storage[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), tc.errors[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), tc.abs_errors[1], 0.00001);
}

test "temporal coherence trace stops propagating beyond the configured horizon (#288)" {
    const gpa = std.testing.allocator;
    var net = try TupleNet.initCount(gpa, 1);
    defer net.deinit(gpa);
    var tc = try TemporalCoherence.init(gpa, &net);
    defer tc.deinit(gpa);

    tc.beginEpisode();
    for (1..6) |entry| {
        var idx: Indices = @splat(0);
        idx[0] = @intCast(entry);
        tc.observe(idx, 1.0, 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.875), net.storage[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.875), net.storage[2], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), net.storage[5], 0.00001);
}

test "time limits reject unfinished n-tuple games before the next move (#336)" {
    var rng = std.Random.DefaultPrng.init(336);
    var net = try TupleNet.initWith(std.testing.allocator, .light);
    defer net.deinit(std.testing.allocator);
    const before = try std.testing.allocator.dupe(f32, net.storage);
    defer std.testing.allocator.free(before);
    const options: GameOptions = .{ .game_time_limit_ns = 0 };
    const trained = trainGameWithOpts(&net, rng.random(), 0.001, .greedy, .on_policy, options);
    try std.testing.expect(trained.timed_out);
    try std.testing.expect(trained.capped);
    try std.testing.expectEqual(@as(u32, 0), trained.moves);
    try std.testing.expect(trained.max_tile <= 4);
    try std.testing.expectEqualSlices(f32, before, net.storage);
    const delayed = trainGameDelayedTDLambdaOpts(&net, rng.random(), 0.001, .greedy, options);
    try std.testing.expect(delayed.timed_out);
    var stats: Stats = .{};
    const played = playWithStatsOpts(&net, rng.random(), .greedy, &stats, options);
    try std.testing.expect(played.timed_out);
    try std.testing.expectEqual(@as(u32, 0), played.moves);
    var tc = try TemporalCoherence.init(std.testing.allocator, &net);
    defer tc.deinit(std.testing.allocator);
    const coherent = trainGameTemporalCoherenceOpts(&tc, rng.random(), 0.001, .greedy, options);
    try std.testing.expect(coherent.timed_out);
    try std.testing.expectEqual(@as(u32, 0), coherent.moves);
    try std.testing.expectEqualSlices(f32, before, net.storage);
}

test "trainGame keeps the on-policy target and max backup differs from it (#131)" {
    const gpa = std.testing.allocator;
    var a = try TupleNet.initWith(gpa, .light);
    defer a.deinit(gpa);
    var b = try TupleNet.initWith(gpa, .light);
    defer b.deinit(gpa);
    var c = try TupleNet.initWith(gpa, .light);
    defer c.deinit(gpa);

    var pa = std.Random.DefaultPrng.init(7);
    var pb = std.Random.DefaultPrng.init(7);
    var pc = std.Random.DefaultPrng.init(7);
    _ = trainGame(&a, pa.random(), 0.05, .greedy);
    _ = trainGameWith(&b, pb.random(), 0.05, .greedy, .on_policy);
    _ = trainGameWith(&c, pc.random(), 0.05, .greedy, .max_backup);

    try std.testing.expectEqualSlices(f32, a.storage, b.storage);
    try std.testing.expect(!std.mem.eql(f32, a.storage, c.storage));
}
