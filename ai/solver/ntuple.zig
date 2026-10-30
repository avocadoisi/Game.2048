const std = @import("std");
const core = @import("core");

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
    // #106: the 2x3 rectangles split into two translation orbits under the 8 dihedral transforms
    // (8 positions around the edge, 4 positions through the middle). The first 6-tuple above is the
    // `min` index of orbit 0 for every board, so orbit 1 needs exactly this one more table.
    .{ .len = 6, .cells = .{ 4, 5, 6, 8, 9, 10 } }, // 2x3 middle-left (orbit 1)
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
/// There is one accumulator per (base pattern, symmetry transform); the table index is their minimum.
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

pub fn indices(board: Board) Indices {
    var acc: [pattern_count * 8]u32 = @splat(0);
    for (0..core.BoardCellCount) |cell| {
        const level: u32 = @intCast(levelOf(board[cell / core.BoardSize][cell % core.BoardSize]));
        for (slot_table.items[cell][0..slot_table.counts[cell]]) |slot| {
            acc[slot.acc] |= level << slot.shift;
        }
    }
    var out: Indices = undefined;
    for (0..pattern_count) |p| {
        var best: u32 = std.math.maxInt(u32);
        for (0..8) |o| best = @min(best, acc[p * 8 + o]);
        out[p] = best;
    }
    return out;
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
/// committed `ntuple_weights.bin` and every existing policy), `.wide` adds both 2x3 orbits (#106).
pub const Set = enum { light, full, wide };

pub const light_count = 5;
pub const full_count = 6;

pub fn countOf(set: Set) usize {
    return switch (set) {
        .light => light_count,
        .full => full_count,
        .wide => pattern_count,
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

    pub fn eval(self: *const TupleNet, board: Board) f64 {
        return self.value(indices(board));
    }

    /// TD step: move every touched entry by `lr * (target - V) / n_tuples`; returns the TD error.
    pub fn train(self: *TupleNet, idx: Indices, target: f64, lr: f64) f64 {
        const delta = target - self.value(idx);
        const step: f32 = @floatCast(lr * delta / @as(f64, @floatFromInt(self.count)));
        for (0..self.count) |p| self.storage[offsets[p] + idx[p]] += step;
        return delta;
    }
};

/// int16 view of a table: one scale per pattern, so the 4-tuples can use a coarser step than the
/// 6-tuples. `q * scale` reproduces the f32 value to within half a step (129 MiB -> 64.6 MiB, #129).
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
/// evaluation code that reads float tables (#129).
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
    @memcpy(bytes[0..8], quantized_magic);
    std.mem.writeInt(u32, bytes[8..12], file_version, .little);
    std.mem.writeInt(u32, bytes[12..16], endian_marker, .little);
    std.mem.writeInt(u32, bytes[16..20], @intCast(q.count), .little);
    var pos = header_len;
    for (0..q.count) |p| {
        std.mem.writeInt(u32, bytes[pos..][0..4], @intCast(tableLen(p)), .little);
        pos += 4;
    }
    for (0..q.count) |p| {
        std.mem.writeInt(u64, bytes[pos..][0..8], @bitCast(q.scales[p]), .little);
        pos += 8;
    }
    for (q.storage) |w| {
        std.mem.writeInt(i16, bytes[pos..][0..2], w, .little);
        pos += 2;
    }
    return bytes;
}

pub fn parseQuantized(gpa: std.mem.Allocator, bytes: []const u8) !Quantized {
    if (bytes.len < header_len) return error.WrongTableFileSize;
    if (!std.mem.eql(u8, bytes[0..8], quantized_magic)) return error.BadTableMagic;
    if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedTableVersion;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongTableEndianness;
    const n: usize = std.mem.readInt(u32, bytes[16..20], .little);
    if (n == 0 or n > pattern_count) return error.WrongPatternCount;
    if (bytes.len != quantizedFileLen(n)) return error.WrongTableFileSize;
    var pos = header_len;
    for (0..n) |p| {
        if (std.mem.readInt(u32, bytes[pos..][0..4], .little) != tableLen(p)) return error.WrongTableLength;
        pos += 4;
    }
    var q = try Quantized.initCount(gpa, n);
    errdefer q.deinit(gpa);
    for (0..n) |p| {
        q.scales[p] = @bitCast(std.mem.readInt(u64, bytes[pos..][0..8], .little));
        pos += 8;
    }
    for (q.storage) |*w| {
        w.* = std.mem.readInt(i16, bytes[pos..][0..2], .little);
        pos += 2;
    }
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
const endian_marker: u32 = 0x01020304;
const header_len: usize = 8 + 4 + 4 + 4;
/// File length for `n` tables; `.wide` is `fileLen(pattern_count)`.
pub fn fileLen(n: usize) usize {
    return header_len + n * 4 + entriesOf(n) * 4;
}

/// File: magic, version, endianness marker, pattern count, per-table lengths (u32 LE), then f32 LE payload.
pub fn encode(gpa: std.mem.Allocator, net: *const TupleNet) ![]u8 {
    const bytes = try gpa.alloc(u8, fileLen(net.count));
    @memcpy(bytes[0..8], file_magic);
    std.mem.writeInt(u32, bytes[8..12], file_version, .little);
    std.mem.writeInt(u32, bytes[12..16], endian_marker, .little);
    std.mem.writeInt(u32, bytes[16..20], @intCast(net.count), .little);
    var pos = header_len;
    for (0..net.count) |p| {
        std.mem.writeInt(u32, bytes[pos..][0..4], @intCast(tableLen(p)), .little);
        pos += 4;
    }
    for (net.storage) |w| {
        std.mem.writeInt(u32, bytes[pos..][0..4], @bitCast(w), .little);
        pos += 4;
    }
    return bytes;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !TupleNet {
    if (bytes.len < header_len) return error.WrongTableFileSize;
    if (!std.mem.eql(u8, bytes[0..8], file_magic)) return error.BadTableMagic;
    if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedTableVersion;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongTableEndianness;
    const n: usize = std.mem.readInt(u32, bytes[16..20], .little);
    if (n == 0 or n > pattern_count) return error.WrongPatternCount;
    if (bytes.len != fileLen(n)) return error.WrongTableFileSize;
    var pos = header_len;
    for (0..n) |p| {
        if (std.mem.readInt(u32, bytes[pos..][0..4], .little) != tableLen(p)) return error.WrongTableLength;
        pos += 4;
    }
    var net = try TupleNet.initCount(gpa, n);
    errdefer net.deinit(gpa);
    for (net.storage) |*w| {
        w.* = @bitCast(std.mem.readInt(u32, bytes[pos..][0..4], .little));
        pos += 4;
    }
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

/// Runner variant: policy mode, table set, and whether to load `table_path` before training.
pub const Variant = struct { mode: Mode = .greedy, set: Set = .full, resume_from_file: bool = false };

pub fn parseVariant(name: []const u8) ?Variant {
    if (std.mem.eql(u8, name, "base")) return .{};
    if (std.mem.eql(u8, name, "chance")) return .{ .mode = .chance };
    if (std.mem.eql(u8, name, "deep")) return .{ .mode = .deep };
    if (std.mem.eql(u8, name, "light")) return .{ .set = .light };
    if (std.mem.eql(u8, name, "lightdeep")) return .{ .mode = .deep, .set = .light };
    if (std.mem.eql(u8, name, "resume")) return .{ .resume_from_file = true };
    if (std.mem.eql(u8, name, "lightresume")) return .{ .resume_from_file = true, .set = .light };
    if (std.mem.eql(u8, name, "wide")) return .{ .set = .wide };
    if (std.mem.eql(u8, name, "widedeep")) return .{ .mode = .deep, .set = .wide };
    return null;
}

pub const PlayResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
    /// `net.eval` leaf calls summed over the game (0 for modes that do not search).
    leaves: u64 = 0,
};
pub const GameResult = struct { score: u32, moves: u32 };

/// Afterstate chosen by a policy, with its scaled reward, cached indices and value estimate.
const Pick = struct {
    dir: core.Direction,
    idx: Indices,
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
        const idx = indices(next.board);
        const pick: Pick = .{
            .dir = d,
            .idx = idx,
            .reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale,
            .value = net.value(idx),
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
            const idx = indices(moved.board);
            best = .{
                .dir = d,
                .idx = idx,
                .reward = @as(f64, @floatFromInt(result.score_delta)) / reward_scale,
                .value = net.value(idx),
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
            const idx = indices(next.board);
            best = .{
                .dir = d,
                .idx = idx,
                .reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale,
                .value = net.value(idx),
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

/// 1-ply greedy move from a quantized table; this is what `algo=ntuple` plays (#129).
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
    var state = core.GameState.gameStart(rng);
    var moves: u32 = 0;
    var move_stats: Stats = .{};
    while (true) {
        move_stats = .{};
        const pick = chooseMove(net, state, mode, &move_stats) orelse break;
        stats.leaves += move_stats.leaves;
        _ = state.move(pick.dir, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| for (row) |value| {
        max_tile = @max(max_tile, value);
    };
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves, .leaves = stats.leaves };
}

/// TD(0) on afterstates over one self-play game.
pub fn trainGame(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode) GameResult {
    var state = core.GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    var stats: Stats = .{};
    while (chooseMove(net, state, mode, &stats)) |pick| {
        moves += 1;
        if (pending) |p| _ = net.train(p.idx, p.reward + pick.value, lr);
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| _ = net.train(p.idx, p.reward, lr);
    return .{ .score = state.score, .moves = moves };
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
    try std.testing.expectEqual(@as(usize, 7), pattern_count);
    try std.testing.expectEqual(@as(usize, 5 * 65536), entryCount(.light));
    try std.testing.expectEqual(@as(usize, 5 * 65536 + 16777216), entryCount(.full));
    try std.testing.expectEqual(total_entries, entryCount(.wide));
    try std.testing.expectEqual(entryCount(.full) + 16777216, entryCount(.wide));

    var wide = try TupleNet.initWith(std.testing.allocator, .wide);
    defer wide.deinit(std.testing.allocator);
    try std.testing.expectEqual(pattern_count, wide.count);
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

test "light tables keep the five base tuples and drop the 6-tuple" {
    try std.testing.expectEqual(@as(usize, 5), light_count);
    try std.testing.expectEqual(@as(usize, 5 * 65536), entryCount(.light));
    try std.testing.expectEqual(entryCount(.wide), total_entries);
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
    try std.testing.expectEqual(Mode.deep, parseVariant("widedeep").?.mode);
    try std.testing.expectEqual(Set.wide, parseVariant("widedeep").?.set);
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
