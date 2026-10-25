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

pub const TupleNet = struct {
    storage: []f32,

    pub fn init(gpa: std.mem.Allocator) !TupleNet {
        const storage = try gpa.alloc(f32, total_entries);
        @memset(storage, 0);
        return .{ .storage = storage };
    }

    pub fn deinit(self: *TupleNet, gpa: std.mem.Allocator) void {
        gpa.free(self.storage);
    }

    pub fn table(self: *TupleNet, p: usize) []f32 {
        return self.storage[offsets[p]..][0..tableLen(p)];
    }

    pub fn value(self: *const TupleNet, idx: Indices) f64 {
        var v: f64 = 0;
        for (0..pattern_count) |p| v += self.storage[offsets[p] + idx[p]];
        return v;
    }

    pub fn eval(self: *const TupleNet, board: Board) f64 {
        return self.value(indices(board));
    }

    /// TD step: move every touched entry by `lr * (target - V) / n_tuples`; returns the TD error.
    pub fn train(self: *TupleNet, idx: Indices, target: f64, lr: f64) f64 {
        const delta = target - self.value(idx);
        const step: f32 = @floatCast(lr * delta / @as(f64, pattern_count));
        for (0..pattern_count) |p| self.storage[offsets[p] + idx[p]] += step;
        return delta;
    }
};

pub const file_magic = "2048-NT\x00";
pub const file_version: u32 = 1;
const endian_marker: u32 = 0x01020304;
const header_len: usize = 8 + 4 + 4 + 4;
pub const file_len = header_len + pattern_count * 4 + total_entries * 4;

/// File: magic, version, endianness marker, pattern count, per-table lengths (u32 LE), then f32 LE payload.
pub fn encode(gpa: std.mem.Allocator, net: *const TupleNet) ![]u8 {
    const bytes = try gpa.alloc(u8, file_len);
    @memcpy(bytes[0..8], file_magic);
    std.mem.writeInt(u32, bytes[8..12], file_version, .little);
    std.mem.writeInt(u32, bytes[12..16], endian_marker, .little);
    std.mem.writeInt(u32, bytes[16..20], pattern_count, .little);
    var pos = header_len;
    for (patterns, 0..) |_, p| {
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
    if (bytes.len != file_len) return error.WrongTableFileSize;
    if (!std.mem.eql(u8, bytes[0..8], file_magic)) return error.BadTableMagic;
    if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedTableVersion;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongTableEndianness;
    if (std.mem.readInt(u32, bytes[16..20], .little) != pattern_count) return error.WrongPatternCount;
    var pos = header_len;
    for (patterns, 0..) |_, p| {
        if (std.mem.readInt(u32, bytes[pos..][0..4], .little) != tableLen(p)) return error.WrongTableLength;
        pos += 4;
    }
    var net = try TupleNet.init(gpa);
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
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(file_len + 1));
    defer gpa.free(bytes);
    var parsed = try parse(gpa, bytes);
    defer parsed.deinit(gpa);
    @memcpy(net.storage, parsed.storage);
}

pub const Mode = enum { greedy, chance };

pub const PlayResult = struct { score: u32, max_tile: core.Tile, moves: u32 };
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

fn chooseMove(net: *const TupleNet, state: core.GameState, mode: Mode) ?Pick {
    return switch (mode) {
        .greedy => pickMove(net, state),
        .chance => chanceMove(net, state),
    };
}

pub fn bestMove(net: *const TupleNet, state: core.GameState, mode: Mode) ?core.Direction {
    return if (chooseMove(net, state, mode)) |p| p.dir else null;
}

/// Play one game to game over with the table policy.
pub fn play(net: *const TupleNet, rng: std.Random, mode: Mode) PlayResult {
    var state = core.GameState.gameStart(rng);
    var moves: u32 = 0;
    while (chooseMove(net, state, mode)) |pick| {
        _ = state.move(pick.dir, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| for (row) |value| {
        max_tile = @max(max_tile, value);
    };
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves };
}

/// TD(0) on afterstates over one self-play game.
pub fn trainGame(net: *TupleNet, rng: std.Random, lr: f64, mode: Mode) GameResult {
    var state = core.GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    while (chooseMove(net, state, mode)) |pick| {
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
    for (0..pattern_count) |p| net.storage[offsets[p] + idx[p]] = 0.5;

    const before = try std.testing.allocator.dupe(f32, net.storage);
    defer std.testing.allocator.free(before);
    _ = net.train(idx, 1.0, 0.1);

    var changed: usize = 0;
    for (net.storage, before) |now, old| {
        if (now != old) changed += 1;
    }
    try std.testing.expectEqual(pattern_count, changed);
}

test "save and load round-trip exactly" {
    var net = try testNet();
    defer net.deinit(std.testing.allocator);
    const idx = indices(sample_board);
    for (0..pattern_count) |p| net.storage[offsets[p] + idx[p]] = @floatCast(@as(f64, @floatFromInt(p)) + 0.25);

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
