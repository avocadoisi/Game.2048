//! Bitboard expectimax with a row-table leaf (#349).
//!
//! The board is a u64 of sixteen 4-bit tile levels (0 = empty, level = log2 tile), every move is
//! four table lookups, and the leaf evaluation of `solver.evaluateW` is precomputed per row. The
//! eval is split exactly, not approximated: the order penalty is row-local plus one term per row
//! boundary, and the column features come from the transposed board, so the value is bit-identical
//! to `solver.evaluateStateW` for the shipped weight set (a test pins this).
//!
//! Native only for now: the tables and the TT cost ~1.8 MiB, which the wasm budget does not have.
//! Tiles above 32768 (level 15) saturate, so a merge at the very top of the game is undervalued;
//! 16384 is the highest tile any shipped policy has reached.

const std = @import("std");
const core = @import("core");
const solver = @import("solver");

const Direction = core.Direction;

pub const Board = u64;

/// Level that marks a tile too large for four bits; merges clamp here.
const max_level: u8 = 15;
const lost_score: f64 = -1e9;
/// Every cell empty, the board a game starts from before its two spawns.
pub const zero_board: core.Board = .{
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 0 },
};

// ---- tables (filled by `init`, read-only afterwards) ----

var left_delta: [65536]u16 = undefined;
var right_delta: [65536]u16 = undefined;
var can_left: [65536]u8 = undefined;
var feat: [65536]u64 = undefined;
var tables_ready = false;

/// Build the tables. Call once, before any search and before threads start.
pub fn init() void {
    if (tables_ready) return;
    for (0..65536) |i| {
        const row: u16 = @intCast(i);
        left_delta[row] = row ^ mergeRowLeft(row);
        right_delta[row] = row ^ reverseRow(mergeRowLeft(reverseRow(row)));
        can_left[row] = @intFromBool(rowMovesLeft(row));
        feat[row] = encodeFeat(rowFeat(row));
    }
    tables_ready = true;
}

// ---- packing ----

pub fn levelOf(tile: core.Tile) u8 {
    if (tile == 0) return 0;
    return @min(@as(u8, @intCast(@ctz(tile))), max_level);
}

pub fn pack(board: core.Board) Board {
    var b: Board = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            const shift: u6 = @intCast(4 * (core.BoardSize * r + c));
            b |= @as(Board, levelOf(board[r][c])) << shift;
        }
    }
    return b;
}

pub fn unpack(b: Board) core.Board {
    var board: core.Board = zero_board;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            const shift: u6 = @intCast(4 * (core.BoardSize * r + c));
            const level: u8 = @intCast((b >> shift) & 0xf);
            board[r][c] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
        }
    }
    return board;
}

fn rowAt(b: Board, r: usize) u16 {
    return @truncate(b >> @intCast(16 * r));
}

fn reverseRow(row: u16) u16 {
    return ((row & 0x000f) << 12) | ((row & 0x00f0) << 4) | ((row & 0x0f00) >> 4) | ((row & 0xf000) >> 12);
}

/// 4x4 transpose of 4-bit cells, row-major: cell (r, c) moves to nibble 4c + r.
pub fn transpose(x: Board) Board {
    const a1 = x & 0xf0f00f0ff0f00f0f;
    const a2 = x & 0x0000f0f00000f0f0;
    const a3 = x & 0x0f0f00000f0f0000;
    const a = a1 | (a2 << 12) | (a3 >> 12);
    const b1 = a & 0xff00ff0000ff00ff;
    const b2 = a & 0x00ff00ff00000000;
    const b3 = a & 0x00000000ff00ff00;
    return b1 | (b2 >> 24) | (b3 << 24);
}

/// One row of four levels merged to the left; each pair merges once and a merged tile never
/// merges again. Levels clamp at `max_level`.
fn mergeRowLeft(row: u16) u16 {
    var cells: [4]u8 = undefined;
    var n: usize = 0;
    for (0..4) |i| {
        const v: u8 = @truncate((row >> @intCast(4 * i)) & 0xf);
        if (v != 0) {
            cells[n] = v;
            n += 1;
        }
    }
    var out: [4]u8 = .{ 0, 0, 0, 0 };
    var m: usize = 0;
    var i: usize = 0;
    while (i < n) {
        if (i + 1 < n and cells[i] == cells[i + 1]) {
            out[m] = @min(cells[i] + 1, max_level);
            m += 1;
            i += 2;
        } else {
            out[m] = cells[i];
            m += 1;
            i += 1;
        }
    }
    var result: u16 = 0;
    for (0..4) |k| result |= @as(u16, out[k]) << @intCast(4 * k);
    return result;
}

fn rowMovesLeft(row: u16) bool {
    var seen_empty = false;
    var prev: u8 = 0;
    for (0..4) |i| {
        const v: u8 = @truncate((row >> @intCast(4 * i)) & 0xf);
        if (v == 0) {
            seen_empty = true;
            continue;
        }
        if (seen_empty or v == prev) return true;
        prev = v;
    }
    return false;
}

// ---- moves ----

pub fn slide(b: Board, d: Direction) Board {
    return switch (d) {
        .left => slideRows(b, &left_delta),
        .right => slideRows(b, &right_delta),
        .up => transpose(slideRows(transpose(b), &left_delta)),
        .down => transpose(slideRows(transpose(b), &right_delta)),
    };
}

fn slideRows(b: Board, table: *const [65536]u16) Board {
    const d0: Board = table[rowAt(b, 0)];
    const d1: Board = table[rowAt(b, 1)];
    const d2: Board = table[rowAt(b, 2)];
    const d3: Board = table[rowAt(b, 3)];
    return b ^ (d0 | (d1 << 16) | (d2 << 32) | (d3 << 48));
}

fn movesLeft(b: Board) bool {
    return can_left[rowAt(b, 0)] != 0 or can_left[rowAt(b, 1)] != 0 or
        can_left[rowAt(b, 2)] != 0 or can_left[rowAt(b, 3)] != 0;
}

fn movesRight(b: Board) bool {
    return can_left[reverseRow(rowAt(b, 0))] != 0 or can_left[reverseRow(rowAt(b, 1))] != 0 or
        can_left[reverseRow(rowAt(b, 2))] != 0 or can_left[reverseRow(rowAt(b, 3))] != 0;
}

// ---- evaluation ----

const Feat = struct {
    empty: u8,
    order: u8,
    merge: u8,
    mono: u8,
    sum: u8,
    first: u8,
    last: u8,
    max: u8,
};

fn encodeFeat(f: Feat) u64 {
    return @as(u64, f.empty) |
        (@as(u64, f.order) << 4) |
        (@as(u64, f.merge) << 11) |
        (@as(u64, f.mono) << 14) |
        (@as(u64, f.sum) << 21) |
        (@as(u64, f.first) << 27) |
        (@as(u64, f.last) << 31) |
        (@as(u64, f.max) << 35);
}

fn decodeFeat(x: u64) Feat {
    return .{
        .empty = @truncate(x & 0xf),
        .order = @truncate((x >> 4) & 0x7f),
        .merge = @truncate((x >> 11) & 0x7),
        .mono = @truncate((x >> 14) & 0x7f),
        .sum = @truncate((x >> 21) & 0x3f),
        .first = @truncate((x >> 27) & 0xf),
        .last = @truncate((x >> 31) & 0xf),
        .max = @truncate((x >> 35) & 0xf),
    };
}

/// Row features in left-to-right order: the order penalty's rise against the previous non-empty
/// cell inside the row, adjacent equal pairs, the monotonicity violation of the line, the level
/// sum, the first/last non-empty level and the largest level.
fn rowFeat(row: u16) Feat {
    var c: [4]u8 = undefined;
    for (0..4) |i| c[i] = @truncate((row >> @intCast(4 * i)) & 0xf);
    var f: Feat = .{ .empty = 0, .order = 0, .merge = 0, .mono = 0, .sum = 0, .first = 0, .last = 0, .max = 0 };
    var prev: u8 = 0;
    for (c) |v| {
        if (v == 0) {
            f.empty += 1;
            continue;
        }
        if (prev != 0 and v > prev) f.order += v - prev;
        prev = v;
        if (f.first == 0) f.first = v;
        f.last = v;
        f.sum += v;
        f.max = @max(f.max, v);
    }
    for (0..3) |i| {
        if (c[i] != 0 and c[i] == c[i + 1]) f.merge += 1;
    }
    var rising: u16 = 0;
    var falling: u16 = 0;
    for (0..3) |i| {
        if (c[i + 1] > c[i]) rising += c[i + 1] - c[i] else falling += c[i] - c[i + 1];
    }
    f.mono = @intCast(@min(rising, falling));
    return f;
}

fn checkWeights(w: solver.Weights) void {
    std.debug.assert(w.layout == .snake and w.anchor == 0 and
        w.ordered == core.BoardCellCount and w.smoothness == 0 and
        w.tile_sum == 0 and w.fragility == 0);
}

/// `solver.evaluateStateW` on the packed board: the same leaf rule, including the `forced_down`
/// penalty when only down is legal. `w` must be a `snake`/anchor-0/full-order weight set.
pub fn evalState(b: Board, w: solver.Weights) f64 {
    checkWeights(w);
    const t = transpose(b);
    var score = evalParts(b, t, w);
    if (movesLeft(t)) return score;
    if (movesLeft(b)) return score;
    if (movesRight(b)) return score;
    if (movesRight(t)) score -= w.forced_down;
    return score;
}

fn evalParts(b: Board, t: Board, w: solver.Weights) f64 {
    var empties: u32 = 0;
    var order: u32 = 0;
    var merge: u32 = 0;
    var mono: u32 = 0;
    var sum: u32 = 0;
    var max_lv: u8 = 0;
    var prev: u8 = 0;
    inline for (0..4) |r| {
        // odd rows are walked right to left, so their features come from the reversed row: the
        // within-row rises have to be counted in path order too, not just reversed at the ends
        const row = rowAt(b, r);
        const f = decodeFeat(feat[if (r % 2 == 0) row else reverseRow(row)]);
        empties += f.empty;
        merge += f.merge;
        mono += f.mono;
        sum += f.sum;
        max_lv = @max(max_lv, f.max);
        order += f.order;
        if (prev != 0 and f.first > prev) order += f.first - prev;
        if (f.last != 0) prev = f.last;
    }
    inline for (0..4) |r| {
        const f = decodeFeat(feat[rowAt(t, r)]);
        merge += f.merge;
        mono += f.mono;
    }
    const order_penalty = if (max_lv >= 12) w.order_penalty_late else w.order_penalty;
    var base = @as(f64, @floatFromInt(empties)) * w.empty -
        @as(f64, @floatFromInt(order)) * order_penalty +
        @as(f64, @floatFromInt(merge)) * w.merge;
    if (w.monotonic != 0) base -= @as(f64, @floatFromInt(mono)) * w.monotonic;
    if (w.tile_sum != 0) base += @as(f64, @floatFromInt(sum)) * w.tile_sum;
    return base;
}

// ---- search ----

pub const TTEntry = struct {
    key: Board = 0,
    value: f32 = 0,
    depth: u8 = 0,
    kind: u8 = 0,
    stamp: u8 = 0,
};

const spawn_levels = [_]struct { level: u4, p: f64 }{
    .{ .level = 1, .p = 0.9 },
    .{ .level = 2, .p = 0.1 },
};

/// One search over a caller-owned transposition table. `tt.len` must be a power of two; the table
/// is reused across moves with a per-move stamp, so it must start zeroed (`clearTable`).
pub const Search = struct {
    tt: []TTEntry,
    w: solver.Weights = solver.baseline,
    depth_limit: u8 = 3,
    cutoff: f64 = 1e-4,
    stamp: u8 = 0,
    /// Leaf evaluations, for the throughput column.
    nodes: u64 = 0,
    /// When set, leaves are scored by the net instead of the row heuristic.
    net: ?*const solver.Net = null,

    pub fn clearTable(self: *Search) void {
        @memset(self.tt, TTEntry{});
        self.stamp = 0;
    }

    /// Leaf count since the last call: `clearTable` runs whenever the stamp wraps, so the counter
    /// is reset here and not there.
    pub fn resetNodes(self: *Search) void {
        self.nodes = 0;
    }

    /// Score leaves with the shipped net instead of the row heuristic (same search, other leaf).
    pub fn useNet(self: *Search, net: *const solver.Net) void {
        self.net = net;
    }

    pub fn bestMove(self: *Search, b: Board) ?Direction {
        self.stamp +%= 1;
        if (self.stamp == 0) self.clearTable();
        var best: ?Direction = null;
        var best_value: f64 = -std.math.inf(f64);
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            const next = slide(b, d);
            if (next == b) continue;
            const value = self.spawnValue(next, self.depth_limit, 1.0);
            if (value > best_value) {
                best_value = value;
                best = d;
            }
        }
        return best;
    }

    fn moveValue(self: *Search, b: Board, depth: u8, cprob: f64) f64 {
        if (depth == 0) return self.leaf(b);
        if (self.get(b, depth, 1)) |v| return v;
        var best: f64 = lost_score;
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            const next = slide(b, d);
            if (next == b) continue;
            best = @max(best, self.spawnValue(next, depth, cprob));
        }
        self.put(b, depth, 1, best);
        return best;
    }

    fn spawnValue(self: *Search, b: Board, depth: u8, cprob: f64) f64 {
        if (self.get(b, depth, 0)) |v| return v;
        const filled = @popCount((b | (b >> 1) | (b >> 2) | (b >> 3)) & 0x1111111111111111);
        if (filled == core.BoardCellCount) return self.leaf(b);
        const cell_prob = cprob / @as(f64, @floatFromInt(core.BoardCellCount - filled));
        var total: f64 = 0;
        var mass: f64 = 0;
        var i: u6 = 0;
        while (i < core.BoardCellCount) : (i += 1) {
            const shift: u6 = i * 4;
            if ((b >> shift) & 0xf != 0) continue;
            for (spawn_levels) |s| {
                const p = cell_prob * s.p;
                // the chance node next to the leaves keeps every branch: cutting there bends the value
                if (depth > 1 and p < self.cutoff) continue;
                total += p * self.moveValue(b | (@as(Board, s.level) << shift), depth - 1, p);
                mass += p;
            }
        }
        const value = if (mass == 0) self.leaf(b) else total / mass;
        self.put(b, depth, 0, value);
        return value;
    }

    fn leaf(self: *Search, b: Board) f64 {
        self.nodes += 1;
        if (self.net) |net| return solver.netLeaf(net, unpack(b));
        return evalState(b, self.w);
    }

    fn slot(self: *Search, b: Board, depth: u8, kind: u8) usize {
        var h = b *% 0x9e3779b97f4a7c15;
        h ^= (@as(u64, depth) << 40) ^ (@as(u64, kind) << 48);
        h ^= h >> 29;
        return @as(usize, @intCast(h)) & (self.tt.len - 1);
    }

    fn get(self: *Search, b: Board, depth: u8, kind: u8) ?f64 {
        const e = self.tt[self.slot(b, depth, kind)];
        if (e.stamp == self.stamp and e.key == b and e.depth == depth and e.kind == kind) return e.value;
        return null;
    }

    fn put(self: *Search, b: Board, depth: u8, kind: u8, value: f64) void {
        self.tt[self.slot(b, depth, kind)] = .{
            .key = b,
            .value = @floatCast(value),
            .depth = depth,
            .kind = kind,
            .stamp = self.stamp,
        };
    }
};

// ---- tests ----

fn testBoard(levels: [core.BoardCellCount]u8) Board {
    var b: Board = 0;
    for (levels, 0..) |level, i| b |= @as(Board, level) << @intCast(4 * i);
    return b;
}

test "pack and unpack round-trip a board" {
    const board: core.Board = .{
        .{ 2, 0, 8, 64 },
        .{ 0, 4, 0, 1024 },
        .{ 16, 32, 0, 0 },
        .{ 0, 0, 2, 32768 },
    };
    try std.testing.expectEqual(board, unpack(pack(board)));
}

test "tiles above 32768 saturate at level 15" {
    try std.testing.expectEqual(@as(u8, 15), levelOf(32768));
    try std.testing.expectEqual(@as(u8, 15), levelOf(65536));
}

test "transpose matches a naive transpose on random boards" {
    var prng = std.Random.DefaultPrng.init(7);
    for (0..2000) |_| {
        var b: Board = 0;
        for (0..core.BoardCellCount) |i| b |= @as(Board, prng.random().uintLessThan(u8, 16)) << @intCast(4 * i);
        var naive: core.Board = zero_board;
        const board = unpack(b);
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| naive[c][r] = board[r][c];
        }
        try std.testing.expectEqual(pack(naive), transpose(b));
    }
}

test "row merge matches the reference merge of core" {
    for (0..65536) |i| {
        const row: u16 = @intCast(i);
        var board: core.Board = zero_board;
        for (0..4) |c| {
            const level: u8 = @truncate((row >> @intCast(4 * c)) & 0xf);
            board[0][c] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
        }
        var state: core.GameState = .{ .board = board };
        _ = state.slide(.left);
        for (0..4) |c| {
            const want = levelOf(state.board[0][c]);
            const got: u8 = @truncate((mergeRowLeft(row) >> @intCast(4 * c)) & 0xf);
            try std.testing.expectEqual(want, got);
        }
    }
}

test "slide matches core in all four directions" {
    init();
    var prng = std.Random.DefaultPrng.init(11);
    for (0..5000) |_| {
        var board: core.Board = zero_board;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                const level = prng.random().uintLessThan(u8, 6);
                board[r][c] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
            }
        }
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            var state: core.GameState = .{ .board = board };
            _ = state.slide(d);
            try std.testing.expectEqual(pack(state.board), slide(pack(board), d));
        }
    }
}

test "evalState matches evaluateStateW exactly" {
    init();
    var prng = std.Random.DefaultPrng.init(13);
    var checked: usize = 0;
    for (0..3000) |_| {
        var board: core.Board = zero_board;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                const level = prng.random().uintLessThan(u8, 8);
                board[r][c] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
            }
        }
        const state: core.GameState = .{ .board = board };
        try std.testing.expectEqual(solver.evaluateStateW(state, solver.baseline), evalState(pack(board), solver.baseline));
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 3000), checked);
}

test "evalState keeps the forced-down penalty" {
    init();
    // the only legal move is down: the top row is full and no row/column can merge
    const board: core.Board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    };
    const state: core.GameState = .{ .board = board };
    try std.testing.expectEqual(solver.evaluateStateW(state, solver.baseline), evalState(pack(board), solver.baseline));
}

test "search finds the only legal move and returns null on a dead board" {
    init();
    var tt: [1024]TTEntry = undefined;
    var s: Search = .{ .tt = &tt, .depth_limit = 2 };
    s.clearTable();

    // rows 0 and 1 are full and merge free, rows 2 and 3 are empty: only down moves
    const down_only: core.Board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expectEqual(@as(?Direction, .down), s.bestMove(pack(down_only)));

    const dead: core.Board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 8 },
    };
    try std.testing.expectEqual(@as(?Direction, null), s.bestMove(pack(dead)));
}

test "search returns the move its own values rank highest, twice in a row" {
    init();
    var tt: [4096]TTEntry = undefined;
    var s: Search = .{ .tt = &tt, .depth_limit = 1 };
    s.clearTable();
    const boards = [_]Board{
        pack(.{
            .{ 2, 2, 0, 0 },
            .{ 0, 0, 0, 0 },
            .{ 0, 0, 0, 0 },
            .{ 0, 0, 0, 0 },
        }),
        pack(.{
            .{ 2, 4, 0, 2 },
            .{ 0, 8, 0, 0 },
            .{ 2, 0, 4, 0 },
            .{ 0, 0, 0, 16 },
        }),
    };
    for (boards) |b| {
        var best: ?Direction = null;
        var best_value: f64 = -std.math.inf(f64);
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            const next = slide(b, d);
            if (next == b) continue;
            const value = s.spawnValue(next, s.depth_limit, 1.0);
            if (value > best_value) {
                best_value = value;
                best = d;
            }
        }
        // the second call runs with a later stamp over the same table
        try std.testing.expectEqual(best, s.bestMove(b));
        try std.testing.expectEqual(best, s.bestMove(b));
    }
}

test "depth 1 plays the same game as the reference expectimax" {
    if (!solver.slow_tests) return error.SkipZigTest;
    init();
    var tt: [1 << 12]TTEntry = undefined;
    var s: Search = .{ .tt = &tt, .depth_limit = 1 };
    s.clearTable();

    // one ply is the tree the two engines can agree on exactly: same seed in, same score out
    var packed_rng = std.Random.DefaultPrng.init(9000);
    var packed_game = core.GameState.gameStart(packed_rng.random());
    while (s.bestMove(pack(packed_game.board))) |d| {
        _ = packed_game.move(d, packed_rng.random());
    }
    var reference_rng = std.Random.DefaultPrng.init(9000);
    var reference_game = core.GameState.gameStart(reference_rng.random());
    while (solver.bestMove(reference_game, 1)) |d| {
        _ = reference_game.move(d, reference_rng.random());
    }
    try std.testing.expectEqual(reference_game.score, packed_game.score);
    try std.testing.expectEqual(core.maxTile(reference_game.board), core.maxTile(packed_game.board));
}
