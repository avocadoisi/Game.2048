const std = @import("std");
const core = @import("core");

const Board = core.Board;
const Direction = core.Direction;
const GameState = core.GameState;

/// Returned for a board where the player has no legal move.
const lost_score: f64 = -1e9;

/// Pick the move with the best expectimax value; null when no move changes the board.
pub fn bestMove(state: GameState, depth: u8) ?Direction {
    var best: ?Direction = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = chanceValue(next, depth);
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

/// Snake path 1..16: row 0 left to right, row 1 right to left, and so on.
/// The largest tile is expected at the start (top-left).
const order: [core.BoardCellCount]core.Position = blk: {
    var cells: [core.BoardCellCount]core.Position = undefined;
    for (0..core.BoardCellCount) |i| {
        const row = i / core.BoardSize;
        const offset = i % core.BoardSize;
        const col = if (row % 2 == 0) offset else core.BoardSize - 1 - offset;
        cells[i] = .{ .row = row, .col = col };
    }
    break :blk cells;
};

/// Heuristic: many empty cells, minus a penalty (x30) for each adjacent pair along the snake path
/// where the later cell holds a larger tile (log2 scale).
/// shortcut: a hand-set order, not learned; replace with an n-tuple table later.
pub fn evaluate(board: Board) f64 {
    var empties: f64 = 0;
    var penalty: f64 = 0;
    var prev: ?f64 = null;
    for (order) |p| {
        const value = board[p.row][p.col];
        if (value == 0) {
            empties += 1;
            continue;
        }
        const level = @log2(@as(f64, @floatFromInt(value)));
        if (prev) |before| penalty += @max(0, level - before);
        prev = level;
    }
    return empties * 10 - penalty * 30;
}

pub const GameResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
};

/// Play one full game with the solver until no move changes the board.
pub fn playGame(rng: anytype, depth: u8) GameResult {
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    while (bestMove(state, depth)) |d| {
        _ = state.move(d, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| {
        for (row) |value| max_tile = @max(max_tile, value);
    }
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves };
}

/// Extra penalty when only `down` is legal: with the top-left corner as the anchor,
/// a forced down move breaks the order.
/// shortcut: fixed penalty, not tuned; tune with `zig build solve` later.
const forced_down_penalty: f64 = 100;

pub fn evaluateState(state: GameState) f64 {
    var score = evaluate(state.board);
    for ([_]Direction{ .up, .left, .right }) |d| {
        var next = state;
        if (next.slide(d).changed) return score;
    }
    var down = state;
    if (down.slide(.down).changed) score -= forced_down_penalty;
    return score;
}

fn maxValue(state: GameState, depth: u8) f64 {
    if (depth == 0) return evaluateState(state);
    var best: f64 = lost_score;
    var any_move = false;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        any_move = true;
        best = @max(best, chanceValue(next, depth));
    }
    if (!any_move) return lost_score;
    return best;
}

/// Average over the random spawn: each empty cell gets 2 (p=0.9) or 4 (p=0.1).
fn chanceValue(state: GameState, depth: u8) f64 {
    var total: f64 = 0;
    var empties: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (state.board[r][c] != 0) continue;
            empties += 1;
            for ([_]struct { value: core.Tile, p: f64 }{
                .{ .value = 2, .p = 0.9 },
                .{ .value = 4, .p = 0.1 },
            }) |spawn| {
                var next = state;
                next.board[r][c] = spawn.value;
                total += spawn.p * maxValue(next, depth - 1);
            }
        }
    }
    if (empties == 0) return evaluateState(state);
    return total / empties;
}

test "bestMove returns null when no move changes the board" {
    const state: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), bestMove(state, 1));
}

test "bestMove takes a merge over a move that does nothing" {
    // only left/right merge the pair; up/down leave the board unchanged
    const state: GameState = .{ .board = .{
        .{ 2, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    const move = bestMove(state, 1).?;
    try std.testing.expect(move == .left or move == .right);
}

test "order follows the snake path 1..16" {
    try std.testing.expectEqual(core.Position{ .row = 0, .col = 0 }, order[0]);
    try std.testing.expectEqual(core.Position{ .row = 1, .col = 3 }, order[4]);
    try std.testing.expectEqual(core.Position{ .row = 1, .col = 0 }, order[7]);
    try std.testing.expectEqual(core.Position{ .row = 2, .col = 0 }, order[8]);
    try std.testing.expectEqual(core.Position{ .row = 3, .col = 0 }, order[15]);
}

test "evaluateState penalizes a board where only down is legal" {
    const forced_down: GameState = .{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 4, 8, 16, 32 },
        .{ 8, 16, 32, 64 },
        .{ 0, 0, 0, 0 },
    } };
    try std.testing.expect(evaluateState(forced_down) < evaluate(forced_down.board));
}

test "evaluate penalizes a larger tile placed after a smaller one in the order" {
    const sorted: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const reversed: Board = .{
        .{ 16, 8, 4, 2 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expect(evaluate(reversed) > evaluate(sorted));
}

test "evaluate rewards a large tile at an earlier position in the order" {
    const early: Board = .{
        .{ 64, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const late: Board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 2, 64 },
    };
    try std.testing.expect(evaluate(early) > evaluate(late));
}

test "evaluate prefers boards with more empty cells" {
    const sparse: Board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const dense: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 2, 4, 8, 16 },
        .{ 2, 4, 8, 16 },
        .{ 2, 4, 8, 0 },
    };
    try std.testing.expect(evaluate(sparse) > evaluate(dense));
}

test "playGame finishes a game and reports its result" {
    var prng = std.Random.DefaultPrng.init(1);
    const result = playGame(prng.random(), 1);
    try std.testing.expect(result.moves > 0);
    try std.testing.expect(result.max_tile >= 2);
}
