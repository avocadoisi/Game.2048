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

/// Heuristic: many empty cells, plus a bonus when the largest tile sits in a corner.
/// shortcut: a hand-tuned score, not learned; replace with an n-tuple table later.
pub fn evaluate(board: Board) f64 {
    const last = core.BoardSize - 1;
    var empties: f64 = 0;
    var max_tile: core.Tile = 0;
    for (board) |row| {
        for (row) |value| {
            if (value == 0) empties += 1;
            max_tile = @max(max_tile, value);
        }
    }
    const corners = [_]core.Tile{ board[0][0], board[0][last], board[last][0], board[last][last] };
    var corner_bonus: f64 = 0;
    for (corners) |value| {
        if (value != 0 and value == max_tile) corner_bonus = @log2(@as(f64, @floatFromInt(max_tile)));
    }
    return empties * 10 + corner_bonus * 5;
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

fn maxValue(state: GameState, depth: u8) f64 {
    if (depth == 0) return evaluate(state.board);
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
    if (empties == 0) return evaluate(state.board);
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

test "evaluate gives a bonus when the largest tile is in a corner" {
    const corner: Board = .{
        .{ 64, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const middle: Board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 64, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expect(evaluate(corner) > evaluate(middle));
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
