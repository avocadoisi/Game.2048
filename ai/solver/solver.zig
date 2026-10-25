const std = @import("std");
const core = @import("core");

const Board = core.Board;
const Direction = core.Direction;
const GameState = core.GameState;
const nn = @import("nn");

/// Returned for a board where the player has no legal move.
const lost_score: f64 = -1e9;
/// shortcut: fixed width keeps branching cheap; tune against deeper searches if needed.
const beam_width = 32;
const beam_horizon = 3;

/// Weights of the snake heuristic (`evaluateW`, `evaluateStateW`). `baseline` is the hand-set version.
pub const Weights = struct {
    empty: f64 = 10,
    order_penalty: f64 = 30,
    merge: f64 = 10,
    /// extra penalty when only `down` is legal (the top-left anchor breaks)
    forced_down: f64 = 100,
};
pub const baseline: Weights = .{};

pub const Algorithm = enum { expectimax, greedy };

pub const Ranked = struct { dir: Direction, value: f64 };

/// Legal moves sorted best first. Ties keep the up, left, down, right order.
pub const Ranking = struct {
    items: [4]Ranked = undefined,
    len: usize = 0,
};

/// expectimax: value after the random spawn, searched `depth` player moves deep.
/// greedy: heuristic of the slid board, no spawn and no search (depth is ignored).
pub fn rank(state: GameState, algo: Algorithm, depth: u8) Ranking {
    var r: Ranking = .{};
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = switch (algo) {
            .expectimax => chanceValueW(next, depth, baseline),
            .greedy => evaluateState(next),
        };
        // insertion sort: strict > keeps the earlier direction on ties
        var i = r.len;
        while (i > 0 and r.items[i - 1].value < value) : (i -= 1) r.items[i] = r.items[i - 1];
        r.items[i] = .{ .dir = d, .value = value };
        r.len += 1;
    }
    return r;
}

/// Pick the move with the best expectimax value; null when no move changes the board.
pub fn bestMove(state: GameState, depth: u8) ?Direction {
    return bestMoveW(state, depth, baseline);
}

/// Expectimax best move with custom evaluation weights; same tie rule as `rank` (earlier direction wins).
pub fn bestMoveW(state: GameState, depth: u8, w: Weights) ?Direction {
    var best: ?Direction = null;
    var best_value: f64 = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = chanceValueW(next, depth, w);
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

/// Move-choosing policies compared in the ranking (`zig build ranking`).
pub const Policy = enum { random, greedy, expectimax, mcts, beam, expectimax_tt, ga, nn, nn2 };

/// Weights evolved by `zig build ga-rl` (holdout +4%, see docs/solver-history.md). 3 decimals as printed.
pub const ga_weights: Weights = .{ .empty = 7.028, .order_penalty = 43.174, .merge = 0, .forced_down = 0 };
/// Trained value net (`zig build nn`), embedded at build time.
const nn_weight_bytes = @embedFile("nn_weights.bin");

/// Next move for a policy; null when no move changes the board.
pub fn choose(policy: Policy, state: GameState, rng: std.Random, depth: u8) ?Direction {
    switch (policy) {
        .random => {
            var legal: [4]Direction = undefined;
            var n: usize = 0;
            for ([_]Direction{ .up, .left, .down, .right }) |d| {
                var next = state;
                if (next.slide(d).changed) {
                    legal[n] = d;
                    n += 1;
                }
            }
            if (n == 0) return null;
            return legal[rng.uintLessThan(usize, n)];
        },
        .greedy => {
            const r = rank(state, .greedy, 0);
            return if (r.len == 0) null else r.items[0].dir;
        },
        .expectimax => return bestMove(state, depth),
        .mcts => return mctsMove(state, rng),
        .beam => return beamMove(state),
        .expectimax_tt => return bestMoveTT(state, 3),
        .ga => return bestMoveW(state, depth, ga_weights),
        // shortcut: copies the 148 KB net per move; keep a static copy if this shows up in profiles
        .nn => {
            const net = nn.parse(nn_weight_bytes) catch unreachable;
            return nn.bestMove(&net, state, .chance);
        },
        // shortcut: same per-move net copy as `nn`; the 2-ply search itself dominates the cost.
        .nn2 => {
            const net = nn.parse(nn_weight_bytes) catch unreachable;
            return nn.bestMove(&net, state, .deep);
        },
    }
}

/// shortcut: fixed 200 full random rollouts per move; raise only while the 6 s/game limit holds.
const mcts_simulations: usize = 200;

const MctsArm = struct {
    dir: Direction,
    visits: usize = 0,
    reward: f64 = 0,
};

/// Root UCT selection with complete random games as rollouts.
fn mctsMove(state: GameState, rng: std.Random) ?Direction {
    var arms: [4]MctsArm = undefined;
    var arm_count: usize = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        arms[arm_count] = .{ .dir = d };
        arm_count += 1;
    }
    if (arm_count == 0) return null;

    for (0..mcts_simulations) |simulation| {
        var selected: usize = 0;
        var best_ucb: f64 = -std.math.inf(f64);
        for (arms[0..arm_count], 0..) |arm, i| {
            const ucb = if (arm.visits == 0)
                std.math.inf(f64)
            else
                arm.reward / @as(f64, @floatFromInt(arm.visits)) +
                    @sqrt(2 * @log(@as(f64, @floatFromInt(simulation + 1))) / @as(f64, @floatFromInt(arm.visits)));
            if (ucb > best_ucb) {
                best_ucb = ucb;
                selected = i;
            }
        }

        var rollout = state;
        _ = rollout.move(arms[selected].dir, rng);
        while (randomMove(&rollout, rng)) {}
        arms[selected].visits += 1;
        arms[selected].reward += @as(f64, @floatFromInt(rollout.score)) / 100000;
    }

    var best = arms[0];
    for (arms[1..arm_count]) |arm| {
        if (arm.reward / @as(f64, @floatFromInt(arm.visits)) > best.reward / @as(f64, @floatFromInt(best.visits))) best = arm;
    }
    return best.dir;
}

/// Apply one uniformly random legal move; false when the rollout is over.
fn randomMove(state: *GameState, rng: std.Random) bool {
    var legal: [4]Direction = undefined;
    var count: usize = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state.*;
        if (next.slide(d).changed) {
            legal[count] = d;
            count += 1;
        }
    }
    if (count == 0) return false;
    _ = state.move(legal[rng.uintLessThan(usize, count)], rng);
    return true;
}

pub fn playPolicy(rng: std.Random, policy: Policy, depth: u8) GameResult {
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    while (choose(policy, state, rng, depth)) |d| {
        _ = state.move(d, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| {
        for (row) |value| max_tile = @max(max_tile, value);
    }
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves };
}

const BeamNode = struct {
    state: GameState,
    first_move: Direction,
    value: f64,
};

/// Beam search over three player moves; each empty cell is considered as a 2 spawn.
fn beamMove(state: GameState) ?Direction {
    var beam: [beam_width]BeamNode = undefined;
    beam[0] = .{ .state = state, .first_move = .up, .value = evaluateState(state) };
    var beam_len: usize = 1;

    for (0..beam_horizon) |ply| {
        var next_beam: [beam_width]BeamNode = undefined;
        var next_len: usize = 0;
        for (beam[0..beam_len]) |node| {
            for ([_]Direction{ .up, .left, .down, .right }) |direction| {
                var moved = node.state;
                if (!moved.slide(direction).changed) continue;
                const first_move = if (ply == 0) direction else node.first_move;
                var spawned = false;
                for (0..core.BoardSize) |r| {
                    for (0..core.BoardSize) |c| {
                        if (moved.board[r][c] != 0) continue;
                        var child = moved;
                        child.board[r][c] = 2;
                        insertBeam(&next_beam, &next_len, .{
                            .state = child,
                            .first_move = first_move,
                            .value = evaluateState(child),
                        });
                        spawned = true;
                    }
                }
                if (!spawned) insertBeam(&next_beam, &next_len, .{
                    .state = moved,
                    .first_move = first_move,
                    .value = evaluateState(moved),
                });
            }
        }
        if (next_len == 0) {
            if (ply == 0) return null;
            break;
        }
        beam = next_beam;
        beam_len = next_len;
    }
    return beam[0].first_move;
}

fn insertBeam(items: *[beam_width]BeamNode, len: *usize, candidate: BeamNode) void {
    var i = len.*;
    while (i > 0 and items[i - 1].value < candidate.value) : (i -= 1) {
        if (i < beam_width) items[i] = items[i - 1];
    }
    if (i == beam_width) return;
    items[i] = candidate;
    if (len.* < beam_width) len.* += 1;
}

// shortcut: 8192 direct-mapped entries; collisions only reduce cache hits, increase this if needed.
const tt_entry_count = 8192;

const TTKind = enum { max, chance };

const TTEntry = struct {
    board: Board,
    value: f64,
    depth: u8,
    kind: TTKind,
    valid: bool,
};

fn bestMoveTT(state: GameState, depth: u8) ?Direction {
    var table: [tt_entry_count]TTEntry = undefined;
    for (&table) |*entry| entry.valid = false;

    var best: ?Direction = null;
    var best_value = lost_score;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = chanceValueTT(next, depth, &table);
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

fn ttSlot(board: Board, depth: u8, kind: TTKind) usize {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(std.mem.asBytes(&board));
    hasher.update(std.mem.asBytes(&depth));
    hasher.update(std.mem.asBytes(&kind));
    return @as(usize, @truncate(hasher.final())) % tt_entry_count;
}

fn ttGet(board: Board, depth: u8, kind: TTKind, table: *[tt_entry_count]TTEntry) ?f64 {
    const entry = table[ttSlot(board, depth, kind)];
    if (entry.valid and entry.depth == depth and entry.kind == kind and std.meta.eql(entry.board, board)) return entry.value;
    return null;
}

fn ttPut(board: Board, depth: u8, kind: TTKind, value: f64, table: *[tt_entry_count]TTEntry) void {
    table[ttSlot(board, depth, kind)] = .{ .board = board, .value = value, .depth = depth, .kind = kind, .valid = true };
}

fn maxValueTT(state: GameState, depth: u8, table: *[tt_entry_count]TTEntry) f64 {
    if (depth == 0) return evaluateState(state);
    if (ttGet(state.board, depth, .max, table)) |value| return value;
    var best: f64 = lost_score;
    var any_move = false;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        any_move = true;
        best = @max(best, chanceValueTT(next, depth, table));
    }
    if (!any_move) best = lost_score;
    ttPut(state.board, depth, .max, best, table);
    return best;
}

fn chanceValueTT(state: GameState, depth: u8, table: *[tt_entry_count]TTEntry) f64 {
    if (ttGet(state.board, depth, .chance, table)) |value| return value;
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
                total += spawn.p * maxValueTT(next, depth - 1, table);
            }
        }
    }
    const value = if (empties == 0) evaluateState(state) else total / empties;
    ttPut(state.board, depth, .chance, value, table);
    return value;
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
    return evaluateW(board, baseline);
}

pub fn evaluateW(board: Board, w: Weights) f64 {
    var empties: f64 = 0;
    var penalty: f64 = 0;
    var prev: ?f64 = null;
    for (order) |p| {
        const value = board[p.row][p.col];
        if (value == 0) {
            empties += 1;
            continue;
        }
        const level: f64 = @floatFromInt(@ctz(value));
        if (prev) |before| penalty += @max(0, level - before);
        prev = level;
    }
    return empties * w.empty - penalty * w.order_penalty + mergePairs(board) * w.merge;
}

/// Count of horizontally or vertically adjacent equal non-empty tiles: more merges available.
/// shortcut: hand-set weight (x10), tuned with `zig build solve` on fresh seeds.
fn mergePairs(board: Board) f64 {
    var n: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            const v = board[r][c];
            if (v == 0) continue;
            if (c + 1 < core.BoardSize and board[r][c + 1] == v) n += 1;
            if (r + 1 < core.BoardSize and board[r + 1][c] == v) n += 1;
        }
    }
    return n;
}

pub const GameResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
};

/// Play one full game with the solver until no move changes the board.
pub fn playGame(rng: std.Random, depth: u8) GameResult {
    return playPolicy(rng, .expectimax, depth);
}

pub fn evaluateState(state: GameState) f64 {
    return evaluateStateW(state, baseline);
}

pub fn evaluateStateW(state: GameState, w: Weights) f64 {
    var score = evaluateW(state.board, w);
    for ([_]Direction{ .up, .left, .right }) |d| {
        if (state.canSlide(d)) return score;
    }
    if (state.canSlide(.down)) score -= w.forced_down;
    return score;
}

fn maxValueW(state: GameState, depth: u8, w: Weights) f64 {
    if (depth == 0) return evaluateStateW(state, w);
    var best: f64 = lost_score;
    var any_move = false;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        any_move = true;
        best = @max(best, chanceValueW(next, depth, w));
    }
    if (!any_move) return lost_score;
    return best;
}

/// Average over the random spawn: each empty cell gets 2 (p=0.9) or 4 (p=0.1).
fn chanceValueW(state: GameState, depth: u8, w: Weights) f64 {
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
                total += spawn.p * maxValueW(next, depth - 1, w);
            }
        }
    }
    if (empties == 0) return evaluateStateW(state, w);
    return total / empties;
}

test "baseline weights reproduce evaluate on a hand-computed board" {
    // 12 empty cells x10, level steps 1->2->3->4 (three x30 penalties), no equal neighbours
    const sorted: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expectEqual(@as(f64, 30), evaluate(sorted));
    try std.testing.expectEqual(evaluate(sorted), evaluateW(sorted, baseline));
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

test "mcts returns only legal moves and null on a dead board" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var prng = std.Random.DefaultPrng.init(7);
    const move = choose(.mcts, state, prng.random(), 0).?;
    var repeat_prng = std.Random.DefaultPrng.init(7);
    try std.testing.expectEqual(move, choose(.mcts, state, repeat_prng.random(), 0).?);
    var next = state;
    try std.testing.expect(next.slide(move).changed);

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), choose(.mcts, dead, prng.random(), 0));
}

test "beam policy returns a legal move and null on a dead board" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var prng = std.Random.DefaultPrng.init(7);
    const move = choose(.beam, state, prng.random(), 0).?;
    var moved = state;
    try std.testing.expect(moved.slide(move).changed);

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), choose(.beam, dead, prng.random(), 0));
}

test "expectimax TT depth 3 matches bestMove on random-play boards" {
    var rng_state = std.Random.DefaultPrng.init(100);
    const rng = rng_state.random();
    var state = GameState.gameStart(rng);
    const sample_after = [_]usize{ 8, 16, 24 };
    var moves: usize = 0;
    for (sample_after) |target| {
        while (moves < target) : (moves += 1) {
            const direction = choose(.random, state, rng, 0) orelse return error.UnexpectedGameOver;
            _ = state.move(direction, rng);
        }
        try std.testing.expectEqual(bestMove(state, 3), choose(.expectimax_tt, state, rng, 0));
    }
}


test "random policy only picks legal moves" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var prng = std.Random.DefaultPrng.init(7);
    for (0..20) |_| {
        const d = choose(.random, state, prng.random(), 0).?;
        try std.testing.expect(d == .left or d == .right);
    }
}

test "rank lists legal moves best first" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    const r = rank(state, .greedy, 0);
    // up/down are no-ops on this board, so only left and right are ranked
    try std.testing.expectEqual(@as(usize, 2), r.len);
    try std.testing.expect(r.items[0].value >= r.items[1].value);
    try std.testing.expect(r.items[0].dir == .left or r.items[0].dir == .right);
    try std.testing.expectEqual(bestMove(state, 2), r.items[0].dir);
}

test "greedy ranks by one-ply evaluation, not expectimax" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    const greedy = rank(state, .greedy, 2);
    const expect = rank(state, .expectimax, 2);
    try std.testing.expectEqual(greedy.len, expect.len);
    try std.testing.expect(greedy.items[0].value != expect.items[0].value);
}

test "bestMove takes a merge over a move that does nothing" {
    // full board: only the top row's 2,2 pair merges left/right; up/down change nothing
    // (the old board let down move a tile and form a pair, which the merge bonus now rewards)
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
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

test "evaluate rewards adjacent equal tiles (merges available)" {
    const merges: Board = .{
        .{ 2, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const apart: Board = .{
        .{ 2, 0, 2, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expect(evaluate(merges) > evaluate(apart));
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
