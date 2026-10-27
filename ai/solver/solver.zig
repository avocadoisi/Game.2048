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
    /// order penalty once the board's largest tile reaches 4096 (endgame phase, issue #92).
    /// Default equals `order_penalty`, so shipped weights behave the same in both phases.
    order_penalty_late: f64 = 30,
    merge: f64 = 10,
    /// extra penalty when only `down` is legal (the top-left anchor breaks)
    forced_down: f64 = 100,
    /// fragility penalty weight (issue #101): boards the death net expects to
    /// die sooner score lower by `fragility * (0 - v)`. 0 keeps evaluateW as-is.
    fragility: f64 = 0,
    /// death-variant net whose value is ≈ -(moves to game over)/reward_scale.
    /// Without it the fragility term is skipped.
    death_net: ?*const nn.Net = null,
};
pub const baseline: Weights = .{ .order_penalty = 43.174, .order_penalty_late = 25 };

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
pub const Policy = enum { random, greedy, expectimax, mcts, beam, expectimax_tt, ga, nn, nn2, nnmcts, nnbeam };

/// Weights evolved by `zig build ga-rl` (holdout +4%, see docs/solver-history.md). 3 decimals as printed.
pub const ga_weights: Weights = .{ .empty = 7.028, .order_penalty = 43.174, .order_penalty_late = 43.174, .merge = 0, .forced_down = 0 };
/// Trained value net (`zig build nn`), embedded at build time.
const nn_weight_bytes = @embedFile("nn_weights.bin");

/// The embedded net parsed once per thread (issue #87). Every move passes this `*const Net` by
/// reference instead of re-parsing the ~148 KB weight block, so parallel ranking / HTTP / WASM each
/// keep their own copy with no shared mutable state.
threadlocal var cached_net: ?nn.Net = null;

/// Per-thread net, parsed from the embedded weights on first use and reused for every later move.
fn nnNet() *const nn.Net {
    if (cached_net == null) cached_net = nn.parse(nn_weight_bytes) catch unreachable;
    return &cached_net.?;
}

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
        // The net is parsed once per thread and reused by reference (issue #87), never copied per move.
        .nn, .nn2, .nnmcts, .nnbeam => {
            const net = nnNet();
            return switch (policy) {
                .nn => nn.bestMove(net, state, .chance),
                // the 2-ply search itself dominates the cost.
                .nn2 => nn.bestMove(net, state, .deep),
                .nnmcts => nnmctsMove(net, state, rng),
                // the sampled beam stays below nn2's leaf count.
                .nnbeam => if (nnBeamBest(net, state, rng)) |best| best.first_move else null,
                else => unreachable,
            };
        },
    }
}

/// shortcut: fixed 200 full random rollouts per move; raise only while the 6 s/game limit holds.
const mcts_simulations: usize = 200;

/// shortcut: 1,000 truncated rollouts per move with 8 random plies before the net leaf; the score
/// keeps climbing with more simulations (~27k at 2,000, docs/solver-history.md) at a linear cost.
const nnmcts_simulations: usize = 1000;
const nnmcts_rollout_moves: usize = 8;

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
        const selected = selectArm(arms[0..arm_count], simulation);
        var rollout = state;
        _ = rollout.move(arms[selected].dir, rng);
        while (randomMove(&rollout, rng) != null) {}
        arms[selected].visits += 1;
        arms[selected].reward += @as(f64, @floatFromInt(rollout.score)) / 100000;
    }
    return bestArm(arms[0..arm_count]).dir;
}

/// Root UCT with truncated random rollouts that end at the learned net's afterstate value.
fn nnmctsMove(net: *const nn.Net, state: GameState, rng: std.Random) ?Direction {
    var arms: [4]MctsArm = undefined;
    var arm_count: usize = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        arms[arm_count] = .{ .dir = d };
        arm_count += 1;
    }
    if (arm_count == 0) return null;

    const cache = nn.sharedValueCache(); // reused across moves (issue #88)
    var stats: nn.Stats = .{};
    for (0..nnmcts_simulations) |simulation| {
        const selected = selectArm(arms[0..arm_count], simulation);
        var rollout = state;
        const moved = rollout.move(arms[selected].dir, rng);
        arms[selected].visits += 1;
        arms[selected].reward += nnmctsRollout(net, cache, &stats, rollout, moved.score_delta, rng);
    }
    return bestArm(arms[0..arm_count]).dir;
}

/// Arm with the highest UCB; unvisited arms first, ties keep the earlier direction.
fn selectArm(arms: []const MctsArm, simulation: usize) usize {
    var selected: usize = 0;
    var best_ucb: f64 = -std.math.inf(f64);
    for (arms, 0..) |arm, i| {
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
    return selected;
}

/// Arm with the highest mean reward; ties keep the earlier direction.
fn bestArm(arms: []const MctsArm) MctsArm {
    var best = arms[0];
    for (arms[1..]) |arm| {
        if (arm.reward / @as(f64, @floatFromInt(arm.visits)) > best.reward / @as(f64, @floatFromInt(best.visits))) best = arm;
    }
    return best;
}

/// Truncated random rollout from the post-move board: score gained along it (plus `gained`) over
/// `nn.reward_scale`, then the net's afterstate value when the budget runs out. The leaf is an
/// afterstate before its spawn, the board shape the net was trained on; a dead board adds 0 for
/// the future, so terminal and leaf values share the net's scale.
fn nnmctsRollout(net: *const nn.Net, cache: *nn.ValueCache, stats: *nn.Stats, state: GameState, gained: u32, rng: std.Random) f64 {
    var rollout = state;
    var score = gained;
    var plies: usize = 0;
    while (plies < nnmcts_rollout_moves) : (plies += 1) {
        const direction = randomSlide(&rollout, rng) orelse break;
        score += rollout.slide(direction).score_delta;
        if (plies + 1 < nnmcts_rollout_moves) _ = rollout.spawnRandomTile(rng);
    }
    const gained_value = @as(f64, @floatFromInt(score)) / nn.reward_scale;
    if (rollout.isGameOver()) return gained_value;
    return gained_value + cache.get(net, rollout.board, stats);
}

/// Uniformly random direction that changes the board; null when the board has no legal move.
fn randomSlide(state: *const GameState, rng: std.Random) ?Direction {
    var legal: [4]Direction = undefined;
    var count: usize = 0;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state.*;
        if (next.slide(d).changed) {
            legal[count] = d;
            count += 1;
        }
    }
    if (count == 0) return null;
    return legal[rng.uintLessThan(usize, count)];
}

/// Apply one uniformly random legal move (slide + spawn); null when the rollout is over, else the score gained.
fn randomMove(state: *GameState, rng: std.Random) ?u32 {
    const direction = randomSlide(state, rng) orelse return null;
    return state.move(direction, rng).score_delta;
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

/// shortcut: fixed small beam so the sampled spawns stay cheap (issue #83); tune only if nnbeam beats nn2.
const nnbeam_width = 8;
const nnbeam_horizon = 3;
const nnbeam_spawn_samples = 2;

/// Leaf score: the NN afterstate value, the same leaf rule `nn` uses (`nn.forward`).
fn nnLeafValue(net: *const nn.Net, board: Board) f64 {
    var cache: nn.Cache = .{};
    return nn.forward(net, board, &cache);
}

const NnBeamNode = struct {
    state: GameState,
    /// root move of the path that reached this leaf; it is the move the policy returns
    first_move: Direction,
    /// score_delta / nn.reward_scale summed down the path
    reward: f64,
    /// reward + `nnLeafValue(state.board)`
    value: f64,
};

/// Beam search with the learned NN leaf and `nnbeam_spawn_samples` sampled spawns between plies;
/// the last ply ends on the afterstate, the leaf rule `nn2` uses (`chance(..., 0)`).
/// Spawns are drawn from `rng`, so a fixed seed reproduces the same move. Null when no move changes the board.
fn nnBeamBest(net: *const nn.Net, state: GameState, rng: std.Random) ?NnBeamNode {
    return nnBeamSearch(net, state, rng, nnbeam_horizon);
}

fn nnBeamSearch(net: *const nn.Net, state: GameState, rng: std.Random, horizon: u8) ?NnBeamNode {
    var beam: [nnbeam_width]NnBeamNode = undefined;
    beam[0] = .{ .state = state, .first_move = .up, .reward = 0, .value = 0 };
    var beam_len: usize = 1;

    for (0..horizon) |ply| {
        var next_beam: [nnbeam_width]NnBeamNode = undefined;
        var next_len: usize = 0;
        for (beam[0..beam_len]) |node| {
            for ([_]Direction{ .up, .left, .down, .right }) |direction| {
                var moved = node.state;
                const result = moved.slide(direction);
                if (!result.changed) continue;
                const first_move = if (ply == 0) direction else node.first_move;
                const reward = node.reward + @as(f64, @floatFromInt(result.score_delta)) / nn.reward_scale;
                if (ply + 1 == @as(usize, horizon)) {
                    insertNnBeam(&next_beam, &next_len, .{
                        .state = moved,
                        .first_move = first_move,
                        .reward = reward,
                        .value = reward + nnLeafValue(net, moved.board),
                    });
                    continue;
                }
                var spawned = false;
                for (0..nnbeam_spawn_samples) |_| {
                    var child = moved;
                    if (!sampleSpawn(&child, rng)) break;
                    spawned = true;
                    insertNnBeam(&next_beam, &next_len, .{
                        .state = child,
                        .first_move = first_move,
                        .reward = reward,
                        .value = reward + nnLeafValue(net, child.board),
                    });
                }
                if (!spawned) insertNnBeam(&next_beam, &next_len, .{
                    .state = moved,
                    .first_move = first_move,
                    .reward = reward,
                    .value = reward + nnLeafValue(net, moved.board),
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
    return beam[0];
}

/// Spawn on a uniformly random empty cell: 2 at 0.9, 4 at 0.1, like the real game.
/// False when the board is full, so the afterstate itself is the leaf.
fn sampleSpawn(state: *GameState, rng: std.Random) bool {
    const empties = nn.countEmpty(state.board);
    if (empties == 0) return false;
    const value: core.Tile = if (rng.uintLessThan(u32, 10) == 0) 4 else 2;
    var index = rng.uintLessThan(usize, empties);
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (state.board[r][c] != 0) continue;
            if (index == 0) {
                state.board[r][c] = value;
                return true;
            }
            index -= 1;
        }
    }
    unreachable; // countEmpty found an empty cell
}

fn insertNnBeam(items: *[nnbeam_width]NnBeamNode, len: *usize, candidate: NnBeamNode) void {
    var i = len.*;
    while (i > 0 and items[i - 1].value < candidate.value) : (i -= 1) {
        if (i < nnbeam_width) items[i] = items[i - 1];
    }
    if (i == nnbeam_width) return;
    items[i] = candidate;
    if (len.* < nnbeam_width) len.* += 1;
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
    return bestMoveTTWith(state, depth, ttWorkspace());
}

/// Per-thread table reused across moves and games (issue #88). An entry is only returned on a
/// full board + depth + kind match, so stale entries are misses at worst, never wrong values;
/// no shared mutable state because each thread keeps its own table.
threadlocal var tt_workspace: [tt_entry_count]TTEntry = undefined;
threadlocal var tt_workspace_ready: bool = false;

fn ttWorkspace() *[tt_entry_count]TTEntry {
    if (!tt_workspace_ready) {
        for (&tt_workspace) |*entry| entry.valid = false;
        tt_workspace_ready = true;
    }
    return &tt_workspace;
}

fn bestMoveTTWith(state: GameState, depth: u8, table: *[tt_entry_count]TTEntry) ?Direction {
    var best: ?Direction = null;
    var best_value = lost_score;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = chanceValueTT(next, depth, table);
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
    var max_level: f64 = 0;
    for (order) |p| {
        const value = board[p.row][p.col];
        if (value == 0) {
            empties += 1;
            continue;
        }
        const level: f64 = @floatFromInt(@ctz(value));
        if (prev) |before| penalty += @max(0, level - before);
        prev = level;
        max_level = @max(max_level, level);
    }
    // endgame phase once the largest tile reaches 4096 = 2^12 (issue #92)
    const order_penalty = if (max_level >= 12) w.order_penalty_late else w.order_penalty;
    const base = empties * w.empty - penalty * order_penalty + mergePairs(board) * w.merge;
    if (w.fragility == 0) return base;
    const net = w.death_net orelse return base;
    var cache: nn.Cache = .{};
    return base + w.fragility * (0 - nn.forward(net, board, &cache));
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
    // 12 empty cells x10, level steps 1->2->3->4 (three x43.174 penalties), no equal neighbours
    const sorted: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    try std.testing.expectApproxEqAbs(@as(f64, 120 - 3 * 43.174), evaluate(sorted), 1e-9);
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

test "nnmcts returns only legal moves and null on a dead board" {
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var prng = std.Random.DefaultPrng.init(7);
    const move = choose(.nnmcts, state, prng.random(), 0).?;
    var repeat_prng = std.Random.DefaultPrng.init(7);
    try std.testing.expectEqual(move, choose(.nnmcts, state, repeat_prng.random(), 0).?);
    var next = state;
    try std.testing.expect(next.slide(move).changed);

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), choose(.nnmcts, dead, prng.random(), 0));
}

test "nnmcts rollout value shares the NN scale and a dead board adds no future value" {
    const net = nn.parse(nn_weight_bytes) catch unreachable;
    var cache: nn.ValueCache = .{};
    var stats: nn.Stats = .{};
    var prng = std.Random.DefaultPrng.init(11);

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(f64, 0), nnmctsRollout(&net, &cache, &stats, dead, 0, prng.random()));

    const live: GameState = .{ .board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    const value = nnmctsRollout(&net, &cache, &stats, live, 0, prng.random());
    // unit scale like the net (remaining score / reward_scale); a raw total score would be in the thousands
    try std.testing.expect(-100 < value and value < 100);
    try std.testing.expect(value > 0);
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

test "nnbeam policy returns a legal move and null on a dead board" {
    // only left/right are legal here, so a deeper ply's direction would show up as an illegal move
    const state: GameState = .{ .board = .{
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
        .{ 256, 512, 1024, 2048 },
        .{ 4096, 8192, 16384, 32768 },
    } };
    var prng = std.Random.DefaultPrng.init(7);
    const move = choose(.nnbeam, state, prng.random(), 0).?;
    try std.testing.expect(move == .left or move == .right);
    var moved = state;
    try std.testing.expect(moved.slide(move).changed);

    const dead: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    try std.testing.expectEqual(@as(?Direction, null), choose(.nnbeam, dead, prng.random(), 0));
}

test "nnbeam reproduces the same move from the same seed" {
    var play = std.Random.DefaultPrng.init(11);
    const rng = play.random();
    var state = GameState.gameStart(rng);
    for (0..12) |_| {
        var first = std.Random.DefaultPrng.init(7);
        var repeat = std.Random.DefaultPrng.init(7);
        try std.testing.expectEqual(
            choose(.nnbeam, state, first.random(), 0),
            choose(.nnbeam, state, repeat.random(), 0),
        );
        const direction = choose(.random, state, rng, 0) orelse break;
        _ = state.move(direction, rng);
    }
}

test "nnbeam returns the root move of the best leaf, not a deeper direction" {
    const net = nn.parse(nn_weight_bytes) catch unreachable;
    var play = std.Random.DefaultPrng.init(23);
    const rng = play.random();
    var state = GameState.gameStart(rng);
    for (0..10) |_| {
        var prng = std.Random.DefaultPrng.init(7);
        const best = nnBeamBest(&net, state, prng.random()) orelse break;
        // the first move is a move of the root board and the leaf value is the NN value plus the path reward
        try std.testing.expect(state.canSlide(best.first_move));
        try std.testing.expectEqual(best.reward + nnLeafValue(&net, best.state.board), best.value);
        var repeat = std.Random.DefaultPrng.init(7);
        try std.testing.expectEqual(@as(?Direction, best.first_move), choose(.nnbeam, state, repeat.random(), 0));

        const direction = choose(.random, state, rng, 0) orelse break;
        _ = state.move(direction, rng);
    }
}

test "nnbeam evaluates the afterstate before spawn at the final ply" {
    const net = nn.parse(nn_weight_bytes) catch unreachable;
    const state: GameState = .{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 512, 1024, 2048, 0 },
        .{ 0, 0, 0, 0 },
    } };
    // horizon 1: the leaf must be the afterstate of the returned move, like `nn2` ends on `chance(..., 0)`
    var prng = std.Random.DefaultPrng.init(7);
    const best = nnBeamSearch(&net, state, prng.random(), 1).?;
    var afterstate = state;
    try std.testing.expect(afterstate.slide(best.first_move).changed);
    try std.testing.expectEqual(afterstate.board, best.state.board);
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

test "reused TT returns the same moves as a fresh table across games (#88)" {
    const warm = ttWorkspace();
    try std.testing.expectEqual(warm, ttWorkspace()); // one workspace per thread, reused
    for ([_]u64{ 100, 200 }) |seed| {
        var rng_a = std.Random.DefaultPrng.init(seed);
        var rng_b = std.Random.DefaultPrng.init(seed);
        var state_a = GameState.gameStart(rng_a.random());
        var state_b = GameState.gameStart(rng_b.random());
        for (0..40) |_| {
            var fresh: [tt_entry_count]TTEntry = undefined;
            for (&fresh) |*entry| entry.valid = false;
            const fresh_move = bestMoveTTWith(state_b, 3, &fresh);
            const warm_move = bestMoveTTWith(state_a, 3, warm);
            try std.testing.expectEqual(fresh_move, warm_move);
            const dir = warm_move orelse break;
            _ = state_a.move(dir, rng_a.random());
            _ = state_b.move(dir, rng_b.random());
        }
    }
}

test "TT slot collision drops the entry, never returns a wrong value (#88)" {
    var table: [tt_entry_count]TTEntry = undefined;
    for (&table) |*entry| entry.valid = false;
    // deterministic pair of distinct boards (valid power-of-two tiles) sharing one slot
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
            if (ttSlot(boards[i], 3, .max) == ttSlot(boards[j], 3, .max) and
                !std.meta.eql(boards[i], boards[j]))
            {
                pair = .{ boards[i], boards[j] };
                break :search;
            }
        }
    }
    try std.testing.expect(pair != null);
    const a_board = pair.?[0];
    const b_board = pair.?[1];
    ttPut(a_board, 3, .max, 42, &table);
    try std.testing.expectEqual(@as(?f64, 42), ttGet(a_board, 3, .max, &table));
    try std.testing.expectEqual(@as(?f64, null), ttGet(b_board, 3, .max, &table));
    ttPut(b_board, 3, .max, 7, &table);
    try std.testing.expectEqual(@as(?f64, 7), ttGet(b_board, 3, .max, &table));
    // board a's entry was dropped by the collision and must never be reported for either board
    try std.testing.expectEqual(@as(?f64, null), ttGet(a_board, 3, .max, &table));
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

test "order penalty follows the game phase at the 4096 tile (#92)" {
    const mid: Board = .{
        .{ 2, 2048, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const late: Board = .{
        .{ 2, 4096, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const w: Weights = .{ .empty = 10, .order_penalty = 30, .order_penalty_late = 300, .merge = 0 };
    // 14 empties x10; snake-order inversion 1->11 (mid) / 1->12 (late)
    try std.testing.expectEqual(@as(f64, 140 - 10 * 30), evaluateW(mid, w));
    try std.testing.expectEqual(@as(f64, 140 - 11 * 300), evaluateW(late, w));
}

test "baseline weights follow the adopted phase weights (#92)" {
    const mid: Board = .{
        .{ 2, 2048, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const late: Board = .{
        .{ 2, 4096, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    // adopted values (see docs/solver-history.md, issue #92): 43.174 before 4096, 25 from 4096
    try std.testing.expectApproxEqAbs(@as(f64, 140 - 10 * 43.174), evaluateW(mid, baseline), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 140 - 11 * 25), evaluateW(late, baseline), 1e-9);
    // ga weights stay phase-invariant at their evolved 43.174
    try std.testing.expectApproxEqAbs(@as(f64, 14 * 7.028 - 11 * 43.174), evaluateW(late, ga_weights), 1e-9);
}

test "fragility is inert at its default and without a death net (#101)" {
    const board: Board = .{
        .{ 2, 2048, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    var net: nn.Net = std.mem.zeroes(nn.Net);
    net.b3 = -0.4; // constant V for every board
    try std.testing.expectEqual(evaluateW(board, .{}), evaluateW(board, .{ .death_net = &net }));
    try std.testing.expectEqual(evaluateW(board, .{}), evaluateW(board, .{ .fragility = 500 }));
}

test "fragility adds the death net value scaled by the weight (#101)" {
    const board: Board = .{
        .{ 2, 2048, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    var net: nn.Net = std.mem.zeroes(nn.Net);
    net.b3 = -0.4; // constant V for every board
    const w: Weights = .{ .fragility = 500, .death_net = &net };
    // V = -0.4: fragility * (0 - V) = 500 * 0.4 = 200
    try std.testing.expectApproxEqAbs(evaluateW(board, .{}) + 200, evaluateW(board, w), 1e-9);
}

test "fragility penalizes boards the death net expects to die sooner (#101)" {
    // V = 0.1 * levelOf(board[0][0]) via one hidden unit fed by cell 0's level rows
    var net: nn.Net = std.mem.zeroes(nn.Net);
    for (0..nn.levels) |l| net.w1[l][0] = 0.1 * @as(f64, @floatFromInt(l));
    net.w2[0][0] = 1;
    net.w3[0] = 1;
    const fragile: Board = .{ // levelOf(32) = 5 -> V = 0.5 (die sooner)
        .{ 32, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const robust: Board = .{ // levelOf(2) = 1 -> V = 0.1
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const w: Weights = .{ .fragility = 100, .death_net = &net };
    // the fragile board takes the larger deduction: 100 * (0 - 0.5) = -50 vs -10
    try std.testing.expectApproxEqAbs(evaluateW(fragile, .{}) - 50, evaluateW(fragile, w), 1e-9);
    try std.testing.expectApproxEqAbs(evaluateW(robust, .{}) - 10, evaluateW(robust, w), 1e-9);
}

test "playGame finishes a game and reports its result" {
    var prng = std.Random.DefaultPrng.init(1);
    const result = playGame(prng.random(), 1);
    try std.testing.expect(result.moves > 0);
    try std.testing.expect(result.max_tile >= 2);
}

/// Golden fingerprint of a full game's move sequence and outcome (issue #87 regression guard).
const Golden = struct { score: u32, max_tile: core.Tile, moves: u32, checksum: u64 };

/// Play one full game with `policy` on `seed`, returning its outcome plus an FNV-1a checksum of
/// the move sequence so any change to the picked moves shows up in the result.
fn goldenPlay(policy: Policy, seed: u64, depth: u8) Golden {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var state = GameState.gameStart(rng);
    var checksum: u64 = 0xcbf29ce484222325;
    var moves: u32 = 0;
    while (choose(policy, state, rng, depth)) |d| {
        checksum = (checksum ^ @intFromEnum(d)) *% 0x100000001b3;
        _ = state.move(d, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| for (row) |value| {
        max_tile = @max(max_tile, value);
    };
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves, .checksum = checksum };
}

test "nn policy reproduces the same move sequence and score on a fixed seed" {
    const g = goldenPlay(.nn, 20240601, 0);
    try std.testing.expectEqual(@as(u32, 70364), g.score);
    try std.testing.expectEqual(@as(core.Tile, 4096), g.max_tile);
    try std.testing.expectEqual(@as(u32, 3176), g.moves);
    try std.testing.expectEqual(@as(u64, 16210708713020604363), g.checksum);
}

test "nn2 policy reproduces the same move sequence and score on a fixed seed" {
    const g = goldenPlay(.nn2, 20240601, 0);
    try std.testing.expectEqual(@as(u32, 73964), g.score);
    try std.testing.expectEqual(@as(core.Tile, 4096), g.max_tile);
    try std.testing.expectEqual(@as(u32, 3320), g.moves);
    try std.testing.expectEqual(@as(u64, 14931741200977583578), g.checksum);
}

test "nn policy parses the weights once per thread, not once per move" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var state = GameState.gameStart(rng);
    const before = nn.parse_count;
    var moves: usize = 0;
    while (moves < 32) : (moves += 1) {
        const d = choose(.nn, state, rng, 0) orelse break;
        _ = state.move(d, rng);
    }
    try std.testing.expectEqual(@as(usize, 32), moves);
    // the ~148 KB net is parsed at most once for the whole game, then reused by reference
    try std.testing.expect(nn.parse_count - before <= 1);
}

