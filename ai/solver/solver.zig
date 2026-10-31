const std = @import("std");
const core = @import("core");

const Board = core.Board;
const Direction = core.Direction;
const GameState = core.GameState;
const nn = @import("nn");
const ntuple = @import("ntuple");
const builtin = @import("builtin");

/// Returned for a board where the player has no legal move.
const lost_score: f64 = -1e9;
/// shortcut: fixed width keeps branching cheap; tune against deeper searches if needed.
const beam_width = 32;
const beam_horizon = 3;

/// Path shapes for the order penalty, compared by `zig build evalab` (issue #112).
/// `flex` / `flex_reverse` are the user-supplied "flex" layout (#112 comment): only 7 ranked cells
/// are ordered, the other 9 stay free.
pub const Layout = enum { snake, column_snake, spiral, block2, flex, flex_reverse };

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
    /// path shape of the order penalty (issue #112). The default is the shipped row snake.
    layout: Layout = .snake,
    /// anchor of that path: rotation (anchor >> 1, 90° clockwise each) plus a column mirror
    /// (anchor & 1) (issue #113). 0 keeps the shipped top-left anchor.
    anchor: u8 = 0,
    /// number of leading path cells that must stay ordered; the rest are free cells (issue #112,
    /// "flex" layouts). 16 keeps the whole path constrained, which is the shipped behaviour.
    ordered: u8 = 16,
    /// penalty for |log2| steps between adjacent tiles (issue #114). 0 keeps the shipped evaluation.
    smoothness: f64 = 0,
    /// penalty for the monotonicity violation of each row and column (issue #114).
    monotonic: f64 = 0,
    /// bonus for the sum of log2 tile values on the board (issue #114).
    tile_sum: f64 = 0,
    /// fragility penalty weight (issue #101): boards the death net expects to
    /// die sooner score lower by `fragility * (0 - v)`. 0 keeps evaluateW as-is.
    fragility: f64 = 0,
    /// death-variant net whose value is ≈ -(moves to game over)/reward_scale.
    /// Without it the fragility term is skipped.
    death_net: ?*const nn.Net = null,
};
/// Shipped weights. `monotonic = 32` was adopted in #117 (see docs/evalab.md).
pub const baseline: Weights = .{ .order_penalty = 43.174, .order_penalty_late = 25, .monotonic = 32 };

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
/// The order is the ranking's display order. The browser build picks its policies by name from
/// `wasm.policies` (#154), so reordering this enum no longer changes the WASM ABI.
pub const Policy = enum { random, greedy, expectimax, mcts, beam, expectimax_tt, ga, nn, nn2, nnmcts, nnbeam, expectimax_id, nn3, ntuple };

/// What every layer needs to know about one `Policy`; the name itself is `@tagName`. Keeping it in
/// one table is the point of #139: the ranking, the HTTP UI and the WASM build used to carry three
/// hand-written copies that drifted apart (a new variant broke `ranking.describe` in #138).
pub const PolicyInfo = struct {
    policy: Policy,
    /// Depth `zig build ranking` passes; 0 = the policy's own default (see `choose`).
    depth: u8,
    /// One line shown next to the ranking row and in the HTTP UI. Japanese, like the UI.
    description: []const u8,
    /// Run by `zig build ranking` without a filter. Off for the heavy / experimental variants.
    ranked: bool = true,
    /// Needs the 64 MiB n-tuple table file next to the binary, so the browser build cannot ship it
    /// (`wasm.policies` must not list it; `src/wasm.zig` fails the build if it does).
    needs_table: bool = false,
};

/// The single source of truth for the per-policy metadata, in `Policy` order.
pub const policy_info = [_]PolicyInfo{
    .{ .policy = .random, .depth = 0, .description = "合法手からランダムに1手選ぶ（比較の基準）" },
    .{ .policy = .greedy, .depth = 0, .description = "1手先だけ見る。滑らせた盤面の評価が最も高い手を選ぶ（出現や探索はしない）" },
    .{ .policy = .expectimax, .depth = 2, .description = "2手先まで探索し、ランダムな出現（2か4）を平均して評価する。評価は蛇行順と合体ペアの手作り関数" },
    .{ .policy = .mcts, .depth = 0, .description = "UCTで候補手を選び、ランダムな対局を繰り返して評価する" },
    .{ .policy = .beam, .depth = 0, .description = "幅32で3手先まで探索し、各空きマスに2が出現すると仮定して評価する" },
    .{ .policy = .expectimax_tt, .depth = 0, .description = "depth 3の期待値探索を置換表で高速化したもの（同じ手を選ぶ）" },
    .{ .policy = .ga, .depth = 2, .description = "GAで評価の重みを進化させた期待値探索（depth 2）。GA内のholdout(seed 500〜)では+4%だが、ranking seed 1000では素の expectimax より低い" },
    .{ .policy = .nn, .depth = 0, .description = "TD学習の小さなMLP価値関数。各手の後に1回出現を見込んで選ぶ（1-ply + 出現1回の先読み）" },
    .{ .policy = .nn2, .depth = 2, .description = "TD学習のMLP価値関数で2手先まで探索する（自分の手→出現→自分の手→出現の期待値、置換表で高速化）" },
    .{ .policy = .nnmcts, .depth = 0, .description = "UCTの各候補手から8手だけランダムに進め、打ち切った盤面をNN価値で評価する（終局は将来価値0として同じスケールで揃える）" },
    .{ .policy = .nnbeam, .depth = 0, .description = "NNの盤面価値を葉にしたビーム探索。幅8・3手先で、ply間のspawnを2回サンプリングする（固定seedで再現）" },
    .{ .policy = .expectimax_id, .depth = 0, .ranked = false, .description = "時間予算付きの反復深化 + 確率枝刈り（#104）。depth 2から1ずつ深くし、予算内に終わった最後の深さの手を選ぶ。予算はdepth引数（ms、0なら既定値）" },
    .{ .policy = .nn3, .depth = 0, .ranked = false, .description = "TD学習のMLP価値関数で3手先まで探索する（深い層の出現確率が1e-2未満の枝を落として再正規化）。既定のランキングには入れていない（重い）" },
    .{ .policy = .ntuple, .depth = 0, .ranked = false, .needs_table = true, .description = "n-tuple 表（量子化した int16、7 パターン）の1手先読み。表は 64 MiB あるので API と CLI 専用" },
};

// The table is indexed by `@intFromEnum`, so a missing, out-of-order or unexplained variant is a
// compile error rather than a runtime surprise.
comptime {
    if (policy_info.len != @typeInfo(Policy).@"enum".field_names.len)
        @compileError("policy_info must list every Policy variant");
    for (policy_info, 0..) |info, i| {
        if (@intFromEnum(info.policy) != i)
            @compileError("policy_info must be in Policy order (indexed by @intFromEnum)");
        if (info.description.len == 0)
            @compileError("policy " ++ @tagName(info.policy) ++ " has no description");
    }
}

/// Metadata for `policy`; total because `policy_info` covers every variant, in order.
pub fn policyInfo(policy: Policy) PolicyInfo {
    return policy_info[@intFromEnum(policy)];
}

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

/// Quantized n-tuple table used by `algo=ntuple`; 64.6 MiB, so it is loaded once per thread and
/// only when that policy is actually chosen (#129). Missing file -> the policy has no move.
const ntuple_weight_bytes = "ai/solver/ntuple_wide_q16.bin";
threadlocal var cached_ntuple: ?ntuple.Quantized = null;

/// Per-thread table, or null when `ntuple_weight_bytes` is not readable (e.g. in the WASM build).
/// The freestanding target has no filesystem, so its branch is a null-returning stub and the
/// `std.Io.Dir` path is never analyzed there.
const ntuple_load = if (builtin.target.os.tag == .freestanding) struct {
    fn get() ?*const ntuple.Quantized {
        return null;
    }
}.get else struct {
    fn get() ?*const ntuple.Quantized {
        if (cached_ntuple == null) {
            cached_ntuple = ntuple.loadQuantized(
                std.Io.Threaded.global_single_threaded.io(),
                std.heap.page_allocator,
                ntuple_weight_bytes,
            ) catch return null;
        }
        return &cached_ntuple.?;
    }
}.get;

fn ntupleTable() ?*const ntuple.Quantized {
    return ntuple_load();
}

/// True when `algo=ntuple` can answer on this target (the table file is readable).
pub fn ntupleAvailable() bool {
    return ntupleTable() != null;
}

/// Next move for a policy; null when no move changes the board.
pub fn choose(policy: Policy, state: GameState, rng: std.Random, depth: u8) ?Direction {
    return switch (policy) {
        inline else => |p| chooseOne(p, state, rng, depth),
    };
}

/// `choose` for one compile-time-known policy. Instantiating this per policy is what lets the
/// browser build link only the policies it ships (`wasm.policies`, #154): a runtime switch over
/// every variant (which is what `choose` is) drags all of them into the module.
pub fn chooseOne(comptime policy: Policy, state: GameState, rng: std.Random, depth: u8) ?Direction {
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
        // `depth` carries the time budget in ms for this policy (0 = the shipped `id_budget_ms`),
        // the way `depth` is per-policy elsewhere: bench/ranking/solve pass it through.
        .expectimax_id => return bestMoveID(state, if (depth == 0) id_budget_ms else depth),
        .ga => return bestMoveW(state, depth, ga_weights),
        // 1-ply greedy from the quantized n-tuple table (#129)
        .ntuple => return if (ntupleTable()) |q| ntuple.bestMoveQuantized(q, state) else null,
        // The net is parsed once per thread and reused by reference (issue #87), never copied per move.
        .nn, .nn2, .nn3, .nnmcts, .nnbeam => {
            const net = nnNet();
            return switch (policy) {
                .nn => nn.bestMove(net, state, .chance),
                // the 2-ply search itself dominates the cost.
                .nn2 => nn.bestMove(net, state, .deep),
                // 3-ply with the spawn cut (issue #108): a few times nn2's cost, so ranking leaves it out.
                .nn3 => nn.bestMove(net, state, .deep3),
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
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves };
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
    const empties = core.countEmpty(state.board);
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

/// `*_pruned` keep the probability-pruned values apart from the full-width ones: once branches are
/// cut the same board and depth is worth a different number, so the keys must not mix (issue #104).
const TTKind = enum { max, chance, max_pruned, chance_pruned };

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

fn freshTable() [tt_entry_count]TTEntry {
    var table: [tt_entry_count]TTEntry = undefined;
    for (&table) |*entry| entry.valid = false;
    return table;
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

/// shortcut: cut a branch when its probability along the path falls under this. 1e-4 keeps the
/// dropped mass under ~0.01% of a depth-4 tree (the test pins the value to the full-width one) and
/// reaches depth +1 in the same budget; raise it for speed, lower it for accuracy, 0 disables the cut.
const prob_cutoff: f64 = 1e-4;

/// Iterative deepening never searches shallower than this, so a zero budget still gets the depth-2 move.
const id_min_depth: u8 = 2;
/// shortcut: cap for one move's overrun, since the budget is only checked between depths (issue #104).
const id_max_depth: u8 = 8;
/// Time budget per move of `expectimax_id`.
/// shortcut: measured, not derived; a larger budget plays stronger and must still fit the 6 s/game limit.
const id_budget_ms: u32 = 1;

/// Deepest fully searched depth on this thread so far (0 before the first move). Only grows, so a
/// batch reports the deepest move of the whole batch, not just the last one.
pub threadlocal var id_depth_reached: u8 = 0;

/// Iterative-deepening expectimax: depth 2, then +1 while `budget_ms` lasts, returning the best move
/// of the last depth that ran to completion. The TT is keyed by depth, so all depths share it.
pub fn bestMoveID(state: GameState, budget_ms: u32) ?Direction {
    const table = ttWorkspace();
    const start = clockNow();
    var best = bestMoveTTPruned(state, id_min_depth, prob_cutoff, table);
    id_depth_reached = @max(id_depth_reached, id_min_depth);
    var depth: u8 = id_min_depth;
    while (best != null and shouldDeepen(depth, budget_ms, clock_available, elapsedMs(start))) {
        const move = bestMoveTTPruned(state, depth + 1, prob_cutoff, table);
        if (move == null) break;
        best = move;
        depth += 1;
        id_depth_reached = @max(id_depth_reached, depth);
    }
    return best;
}

/// Whether iterative deepening may search one more depth. `clock_available` is false where
/// `clockNow` is the freestanding stub (it returns a fixed timestamp), and a timed loop there could
/// never expire: with `elapsed_ms` pinned at 0 the search ran to `id_max_depth` and took minutes per
/// move in the browser (issue #145). Such a target keeps the depth it already has.
pub fn shouldDeepen(depth: u8, budget_ms: u32, clock_available_: bool, elapsed_ms: i96) bool {
    return depth < id_max_depth and clock_available_ and elapsed_ms < @as(i96, budget_ms);
}

/// Zig 0.17 has no `std.time.Instant`; the monotonic clock lives behind `std.Io`. The global
/// single-threaded instance's `now` only reads the OS clock and touches no shared state.
/// Freestanding (WASM) has no clock: it returns a fixed timestamp, so a timed search there
/// degenerates to its minimum depth instead of failing to build.
const clock_available = builtin.target.os.tag != .freestanding;
const clockNow = if (clock_available) struct {
    fn now() std.Io.Timestamp {
        return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io());
    }
}.now else struct {
    fn now() std.Io.Timestamp {
        return .{ .nanoseconds = 0 };
    }
}.now;

fn elapsedMs(start: std.Io.Timestamp) i96 {
    return @divTrunc(start.durationTo(clockNow()).nanoseconds, std.time.ns_per_ms);
}

fn bestMoveTTPruned(state: GameState, depth: u8, cutoff: f64, table: *[tt_entry_count]TTEntry) ?Direction {
    var best: ?Direction = null;
    var best_value = lost_score;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        const value = chanceValueTTPruned(next, depth, 1, cutoff, table);
        if (best == null or value > best_value) {
            best = d;
            best_value = value;
        }
    }
    return best;
}

/// `maxValueTT` of the pruned search. The TT key leaves the path probability out, so the first
/// board+depth pair to be stored wins (the approximation nneonneo's table makes); `.max_pruned` keeps
/// these values from ever mixing with the full-width `.max` entries.
fn maxValueTTPruned(state: GameState, depth: u8, cutoff: f64, table: *[tt_entry_count]TTEntry) f64 {
    if (depth == 0) return evaluateState(state);
    if (ttGet(state.board, depth, .max_pruned, table)) |value| return value;
    var best: f64 = lost_score;
    var any_move = false;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        if (!next.slide(d).changed) continue;
        any_move = true;
        best = @max(best, chanceValueTTPruned(next, depth, 1, cutoff, table));
    }
    if (!any_move) best = lost_score;
    ttPut(state.board, depth, .max_pruned, best, table);
    return best;
}

/// `chanceValueTT` with the probability cut: branches below `cutoff` are dropped and the kept ones
/// are renormalized by their summed probability, so the value keeps the scale of a full average.
/// The last chance node (depth 1) keeps every branch, because cutting next to the leaves bends the value.
fn chanceValueTTPruned(state: GameState, depth: u8, cprob: f64, cutoff: f64, table: *[tt_entry_count]TTEntry) f64 {
    if (ttGet(state.board, depth, .chance_pruned, table)) |value| return value;
    var total: f64 = 0;
    var mass: f64 = 0;
    var empties: usize = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (state.board[r][c] == 0) empties += 1;
        }
    }
    if (empties == 0) {
        const value = evaluateState(state);
        ttPut(state.board, depth, .chance_pruned, value, table);
        return value;
    }
    const cell_prob = cprob / @as(f64, @floatFromInt(empties));
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (state.board[r][c] != 0) continue;
            for ([_]struct { value: core.Tile, p: f64 }{
                .{ .value = 2, .p = 0.9 },
                .{ .value = 4, .p = 0.1 },
            }) |spawn| {
                const p = cell_prob * spawn.p;
                if (depth > 1 and p < cutoff) continue;
                var next = state;
                next.board[r][c] = spawn.value;
                total += p * maxValueTTPruned(next, depth - 1, cutoff, table);
                mass += p;
            }
        }
    }
    // every branch cut (the node itself is already this unlikely): fall back to the leaf rule
    const value = if (mass == 0) evaluateState(state) else total / mass;
    ttPut(state.board, depth, .chance_pruned, value, table);
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
    var path: [core.BoardCellCount]core.Position = undefined;
    pathFor(w.layout, w.anchor, &path);
    var empties: f64 = 0;
    var penalty: f64 = 0;
    var prev: ?f64 = null;
    var max_level: f64 = 0;
    for (path, 0..) |p, i| {
        const value = board[p.row][p.col];
        if (value == 0) {
            empties += 1;
            continue;
        }
        const level: f64 = @floatFromInt(@ctz(value));
        if (i < w.ordered) {
            if (prev) |before| penalty += @max(0, level - before);
            prev = level;
        }
        max_level = @max(max_level, level);
    }
    // endgame phase once the largest tile reaches 4096 = 2^12 (issue #92)
    const order_penalty = if (max_level >= 12) w.order_penalty_late else w.order_penalty;
    var base = empties * w.empty - penalty * order_penalty + mergePairs(board) * w.merge;
    if (w.smoothness != 0) base -= rough(board) * w.smoothness;
    if (w.monotonic != 0) base -= monoViolation(board) * w.monotonic;
    if (w.tile_sum != 0) base += tileSum(board) * w.tile_sum;
    if (w.fragility == 0) return base;
    const net = w.death_net orelse return base;
    var cache: nn.Cache = .{};
    return base + w.fragility * (0 - nn.forward(net, board, &cache));
}

/// Traversal order for the order penalty: `layout` shape, rotated clockwise `anchor >> 1` times and
/// mirrored along the columns when `anchor & 1` is set. anchor 0 / `.snake` is the shipped path.
pub fn pathFor(layout: Layout, anchor: u8, out: *[core.BoardCellCount]core.Position) void {
    const rot = anchor >> 1;
    for (0..core.BoardCellCount) |i| {
        var p = baseCell(layout, i);
        // 90° clockwise: (row, col) -> (col, n-1-row). The row is saved first: `p = .{...p...}`
        // would read the already-written field under result-location semantics.
        for (0..rot) |_| {
            const row = p.col;
            p.col = core.BoardSize - 1 - p.row;
            p.row = row;
        }
        if (anchor & 1 == 1) p.col = core.BoardSize - 1 - p.col;
        out[i] = p;
    }
}

/// cell `i` of the unrotated path for `layout`.
fn baseCell(layout: Layout, i: usize) core.Position {
    const n = core.BoardSize;
    return switch (layout) {
        .snake => order[i],
        // columns instead of rows; the largest tile ends up top-right of the anchor column pair
        .column_snake => .{ .row = if ((i / n) % 2 == 0) i % n else n - 1 - i % n, .col = i / n },
        .spiral => spiral[i],
        .flex => flex[i],
        .flex_reverse => flex[core.BoardCellCount - 1 - i],
        // 2x2 blocks in row-major order, each block walked as a small snake
        .block2 => blk: {
            const blocks_per_row = n / 2;
            const origin_row = (i / 4 / blocks_per_row) * 2;
            const origin_col = (i / 4 % blocks_per_row) * 2;
            const step = block2_step[i % 4];
            break :blk .{ .row = origin_row + step.row, .col = origin_col + step.col };
        },
    };
}

/// flex layout: the 7 ranked cells in size order (1 = largest), then the 9 free cells.
/// rank 1..7 sit at (1,3) (2,2) (2,3) (2,0) (3,2) (3,0) (3,3); empties are skipped by the penalty,
/// so their order is irrelevant.
const flex: [core.BoardCellCount]core.Position = .{
    .{ .row = 1, .col = 3 }, .{ .row = 2, .col = 2 }, .{ .row = 2, .col = 3 }, .{ .row = 2, .col = 0 },
    .{ .row = 3, .col = 2 }, .{ .row = 3, .col = 0 }, .{ .row = 3, .col = 3 },
    .{ .row = 0, .col = 0 }, .{ .row = 0, .col = 1 }, .{ .row = 0, .col = 2 }, .{ .row = 0, .col = 3 },
    .{ .row = 1, .col = 0 }, .{ .row = 1, .col = 1 }, .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 1 },
    .{ .row = 3, .col = 1 },
};

/// Cells of one 2x2 block in walk order: top-left, top-right, bottom-right, bottom-left.
const block2_step: [4]core.Position = .{
    .{ .row = 0, .col = 0 }, .{ .row = 0, .col = 1 }, .{ .row = 1, .col = 1 }, .{ .row = 1, .col = 0 },
};

/// Outer ring clockwise from the top-left, then the inner 2x2 clockwise.
const spiral: [core.BoardCellCount]core.Position = .{
    .{ .row = 0, .col = 0 }, .{ .row = 0, .col = 1 }, .{ .row = 0, .col = 2 }, .{ .row = 0, .col = 3 },
    .{ .row = 1, .col = 3 }, .{ .row = 2, .col = 3 }, .{ .row = 3, .col = 3 }, .{ .row = 3, .col = 2 },
    .{ .row = 3, .col = 1 }, .{ .row = 3, .col = 0 }, .{ .row = 2, .col = 0 }, .{ .row = 1, .col = 0 },
    .{ .row = 1, .col = 1 }, .{ .row = 1, .col = 2 }, .{ .row = 2, .col = 2 }, .{ .row = 2, .col = 1 },
};

/// log2 of a tile; empties count as level 0.
fn levelOf(value: core.Tile) f64 {
    return if (value == 0) 0 else @floatFromInt(@ctz(value));
}

/// Sum of |log2 difference| between non-empty neighbours: a rough board scores higher (worse).
fn rough(board: Board) f64 {
    var total: f64 = 0;
    for (0..core.BoardSize) |r| {
        for (0..core.BoardSize) |c| {
            if (board[r][c] == 0) continue;
            if (c + 1 < core.BoardSize and board[r][c + 1] != 0) total += @abs(levelOf(board[r][c]) - levelOf(board[r][c + 1]));
            if (r + 1 < core.BoardSize and board[r + 1][c] != 0) total += @abs(levelOf(board[r][c]) - levelOf(board[r + 1][c]));
        }
    }
    return total;
}

/// Monotonicity violation over every row and column, counted in the line's better direction.
fn monoViolation(board: Board) f64 {
    var total: f64 = 0;
    for (0..core.BoardSize) |i| {
        total += lineViolation(.{ board[i][0], board[i][1], board[i][2], board[i][3] });
        total += lineViolation(.{ board[0][i], board[1][i], board[2][i], board[3][i] });
    }
    return total;
}

/// steps against the better of the two directions: 0 when the line is monotone either way.
fn lineViolation(tiles: [4]core.Tile) f64 {
    var rising: f64 = 0;
    var falling: f64 = 0;
    for (0..3) |i| {
        const diff = levelOf(tiles[i + 1]) - levelOf(tiles[i]);
        if (diff > 0) rising += diff else falling -= diff;
    }
    return @min(rising, falling);
}

/// Sum of log2 tile values: rewards a board of many large tiles.
fn tileSum(board: Board) f64 {
    var total: f64 = 0;
    for (board) |row| {
        for (row) |value| total += levelOf(value);
    }
    return total;
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

/// Play one full game with custom evaluation weights (`zig build evalab`).
pub fn playWeights(rng: std.Random, w: Weights, depth: u8) GameResult {
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    while (bestMoveW(state, depth, w)) |d| {
        _ = state.move(d, rng);
        moves += 1;
    }
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves };
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

test "probability cut drops the 4-spawn and renormalizes the kept branches" {
    // two empty cells: each 2-spawn carries p=0.45 and each 4-spawn p=0.05, so a cutoff of 0.06
    // keeps only the two 2-spawns and renormalizes them to a plain average
    const state: GameState = .{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 512, 1024, 2048, 4096 },
        .{ 8192, 16384, 0, 0 },
    } };
    var pruned_table = freshTable();
    var full_table = freshTable();
    var first = state;
    first.board[3][2] = 2;
    var second = state;
    second.board[3][3] = 2;
    const want = (maxValueTT(first, 1, &full_table) + maxValueTT(second, 1, &full_table)) / 2;
    try std.testing.expectApproxEqRel(want, chanceValueTTPruned(state, 2, 1, 0.06, &pruned_table), 1e-9);
}

test "probability cut keeps the value within 5% of the full-width search" {
    const state: GameState = .{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 512, 1024, 2048, 4096 },
        .{ 8192, 16384, 0, 0 },
    } };
    var pruned_table = freshTable();
    var full_table = freshTable();
    const full = chanceValueTT(state, 4, &full_table);
    const pruned = chanceValueTTPruned(state, 4, 1, prob_cutoff, &pruned_table);
    try std.testing.expectApproxEqRel(full, pruned, 0.05);
}

test "iterative deepening returns a legal move on random boards" {
    var prng = std.Random.DefaultPrng.init(41);
    const rng = prng.random();
    var state = GameState.gameStart(rng);
    for (0..20) |_| {
        if (bestMoveID(state, 1)) |d| {
            var next = state;
            try std.testing.expect(next.slide(d).changed);
        } else {
            try std.testing.expect(state.isGameOver());
        }
        const direction = choose(.random, state, rng, 0) orelse break;
        _ = state.move(direction, rng);
    }
}

test "iterative deepening with a zero budget returns the depth 2 move" {
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    var state = GameState.gameStart(rng);
    for (0..8) |_| {
        const direction = choose(.random, state, rng, 0) orelse break;
        _ = state.move(direction, rng);
    }
    var table = freshTable();
    try std.testing.expectEqual(bestMoveTTPruned(state, id_min_depth, prob_cutoff, &table), bestMoveID(state, 0));
    try std.testing.expectEqual(id_min_depth, id_depth_reached);
}

// The freestanding clock is a fixed timestamp, so `elapsed_ms` never advances there: a timed loop
// would run to `id_max_depth` and take minutes per move in the browser (issue #145). The loop
// condition is a pure function so this target can pin the frozen-clock case directly.
test "iterative deepening never deepens without an advancing clock" {
    try std.testing.expect(!shouldDeepen(id_min_depth, 1, false, 0)); // frozen clock (WASM)
    try std.testing.expect(!shouldDeepen(id_min_depth, 10_000, false, 0));
    try std.testing.expect(shouldDeepen(id_min_depth, 1, true, 0)); // live clock, budget left
    try std.testing.expect(!shouldDeepen(id_min_depth, 1, true, 1)); // budget spent
    try std.testing.expect(!shouldDeepen(id_max_depth, 1, true, 0)); // already at the cap
}

// The order is the ranking table's order and the `algo=` values of the HTTP API, so it is pinned
// here — a rename or a reorder silently changes both (#139). The browser build no longer depends on
// it: it names its policies in `wasm.policies` (#154).
test "Policy order is the ranking order" {
    const abi = [_][]const u8{
        "random",        "greedy",        "expectimax", "mcts",   "beam",
        "expectimax_tt", "ga",            "nn",         "nn2",    "nnmcts",
        "nnbeam",        "expectimax_id", "nn3",        "ntuple",
    };
    try std.testing.expectEqual(abi.len, std.enums.values(Policy).len);
    for (std.enums.values(Policy), 0..) |p, i| try std.testing.expectEqualStrings(abi[i], @tagName(p));
}

test "choose and chooseOne agree" {
    const state: GameState = .{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 2, 0, 4 },
        .{ 0, 0, 8, 0 },
        .{ 0, 0, 0, 2 },
    } };
    inline for ([_]Policy{ .random, .greedy }) |policy| { // inline: `chooseOne` takes the policy comptime
        var a = std.Random.DefaultPrng.init(31);
        var b = std.Random.DefaultPrng.init(31);
        try std.testing.expectEqual(
            choose(policy, state, a.random(), 2),
            chooseOne(policy, state, b.random(), 2),
        );
    }
}

test "policy_info covers every variant and the ranking depth of each" {
    for (std.enums.values(Policy)) |p| {
        const info = policyInfo(p);
        try std.testing.expectEqual(p, info.policy);
        try std.testing.expect(info.description.len > 0);
    }
    // only ntuple carries a data-file requirement, so it is the only policy the browser cannot ship
    try std.testing.expectEqual(Policy.ntuple, std.enums.values(Policy)[policy_info.len - 1]);
    for (policy_info) |info| try std.testing.expectEqual(info.policy == .ntuple, info.needs_table);
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
    // since monotonic (#117) greedy and expectimax disagree here; compare like with like
    try std.testing.expectEqual(bestMove(state, 2), rank(state, .expectimax, 2).items[0].dir);
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

test "pathFor snake anchor 0 is the shipped order" {
    var path: [core.BoardCellCount]core.Position = undefined;
    pathFor(.snake, 0, &path);
    try std.testing.expectEqual(order, path);
}

test "every layout and anchor visits each cell exactly once" {
    var path: [core.BoardCellCount]core.Position = undefined;
    for (std.enums.values(Layout)) |layout| {
        for (0..8) |anchor| {
            pathFor(layout, @intCast(anchor), &path);
            var seen: u16 = 0;
            for (path) |p| {
                const bit: u4 = @intCast(p.row * core.BoardSize + p.col);
                try std.testing.expect(seen & (@as(u16, 1) << bit) == 0);
                seen |= @as(u16, 1) << bit;
            }
            try std.testing.expectEqual(@as(u16, 0xffff), seen);
        }
    }
}

test "anchor rotates clockwise then mirrors the columns" {
    var path: [core.BoardCellCount]core.Position = undefined;
    // snake path index 1 is (0, 1)
    const cases = [_]struct { anchor: u8, cell: core.Position }{
        .{ .anchor = 0, .cell = .{ .row = 0, .col = 1 } },
        .{ .anchor = 1, .cell = .{ .row = 0, .col = 2 } }, // mirror
        .{ .anchor = 2, .cell = .{ .row = 1, .col = 3 } }, // 90° clockwise
        .{ .anchor = 3, .cell = .{ .row = 1, .col = 0 } }, // rotate, then mirror
        .{ .anchor = 6, .cell = .{ .row = 2, .col = 0 } }, // 270° clockwise
    };
    for (cases) |c| {
        pathFor(.snake, c.anchor, &path);
        try std.testing.expectEqual(c.cell, path[1]);
    }
}

test "layouts give different paths" {
    var snake: [core.BoardCellCount]core.Position = undefined;
    var column: [core.BoardCellCount]core.Position = undefined;
    var spiral_path: [core.BoardCellCount]core.Position = undefined;
    var block: [core.BoardCellCount]core.Position = undefined;
    pathFor(.snake, 0, &snake);
    pathFor(.column_snake, 0, &column);
    pathFor(.spiral, 0, &spiral_path);
    pathFor(.block2, 0, &block);
    // column snake starts down the first column, spiral walks the outer ring, block2 its first 2x2
    try std.testing.expectEqual(core.Position{ .row = 1, .col = 0 }, column[1]);
    try std.testing.expectEqual(core.Position{ .row = 1, .col = 3 }, spiral_path[4]);
    try std.testing.expectEqual(core.Position{ .row = 1, .col = 1 }, block[2]);
    try std.testing.expectEqual(core.Position{ .row = 2, .col = 2 }, block[12]);
    try std.testing.expect(snake[1].col != column[1].col);
}

test "ordered cut keeps only the leading path cells constrained" {
    // (3,0) and (2,1) are path cells 8 and 13 of the snake, so a cut at 8 ignores their rise
    const board: Board = .{
        .{ 0, 0, 0, 2 },
        .{ 0, 0, 0, 0 },
        .{ 0, 8, 0, 0 },
        .{ 16, 0, 0, 0 },
    };
    var cut = baseline;
    cut.ordered = 8;
    try std.testing.expect(evaluateW(board, cut) > evaluateW(board, baseline));
    var full = baseline;
    full.ordered = 16;
    try std.testing.expectEqual(evaluateW(board, baseline), evaluateW(board, full));
}

test "zero extra weights keep the shipped evaluation" {
    const board: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    var w = baseline;
    w.smoothness = 0;
    w.monotonic = 0;
    w.tile_sum = 0;
    try std.testing.expectEqual(evaluate(board), evaluateW(board, w));
}

test "extra terms change the score as documented" {
    const sorted: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const jumbled: Board = .{
        .{ 2, 16, 4, 8 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    // three level steps along the row, and the levels sum to 1 + 2 + 3 + 4
    try std.testing.expectEqual(@as(f64, 3), rough(sorted));
    try std.testing.expectEqual(@as(f64, 10), tileSum(sorted));
    try std.testing.expectEqual(@as(f64, 0), monoViolation(sorted));
    try std.testing.expect(monoViolation(jumbled) > 0);
    // penalties lower the score, the tile sum raises it (same weights apart from the added term)
    var plain = baseline;
    plain.monotonic = 0;
    var smooth = plain;
    smooth.smoothness = 5;
    var mono = plain;
    mono.monotonic = 5;
    var tiles = plain;
    tiles.tile_sum = 1;
    try std.testing.expect(evaluateW(jumbled, smooth) < evaluateW(jumbled, plain));
    try std.testing.expect(evaluateW(jumbled, mono) < evaluateW(jumbled, plain));
    try std.testing.expect(evaluateW(sorted, tiles) > evaluateW(sorted, plain));
}

test "shipped baseline penalizes monotonicity violations (#117)" {
    const monotone: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    const broken: Board = .{
        .{ 2, 16, 4, 8 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };
    var off = baseline;
    off.monotonic = 0;
    const shipped_gap = evaluateW(monotone, baseline) - evaluateW(broken, baseline);
    const off_gap = evaluateW(monotone, off) - evaluateW(broken, off);
    try std.testing.expect(shipped_gap > off_gap);
}

test "playWeights with the baseline matches the expectimax policy" {
    var prng_a = std.Random.DefaultPrng.init(11);
    var prng_b = std.Random.DefaultPrng.init(11);
    const custom = playWeights(prng_a.random(), baseline, 1);
    const shipped = playPolicy(prng_b.random(), .expectimax, 1);
    try std.testing.expectEqual(shipped.score, custom.score);
    try std.testing.expectEqual(shipped.moves, custom.moves);
    try std.testing.expectEqual(shipped.max_tile, custom.max_tile);
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
    // adopted values (see docs/solver-history.md, issue #92): 43.174 before 4096, 25 from 4096;
    // the monotonic term adds 10 / 11 violation steps x32 since #117
    try std.testing.expectApproxEqAbs(@as(f64, 140 - 10 * 43.174 - 10 * 32), evaluateW(mid, baseline), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 140 - 11 * 25 - 11 * 32), evaluateW(late, baseline), 1e-9);
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
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .checksum = checksum };
}

test "nn policy reproduces the same move sequence and score on a fixed seed" {
    const g = goldenPlay(.nn, 20240601, 0);
    try std.testing.expectEqual(@as(u32, 32380), g.score);
    try std.testing.expectEqual(@as(core.Tile, 2048), g.max_tile);
    try std.testing.expectEqual(@as(u32, 1669), g.moves);
    try std.testing.expectEqual(@as(u64, 3421224406919827547), g.checksum);
}

test "nn2 policy reproduces the same move sequence and score on a fixed seed" {
    const g = goldenPlay(.nn2, 20240601, 0);
    try std.testing.expectEqual(@as(u32, 77164), g.score);
    try std.testing.expectEqual(@as(core.Tile, 4096), g.max_tile);
    try std.testing.expectEqual(@as(u32, 3542), g.moves);
    try std.testing.expectEqual(@as(u64, 3580527611182212019), g.checksum);
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

test "ntuple policy plays a legal move from the quantized table (#129)" {
    // The table file ships with the repo; a checkout without it simply skips the check.
    if (!ntupleAvailable()) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(1);
    var state = GameState.gameStart(prng.random());
    const move = choose(.ntuple, state, prng.random(), 0) orelse return error.TestUnexpectedResult;
    try std.testing.expect(state.slide(move).changed);
}
