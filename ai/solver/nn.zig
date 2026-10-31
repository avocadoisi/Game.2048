const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const parallel = @import("parallel");

const Board = core.Board;
const Direction = core.Direction;
const GameState = core.GameState;

/// Cell encoding: log2 tile level (0 = empty), one-hot per cell. Tiles >= 2^15 share the top level.
pub const levels = 16;
pub const input_count = core.BoardCellCount * levels;
pub const h1_size = 64;
pub const h2_size = 32;
/// Score deltas are divided by this before they enter the value target, so values stay near unit scale.
pub const reward_scale: f64 = 1000;

/// Plain f64 weights. `extern` keeps the layout fixed, so the file is the raw struct bytes.
pub const Net = extern struct {
    /// First layer, one row per (cell, level) pair. Only 16 rows are active for any board.
    w1: [input_count][h1_size]f64,
    b1: [h1_size]f64,
    w2: [h2_size][h1_size]f64,
    b2: [h2_size]f64,
    w3: [h2_size]f64,
    b3: f64,

    pub fn init(rng: std.Random) Net {
        var net = std.mem.zeroes(Net);
        for (&net.w1) |*row| for (row) |*w| {
            w.* = uniform(rng, 1.0 / 4.0);
        };
        for (&net.w2) |*row| for (row) |*w| {
            w.* = uniform(rng, 1.0 / 8.0);
        };
        for (&net.w3) |*w| w.* = uniform(rng, 1.0 / @sqrt(@as(f64, h2_size)));
        return net;
    }
};

fn uniform(rng: std.Random, scale: f64) f64 {
    return (rng.float(f64) * 2 - 1) * scale;
}

/// Optimistic initialization (#107): multiply every weight by `k`, so the initial value surface has
/// `k` times more spread and the self-play policy explores more early on. Only the spread can matter:
/// a constant shift of `b3` adds the same amount to every afterstate, so the argmax over moves is
/// unchanged and such an "optimistic" bias is a no-op for the data the learner sees.
/// `shortcut: one scalar for all layers, per-layer scales if this shows an effect.`
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
};

pub fn levelOf(value: core.Tile) usize {
    if (value == 0) return 0;
    return @min(levels - 1, std.math.log2_int(core.Tile, value));
}

/// Afterstate value estimate (scalar, unit-scale). Fills `cache` for a following `train`.
pub fn forward(net: *const Net, board: Board, cache: *Cache) f64 {
    for (0..core.BoardCellCount) |cell| {
        const value = board[cell / core.BoardSize][cell % core.BoardSize];
        cache.rows[cell] = cell * levels + levelOf(value);
    }
    cache.z1 = net.b1;
    for (cache.rows) |row| {
        for (0..h1_size) |i| cache.z1[i] += net.w1[row][i];
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
    const d = cache.out - target;
    var dz2: [h2_size]f64 = undefined;
    var da1 = std.mem.zeroes([h1_size]f64);
    for (0..h2_size) |j| {
        dz2[j] = if (cache.z2[j] > 0) d * net.w3[j] else 0;
        for (0..h1_size) |i| da1[i] += net.w2[j][i] * dz2[j];
    }

    net.b3 -= lr * d;
    for (0..h2_size) |j| {
        net.w3[j] -= lr * d * cache.a2[j];
        net.b2[j] -= lr * dz2[j];
        for (0..h1_size) |i| net.w2[j][i] -= lr * dz2[j] * cache.a1[i];
    }

    for (0..h1_size) |i| {
        const dz1 = if (cache.z1[i] > 0) da1[i] else 0;
        net.b1[i] -= lr * dz1;
        for (cache.rows) |row| net.w1[row][i] -= lr * dz1;
    }
}

/// MS-TD (#107): one value net per afterstate stage, staged by the number of empty cells.
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

/// Value function used by the search: one net for every board, or one net per empty-cell stage (#107).
pub const Value = union(enum) { single: *const Net, staged: *const Staged };

/// `forward` for a `Value`; `cache` belongs to the net being evaluated and is unused for staged values.
pub fn valueAt(value: Value, board: Board, cache: *Cache) f64 {
    return switch (value) {
        .single => |net| forward(net, board, cache),
        .staged => |staged| staged.forward(board),
    };
}

// Container-level aliases: the `Staged` methods below shadow `forward` / `train` by name.
const netForward = forward;
const netTrain = train;

/// Multi-stage TD value function (#107): `stage_count` nets of the `Net` architecture, chosen by the
/// afterstate's empty-cell count. Search and training reach it through `Value`.
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
        @memcpy(bytes[0..8], staged_file_magic);
        std.mem.writeInt(u32, bytes[8..12], file_version, .little);
        std.mem.writeInt(u32, bytes[12..16], endian_marker, .little);
        for (0..stage_count) |k| writeWeights(bytes[header_len + k * @sizeOf(Net) ..][0..@sizeOf(Net)], &self.nets[k]);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    pub fn loadFile(self: *Staged, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(staged_file_len + 1));
        defer gpa.free(bytes);
        try self.parseStaged(bytes);
    }

    fn parseStaged(self: *Staged, bytes: []const u8) !void {
        if (bytes.len != staged_file_len) return error.WrongWeightFileSize;
        if (!std.mem.eql(u8, bytes[0..8], staged_file_magic)) return error.BadWeightMagic;
        if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedWeightVersion;
        if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongWeightEndianness;
        for (0..stage_count) |k| self.nets[k] = readWeights(bytes[header_len + k * @sizeOf(Net) ..][0..@sizeOf(Net)]);
    }
};

/// Dihedral symmetry (issue #72): 8 cell-index permutations, rotations 0..3 then their mirrors 4..7.
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

/// Best legal move by reward + V(afterstate); null when no move changes the board.
fn pickMove(net: *const Net, state: GameState, death: bool) ?Pick {
    return pickMoveValue(.{ .single = net }, state, death);
}

/// `pickMove` for a single or staged value function (#107).
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

/// Search statistics for one move (issue #69): afterstate evaluations and value-cache hits.
pub const Stats = struct { leaves: u64 = 0, hits: u64 = 0 };

/// Depth one search runs at: `ply` own moves deep, dropping spawn branches under path probability
/// `cutoff` (`0` keeps every branch). Part of the value-cache key (#157), because a phase game asks
/// for the same board at two depths.
pub const Depth = struct { ply: u8 = 0, cutoff: f64 = 0 };

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

    fn keyOf(board: Board) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (board) |row| for (row) |value| {
            h = (h ^ @as(u64, value)) *% 0x100000001b3;
        };
        return h;
    }

    /// `keyOf` mixed with the search depth (#157). The stored value is the depth-free leaf value
    /// today, but a phase game asks for one board at 2-ply and at 3-ply: without the depth in the key
    /// one entry would stand in for the other, and any depth-dependent leaf or backup would then be
    /// silently wrong. Both fields are mixed here, so the table keeps one key space.
    fn keyOfDepth(board: Board, depth: Depth) u64 {
        var h = keyOf(board);
        h = (h ^ @as(u64, depth.ply)) *% 0x100000001b3;
        return (h ^ @as(u64, @bitCast(depth.cutoff))) *% 0x100000001b3;
    }

    /// `forward(net, board)` with memoization; `stats` counts leaf evaluations and hits.
    pub fn get(self: *ValueCache, net: *const Net, board: Board, stats: *Stats) f64 {
        return self.getValue(.{ .single = net }, board, stats);
    }

    /// `get` for a single or staged value function (#107).
    pub fn getValue(self: *ValueCache, vfn: Value, board: Board, stats: *Stats) f64 {
        return self.getValueAtDepth(vfn, board, leaf_depth, stats);
    }

    /// `getValue` for one search depth: the same board at another (ply, cutoff) is another entry.
    pub fn getValueAtDepth(self: *ValueCache, vfn: Value, board: Board, depth: Depth, stats: *Stats) f64 {
        const key = keyOfDepth(board, depth);
        if (key == empty_key) return leafValue(vfn, board, stats);
        const slot: usize = @intCast(key & (capacity - 1));
        if (self.entries[slot].key == key) {
            stats.hits += 1;
            return self.entries[slot].value;
        }
        const value_ = leafValue(vfn, board, stats);
        self.entries[slot] = .{ .key = key, .value = value_ };
        return value_;
    }
};

fn leafValue(vfn: Value, board: Board, stats: *Stats) f64 {
    var cache: Cache = .{};
    stats.leaves += 1;
    return valueAt(vfn, board, &cache);
}

/// Per-thread cache reused across moves (issue #88). A value is a pure function of the board and
/// the net, so a warm cache can only change speed, never the chosen move; there is no shared
/// mutable state because each thread keeps its own workspace.
threadlocal var move_cache: ValueCache = .{};

/// The calling thread's reusable cache, e.g. for policies that keep their own search loops.
pub fn sharedValueCache() *ValueCache {
    return &move_cache;
}

/// shortcut: branch probability cut for `deep3` (issue #108), the same idea as `solver.zig`'s
/// `prob_cutoff` at a coarser scale. A 4 under a 4 is already ~1e-3 of the path at the second spawn
/// layer, so 1e-2 drops those subtrees (about 10% of a node's mass, renormalized away) while the
/// last spawn layer next to the leaves stays full width. Raise it for speed, lower it for accuracy,
/// 0 disables the cut (that is `.deep`).
const deep3_cutoff: f64 = 1e-2;

/// Depth-limited expectimax: our moves and every spawn (2 at 0.9, 4 at 0.1), no RNG.
/// The caller owns the value cache so a game can reuse one workspace for all of its moves.
const Search = struct {
    vfn: Value,
    cache: *ValueCache,
    stats: Stats = .{},
    /// Spawn branches under this path probability are dropped; 0 searches every branch (`.deep`).
    cutoff: f64 = 0,
    /// Own-move plies this whole search runs at; with `cutoff` it keys every leaf lookup (#157).
    ply: u8 = 0,

    /// `cache.getValue` at this search's depth: one entry per (board, ply, cutoff).
    fn leaf(self: *Search, board: Board) f64 {
        return self.cache.getValueAtDepth(self.vfn, board, .{ .ply = self.ply, .cutoff = self.cutoff }, &self.stats);
    }

    /// `value` with the leaf one own move further out and no reward at it: `bootValue(S, 0)` is the
    /// best afterstate value one move from `S`, and every level above adds one reward plus the spawn
    /// average below it, `bootValue(S, r) = max_d (score_delta / reward_scale +
    /// E_spawn[bootValue(S', r - 1)])`. So `bootValue(S, r)` is `r` rewards and the value of the
    /// afterstate `r + 1` moves ahead: exactly one move past the last reward it summed. A leaf taken
    /// at the last reward's afterstate (what `value` does) would count that reward twice (#158).
    fn bootValue(self: *Search, state: GameState, rewards: u8, cprob: f64) f64 {
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
    fn value(self: *Search, state: GameState, depth: u8, cprob: f64) f64 {
        var best: f64 = -std.math.inf(f64);
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            var next = state;
            const moved = next.slide(d);
            if (!moved.changed) continue;
            const v = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + self.chance(next, depth - 1, cprob);
            best = @max(best, v);
        }
        return if (best == -std.math.inf(f64)) 0 else best;
    }

    /// Expected value of a post-move board after spawning; depth 0 is the value leaf (the afterstate itself).
    /// `cprob` is the probability of reaching this node. Kept branches are averaged over their own
    /// mass, so a cut node keeps the scale of a full average instead of shrinking with the dropped ones.
    fn chance(self: *Search, state: GameState, depth: u8, cprob: f64) f64 {
        if (depth == 0) return self.leaf(state.board);
        var empties: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] == 0) empties += 1;
            }
        }
        if (empties == 0) return self.value(state, depth, cprob);
        var total: f64 = 0;
        var mass: f64 = 0;
        for (0..core.BoardSize) |r| {
            for (0..core.BoardSize) |c| {
                if (state.board[r][c] != 0) continue;
                for ([_]struct { value: core.Tile, p: f64 }{ .{ .value = 2, .p = 0.9 }, .{ .value = 4, .p = 0.1 } }) |spawn| {
                    const p = cprob * spawn.p;
                    if (depth > 1 and p / empties < self.cutoff) continue;
                    var child = state;
                    child.board[r][c] = spawn.value;
                    total += p * self.value(child, depth, p);
                    mass += p;
                }
            }
        }
        // every branch cut: the node is already that unlikely, so the afterstate leaf is the answer
        return if (mass == 0) self.leaf(state.board) else total / mass;
    }
};

/// `depth.ply` own moves deep: my move -> spawn -> ... -> afterstate value + score_delta/reward_scale.
/// `ply` 2 is `deep` (full width); `ply` 3 is `deep3`, whose cost only fits with a spawn-probability cut.
fn deepMove(vfn: Value, state: GameState, stats: *Stats, cache: *ValueCache, depth: Depth) ?Direction {
    var search: Search = .{ .vfn = vfn, .cache = cache, .cutoff = depth.cutoff, .ply = depth.ply };
    defer stats.* = search.stats;
    var best: ?Direction = null;
    var best_value: f64 = -std.math.inf(f64);
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        const v = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale + search.chance(next, depth.ply, 1);
        if (best == null or v > best_value) {
            best = d;
            best_value = v;
        }
    }
    return best;
}

/// Lookahead of the #158 2-ply training variants: `search2_ply` own moves below the board being
/// valued, with a spawn layer before each of them — the tree `Mode.deep` visits at this ply.
const search2_ply: u8 = 2;

/// Bootstrap of the `search2` TD target (#158) of a post-spawn board: the `.deep` tree's shape with
/// its last reward moved into the leaf, `search2_ply - 1` searched rewards and then `V` of the
/// afterstate one own move past the last of them.
///
/// The per-move target is `r_t + bootValue(S_{t+1}, search2_ply - 1)` = `r_t + r_{t+1} + V(A_{t+2})`:
/// the reward of the step being trained plus the searched reward `r_{t+1}` and the value of the
/// afterstate both moves led to, whose own reward `r_{t+2}` is inside `V` and nowhere else.
/// `Search.value` at the same count instead returns `search2_ply` rewards plus `V` of the afterstate
/// the *last* reward reached, so that last reward lands in the target twice and the fixed point of
/// the `search2` target sits ~1.5x above the true value on a 1-reward-per-move chain.
fn search2Value(vfn: Value, state: GameState, cache: *ValueCache) f64 {
    // The leaf sits one own move deeper than `value` at the same reward count: key it as such.
    var search: Search = .{ .vfn = vfn, .cache = cache, .ply = search2_ply + 1 };
    return search.bootValue(state, search2_ply - 1, 1);
}

/// `pickMoveValue` for the 2-ply self-play policy (#158): candidates are ranked by the eval's `.deep`
/// value, `reward + Search.chance(afterstate)` — the `1 + search2_ply` own moves deep that `deepMove`
/// at `ply = search2_ply` scores — so `Pick.value` is that chance value (not `V`), covering
/// `search2_ply` moves and a reward-bearing leaf below the afterstate. `search2Value` bootstraps the
/// same tree with its last reward moved into the leaf, so a `search2` target spans `search2_ply`
/// rewards and V one move past the last of them. Spawns are enumerated, so this draws nothing from
/// the rng and the spawn sequence of the game is the same as with `greedy1`.
fn pickSearch2Value(vfn: Value, state: GameState, cache: *ValueCache) ?Pick {
    var search: Search = .{ .vfn = vfn, .cache = cache };
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
            .value = search.chance(next, search2_ply, 1),
        };
        const value = pick.reward + pick.value;
        if (best == null or value > best_value) {
            best = pick;
            best_value = value;
        }
    }
    return best;
}

pub const Mode = enum {
    /// 1-ply over afterstates: argmax of reward + V(afterstate).
    greedy,
    /// greedy plus one spawn of chance-node lookahead (see `chanceMove`).
    chance,
    /// 2-ply: one more own move and spawn layer below `chance` (see `deepMove`).
    deep,
    /// 3-ply: `deep` plus one more own move and spawn layer, with the probability cut (issue #108).
    deep3,
    /// 2-ply while the board has 8+ empty cells, then 3-ply with the 1e-2 cut (#157).
    phase_h1,
    /// Same start, 3-ply from 5..7 empty cells, and the 1e-4 cut once 4 or fewer are left (#157).
    phase_h2,
    /// 2-ply only down to 10 empty cells, then 3-ply from 6 and the 1e-4 cut near the end (#157).
    phase_h3,
};

/// One row of a phase schedule (#157): a board with at least `min_empty` empty cells searches `ply`
/// own moves deep, dropping spawn branches under path probability `cutoff` (`full_width_cutoff` keeps
/// every branch).
pub const Phase = struct { min_empty: u8, ply: u8, cutoff: f64 };

/// A row cutoff that keeps every spawn branch: no branch probability ever reaches 1.
const full_width_cutoff: f64 = 1;

/// Tighter cut for the last rows of `phase_h2` / `phase_h3` (#157): the board is nearly full there, so
/// the 3-ply tree stays small even with more spawn branches kept.
const phase_tight_cutoff: f64 = 1e-4;

/// The search depth of every `Mode`, one table (#139 style): rows are in descending `min_empty`, so the
/// first row the board's empty cells reach is its phase. `deep` and `deep3` are one row each, which is
/// what keeps their behavior exactly the pre-#157 search.
const phase_schedules = struct {
    const deep = [_]Phase{.{ .min_empty = 0, .ply = 2, .cutoff = full_width_cutoff }};
    const deep3 = [_]Phase{.{ .min_empty = 0, .ply = 3, .cutoff = deep3_cutoff }};
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
};

/// Rows `mode` searches with, in the order they are matched; empty for the 1-ply modes.
fn schedule(mode: Mode) []const Phase {
    return switch (mode) {
        .greedy, .chance => &.{},
        .deep => &phase_schedules.deep,
        .deep3 => &phase_schedules.deep3,
        .phase_h1 => &phase_schedules.phase_h1,
        .phase_h2 => &phase_schedules.phase_h2,
        .phase_h3 => &phase_schedules.phase_h3,
    };
}

/// `Search` runs full width at `cutoff = 0`; the table spells that as `full_width_cutoff` because no
/// branch probability reaches it. Converted once, where a row becomes a search.
fn searchCutoff(cutoff: f64) f64 {
    return if (cutoff >= full_width_cutoff) 0 else cutoff;
}

/// The row that applies to `board` under the search `mode`, in `Search`'s own units. Rows are in
/// descending `min_empty` and every schedule ends at 0, so one row always matches.
fn phaseDepth(mode: Mode, board: Board) Depth {
    const rows = schedule(mode);
    std.debug.assert(rows.len != 0);
    const empties = countEmpty(board);
    for (rows) |row| {
        if (empties >= row.min_empty) return .{ .ply = row.ply, .cutoff = searchCutoff(row.cutoff) };
    }
    unreachable; // every schedule ends with min_empty = 0
}

pub fn bestMove(net: *const Net, state: GameState, mode: Mode) ?Direction {
    return bestMoveValue(.{ .single = net }, state, mode);
}

/// `bestMove` for a single or staged value function (#107).
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

/// `bestMoveWithStats` with a caller-owned cache: reuse it across moves (issue #88) for speed.
/// A warm cache returns the same moves as a fresh one — only faster.
pub fn bestMoveWithCache(net: *const Net, state: GameState, mode: Mode, stats: *Stats, cache: *ValueCache) ?Direction {
    return bestMoveWithCacheValue(.{ .single = net }, state, mode, stats, cache);
}

pub fn bestMoveWithCacheValue(vfn: Value, state: GameState, mode: Mode, stats: *Stats, cache: *ValueCache) ?Direction {
    return switch (mode) {
        .greedy => if (pickMoveValue(vfn, state, false)) |p| p.dir else null,
        .chance => chanceMove(vfn, state),
        // every search mode takes its (ply, cutoff) for this board from the phase table (#157)
        .deep, .deep3, .phase_h1, .phase_h2, .phase_h3 => deepMove(vfn, state, stats, cache, phaseDepth(mode, state.board)),
    };
}

pub const PlayResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
    /// `forward` leaf evaluations summed over the game (0 for modes that do not search).
    leaves: u64 = 0,
};

/// Play one game to game over with the network policy.
pub fn play(net: *const Net, rng: std.Random, mode: Mode) PlayResult {
    return playValue(.{ .single = net }, rng, mode);
}

/// `play` for a single or staged value function (#107).
pub fn playValue(vfn: Value, rng: std.Random, mode: Mode) PlayResult {
    var stats: Stats = .{};
    return playWithStatsValue(vfn, rng, mode, &stats);
}

pub fn playWithStats(net: *const Net, rng: std.Random, mode: Mode, stats: *Stats) PlayResult {
    return playWithStatsValue(.{ .single = net }, rng, mode, stats);
}

pub fn playWithStatsValue(vfn: Value, rng: std.Random, mode: Mode, stats: *Stats) PlayResult {
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    var move_stats: Stats = .{};
    var cache: ValueCache = .{}; // one workspace for the whole game (issue #88)
    while (true) {
        move_stats = .{};
        const dir = bestMoveWithCacheValue(vfn, state, mode, &move_stats, &cache) orelse break;
        stats.leaves += move_stats.leaves;
        stats.hits += move_stats.hits;
        _ = state.move(dir, rng);
        moves += 1;
    }
    return .{ .score = state.score, .max_tile = core.maxTile(state.board), .moves = moves, .leaves = stats.leaves };
}

/// Buckets for the training diagnostic (issue #32): empty cells 0..3, 4..7, 8+; moves left 0..9, 10..29, 30+.
pub fn emptyBucket(empties: usize) usize {
    return if (empties <= 3) 0 else if (empties <= 7) 1 else 2;
}

pub fn distBucket(moves_left: usize) usize {
    return if (moves_left < 10) 0 else if (moves_left < 30) 1 else 2;
}

/// Max-tile buckets of the deep-position diagnostic (issue #159): <=256, 512, 1024, 2048, 4096, 8192+.
/// The tail beyond 4096 is where the value error is concentrated (#28/#32), so the table has to show
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
    /// Steps and |TD error| by max tile (#159). No window: the bucket of an afterstate is known at once.
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
/// compiling; the scan itself is the core board predicate (#139).
pub fn countEmpty(board: Board) usize {
    return core.countEmpty(board);
}

/// TD target of the per-move trainer (#158); see `Options.target`.
pub const Target = enum { td1, search2 };

/// Self-play policy of the per-move trainer (#158); see `Options.selfplay`.
pub const SelfPlay = enum { greedy1, deep2 };

/// Opt-in training variants (issue #33). Default `.{}` is the plain per-move TD(0) of `trainGame`.
pub const Options = struct {
    /// Candidate A: no per-move update during the game; afterwards train every stored afterstate from last to first.
    backward: bool = false,
    /// Candidate B: after the game, re-train the last `replay_k` afterstates once more (0 = off), oldest first.
    replay_k: usize = 0,
    /// Learning rate of the replay pass, as a factor of the training lr.
    replay_lr_scale: f64 = 0.5,
    /// Candidate D (#34): prioritized replay buffer; null = off. Normal per-move updates run as well.
    priority: ?*Priority = null,
    /// Candidate E (#59): with `priority` set, the `m` extra updates draw uniformly from buffer entries
    /// with 4..7 empty cells (midgame) instead of by |TD error|.
    midgame: bool = false,
    /// Candidate (#63): with `priority` set, the `m` extra updates keep the uniform share over the whole
    /// buffer and draw the rest by |TD error| from the 4..7 empty-cell bucket only.
    midmix: bool = false,
    /// Candidate (#72): apply every per-move TD update to all 8 dihedral transforms of the afterstate,
    /// with the same target (8 samples per visited board, deterministic).
    symmetry: bool = false,
    /// Candidate (#101): reward every move with -1 and sweep the afterstates backward at game over, so
    /// the value learns V ≈ -(moves to game over) and the self-play policy plays the fastest game over
    /// (a death net). Backward full return instead of per-move TD(0): with -1 per move the TD target is
    /// the same for every board and the net collapses to a constant (observed while building #101).
    death: bool = false,
    /// Candidate (#72): TD(lambda) with accumulating eligibility traces over the parameters; 0 = TD(0).
    lambda: f64 = 0,
    /// Candidate (#98): true batch symmetry augmentation — one averaged step per afterstate, where all 8
    /// transformed gradients come from one net snapshot (unlike `symmetry`, which steps 8 times sequentially).
    sym_batch: bool = false,
    /// Candidate (#84): fixed n-step TD on the afterstate chain; 0 = off. Standalone: when set it owns the
    /// game loop and the other options are ignored. 1 is TD(0), larger n propagates rewards faster.
    nstep: usize = 0,
    /// Candidate (#158): TD target of the per-move path (`trainGamePerMove`, the default). `td1` is the
    /// historical reward + V(next afterstate). `search2` keeps the eval's `.deep` tree (the same spawn
    /// layers) with `search2_ply - 1` searched rewards and then V of the afterstate one move past the
    /// last of them, so no reward is in the target twice (see `search2Value`). The
    /// `backward`/`death`/`nstep` paths keep the 1-ply policy.
    target: Target = .td1,
    /// Candidate (#158): self-play move of the per-move path. `greedy1` is the historical 1-ply argmax;
    /// `deep2` plays the eval's 2-ply search (`.deep`: same tie-break, spawns enumerated, no rng use).
    selfplay: SelfPlay = .greedy1,
    /// Candidate (#159) `dump`: append every afterstate the plain per-move path visits whose max tile
    /// reached `dump_min_tile` to `dump` (bounded, overflow dropped). The updates are unchanged.
    dump_record: bool = false,
    /// Deep-position buffer (#159): written by `dump_record`, read by the variants that train from it.
    dump: ?*Dump = null,
    /// Candidate (#159) `ft2`: one extra TD(0) update from a uniformly drawn deep afterstate of `dump`
    /// per self-play step, so half the update budget stays on the self-play trajectory (uniform over the
    /// whole game, shallow positions included) and half lands on the deep tail.
    deep_mix: bool = false,
};

/// Prioritized replay (issue #34): ring buffer of recent TD updates. After each game, `m` extra updates
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

    /// `m` extra updates, midgame variant (#59): uniform over entries with 4..7 empty cells,
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

    /// Midgame branch of the midmix variant (#63): entries with 4..7 empty cells, by priority weight.
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

/// One TD(0) step on a stored deep afterstate (#159 `ft2`). The board carries no history, so the reward
/// of the move that led out of it is unknown and the target is the value of its greedy successor (reward 0).
/// A dead board has no successor, hence no target.
fn deepStep(net: *Net, rng: std.Random, buffer: *Dump, lr: f64, cache: *Cache) void {
    const start = GameState{ .board = buffer.draw(rng), .score = 0 };
    const pick = pickMove(net, start, false) orelse return;
    _ = forward(net, start.board, cache);
    train(net, cache, pick.value, lr);
}

/// Deep-position dump (issue #159): fixed-capacity buffer of afterstates whose max tile reached
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
        for (0..core.BoardCellCount) |cell| {
            const value = board[cell / core.BoardSize][cell % core.BoardSize];
            self.entries[self.len][cell] = if (value == 0) 0 else @intCast(std.math.log2_int(core.Tile, value));
        }
        self.len += 1;
    }

    /// The `index`-th stored afterstate, decoded back to tiles.
    pub fn boardAt(self: *const Dump, index: usize) Board {
        var out: Board = undefined;
        for (0..core.BoardCellCount) |cell| {
            const level = self.entries[index][cell];
            out[cell / core.BoardSize][cell % core.BoardSize] = if (level == 0) 0 else @as(core.Tile, 1) << @intCast(level);
        }
        return out;
    }

    /// Uniform draw of one stored afterstate, counted so the 50/50 mix of `ft2` is observable.
    /// Uniform on purpose: #33/#34/#56 lost with |TD error| and midgame resampling of existing data.
    pub fn draw(self: *Dump, rng: std.Random) Board {
        std.debug.assert(self.len > 0);
        self.draws += 1;
        return self.boardAt(rng.uintLessThan(usize, self.len));
    }

    fn encodeInto(self: *const Dump, bytes: []u8) void {
        @memcpy(bytes[0..8], dump_magic);
        std.mem.writeInt(u32, bytes[8..12], @intCast(self.len), .little);
        @memcpy(bytes[dump_header_len..], std.mem.sliceAsBytes(self.entries[0..self.len]));
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
        if (!std.mem.eql(u8, bytes[0..8], dump_magic)) return error.BadDumpMagic;
        const count = std.mem.readInt(u32, bytes[8..12], .little);
        if (count > dump_cap) return error.DumpTooLarge;
        if (bytes.len != dump_header_len + @as(usize, count) * dump_entry_len) return error.WrongDumpFileSize;
        @memcpy(std.mem.sliceAsBytes(self.entries[0..count]), bytes[dump_header_len..]);
        self.len = count;
    }

    pub fn load(self: *Dump, io: std.Io, gpa: std.mem.Allocator, path: []const u8) !void {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(dump_header_len + dump_cap * dump_entry_len));
        defer gpa.free(bytes);
        try self.parse(bytes);
    }
};

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
    return trainGameOpts(net, rng, lr, diag, .{}).score;
}

pub const GameResult = struct { score: u32, moves: u32 };

/// `trainGame` with options. `diag` is only recorded on the default path (`opts` all off).
pub fn trainGameOpts(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) GameResult {
    if (opts.nstep > 0) return trainGameNStep(net, rng, lr, opts.nstep);
    if (opts.priority != null and !opts.death) return trainGamePerMove(net, rng, lr, diag, opts);
    if (!opts.backward and opts.replay_k == 0 and !opts.death) return trainGamePerMove(net, rng, lr, diag, opts);
    var steps: [step_cap]Step = undefined;
    var n: usize = 0;
    var state = GameState.gameStart(rng);
    var cache: Cache = .{};
    var pending: ?Pick = null;
    while (pickMove(net, state, opts.death)) |pick| {
        if (pending) |p| if (!opts.backward and !opts.death) trainAfter(net, &cache, p.after, p.reward + pick.value, lr);
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
            const next = if (i + 1 < n) valueOf(net, steps[(i + 1) % step_cap].after) else 0;
            trainAfter(net, &cache, s.after, s.reward + next, lr);
        }
    }
    if (opts.replay_k > 0) {
        const lr_replay = lr * opts.replay_lr_scale;
        var i = n -| @min(opts.replay_k, step_cap);
        while (i < n) : (i += 1) {
            const s = steps[i % step_cap];
            const next = if (i + 1 < n) valueOf(net, steps[(i + 1) % step_cap].after) else 0;
            trainAfter(net, &cache, s.after, s.reward + next, lr_replay);
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

/// Global L2 norm the averaged symmetry gradient is clipped to (#98). Measured averaged-gradient norms are
/// 0.04..0.26 on a fresh net but 4..300 late in training, so a unit-norm step is a no-op early and, with lr,
/// caps a late-game update at lr = 0.003 against a mean |w| of 0.44.
pub const sym_batch_clip: f64 = 1.0;

/// Batch symmetry (#98): 8 gradients from one snapshot, averaged elementwise, one SGD step.
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

/// One per-move TD update on `after` with `target`: all 8 dihedral transforms when `opts.symmetry`,
/// and TD(lambda) through `trace` when `opts.lambda > 0` (otherwise the plain TD(0) `train` step).
fn trainStep(net: *Net, cache: *Cache, after: Board, target: f64, lr: f64, opts: Options, trace: *Net) void {
    if (opts.sym_batch) return trainSymBatch(net, after, target, lr);
    const transforms: usize = if (opts.symmetry) symmetry_count else 1;
    for (0..transforms) |k| {
        _ = forward(net, if (opts.symmetry) transformBoard(k, after) else after, cache);
        if (opts.lambda > 0) trainTrace(net, cache, target, lr, trace, opts.lambda) else train(net, cache, target, lr);
    }
}

fn trainGamePerMove(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) GameResult {
    const pr = opts.priority;
    const vfn: Value = .{ .single = net };
    var state = GameState.gameStart(rng);
    var pending: ?Pick = null;
    var cache: Cache = .{};
    var moves: u32 = 0;
    var trace: Net = std.mem.zeroes(Net);
    while (selfplayPick(vfn, state, opts.death, opts.selfplay)) |pick| {
        moves += 1;
        if (opts.dump_record) if (opts.dump) |buffer| buffer.offer(pick.after);
        if (pending) |p| {
            // `state` is still the post-spawn board the pending step led into (this move has not been
            // applied yet), which is the board the `search2` target evaluates.
            const target = p.reward + tdBootstrap(net, pick, state, opts);
            if (diag) |d| d.push(p.after, @abs(target - p.value), pick.value - p.value);
            if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = pick.after, .terminal = false, .err = @abs(target - p.value) });
            trainStep(net, &cache, p.after, target, lr, opts, &trace);
        }
        pending = pick;
        // `ft2` mixes one deep update into every self-play step: 50% self-play / 50% deep afterstates.
        if (opts.deep_mix) if (opts.dump) |buffer| if (buffer.len > 0) deepStep(net, rng, buffer, lr, &cache);
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| {
        if (diag) |d| d.push(p.after, @abs(p.reward - p.value), -p.value);
        if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = p.after, .terminal = true, .err = @abs(p.reward - p.value) });
        trainStep(net, &cache, p.after, p.reward, lr, opts, &trace);
    }
    if (pr) |b| for (0..b.m) |_| {
        const e = &b.items[if (opts.midmix)
            b.sampleMidmix(rng)
        else if (opts.midgame)
            b.sampleMidgame(rng)
        else
            b.sample(rng)];
        const target = e.reward + if (e.terminal) 0 else valueOf(net, e.next);
        _ = forward(net, e.after, &cache);
        e.err = @abs(target - cache.out);
        train(net, &cache, target, lr);
    };
    if (diag) |d| d.finish();
    return .{ .score = state.score, .moves = moves };
}

/// One self-play move of the per-move trainer (#158). `greedy1` is the historical 1-ply argmax;
/// `deep2` plays the eval's 2-ply search. Neither draws from the rng (the search enumerates spawns), so
/// both policies spawn the same tiles in the same order and differ only in the move they pick.
fn selfplayPick(vfn: Value, state: GameState, death: bool, selfplay: SelfPlay) ?Pick {
    return switch (selfplay) {
        .greedy1 => pickMoveValue(vfn, state, death),
        .deep2 => pickSearch2Value(vfn, state, sharedValueCache()),
    };
}

/// Bootstrap of one per-move TD target (#158): the value of what follows the step's reward.
/// `td1` keeps the historical V of the afterstate the next move reached; `search2` bootstraps
/// `spawned`, the post-spawn board the step led into, with `search2_ply - 1` searched rewards and V of
/// the afterstate one move past the last of them (see `search2Value`). With a `deep2` policy `next.value`
/// is already a search value, so `td1` reads V of the afterstate directly then.
fn tdBootstrap(net: *const Net, next: Pick, spawned: GameState, opts: Options) f64 {
    return switch (opts.target) {
        .td1 => if (opts.selfplay == .deep2) valueOf(net, next.after) else next.value,
        .search2 => search2Value(.{ .single = net }, spawned, sharedValueCache()),
    };
}

/// Online n-step TD (issue #84) on the same 1-ply self-play policy as `trainGamePerMove`: each step enters
/// the window with V of its afterstate, and the step leaving the window is trained on its n rewards plus
/// that value. The last min(n, game length) steps are flushed at game over with the rewards alone.
fn trainGameNStep(net: *Net, rng: std.Random, lr: f64, n: usize) GameResult {
    var window = NStepWindow.init(n);
    var state = GameState.gameStart(rng);
    var cache: Cache = .{};
    var moves: u32 = 0;
    while (pickMove(net, state, false)) |pick| {
        moves += 1;
        if (window.push(.{ .after = pick.after, .reward = pick.reward }, pick.value)) |done|
            trainAfter(net, &cache, done.after, done.target, lr);
        _ = state.move(pick.dir, rng);
    }
    while (window.nextFinish()) |done| trainAfter(net, &cache, done.after, done.target, lr);
    return .{ .score = state.score, .moves = moves };
}

/// MS-TD (#107) self-play game over the staged value: the same 1-ply policy as `trainGamePerMove`,
/// one TD(0) step per move (gamma = 1), and each step trains only the net of the afterstate it reached.
pub fn trainGameStaged(staged: *Staged, rng: std.Random, lr: f64) GameResult {
    const value: Value = .{ .staged = staged };
    var state = GameState.gameStart(rng);
    var pending: ?Pick = null;
    var moves: u32 = 0;
    while (pickMoveValue(value, state, false)) |pick| {
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
    diag: Diag = .{},
};

fn runBatchGame(ctx: BatchGameCtx, i: usize) BatchGame {
    const game = ctx.first_game + i;
    var result: BatchGame = .{ .net = ctx.net, .score = 0 };
    var rng = std.Random.DefaultPrng.init(trainingSeed(ctx.seed, game));
    result.score = trainGameOpts(&result.net, rng.random(), ctx.lr, if (ctx.want_diag) &result.diag else null, ctx.opts).score;
    return result;
}

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
pub fn trainBatch(net: *Net, seed: u64, first_game: usize, games: usize, lr: f64, diag: ?*Diag) u32 {
    return trainBatchOpts(net, seed, first_game, games, lr, diag, .{});
}

/// `trainBatch` with options: every game trains with `trainGameOpts(.., opts)`, so any stateless
/// variant (`backward`, `replay`, `sym`, `symlam`, `nstep`) batches the same way `base` does.
pub fn trainBatchOpts(net: *Net, seed: u64, first_game: usize, games: usize, lr: f64, diag: ?*Diag, opts: Options) u32 {
    std.debug.assert(games >= 1 and games <= parallel.max_threads);
    // the priority buffer is cross-game state shared by pointer: batching it would be a data race
    std.debug.assert(opts.priority == null);
    var units: [parallel.max_threads]BatchGame = undefined;
    parallel.map(BatchGame, units[0..games], BatchGameCtx{
        .net = net.*,
        .seed = seed,
        .first_game = first_game,
        .lr = lr,
        .want_diag = diag != null,
        .opts = opts,
    }, runBatchGame);

    const start = net.*;
    const start_w = std.mem.bytesAsSlice(f64, std.mem.asBytes(&start));
    const out_w = std.mem.bytesAsSlice(f64, std.mem.asBytes(net));
    for (out_w) |*w| w.* = 0;
    for (units[0..games]) |*u| {
        const after = std.mem.bytesAsSlice(f64, std.mem.asBytes(&u.net));
        for (out_w, after, start_w) |*o, v, s| o.* += v - s;
    }
    const n: f64 = @floatFromInt(games);
    for (out_w, start_w) |*o, s| o.* = s + o.* / n;
    if (diag) |d| for (units[0..games]) |u| mergeDiag(d, &u.diag);
    return units[games - 1].score;
}

/// Weight file: 8-byte magic, u32 version, u32 endianness marker (all little-endian), then the f64 weights as little-endian.
pub const file_magic = "2048-NN\x00";
pub const file_version: u32 = 1;
const endian_marker: u32 = 0x01020304;
const header_len = 16;
pub const file_len = header_len + @sizeOf(Net);

/// Staged weight file (#107): the same header with its own 8-byte magic and `stage_count` weight blocks.
pub const staged_file_magic = "2048-MS\x00";
pub const staged_file_len = header_len + @sizeOf(Net) * stage_count;

fn writeWeights(out: []u8, net: *const Net) void {
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(net)), 0..) |w, i| {
        std.mem.writeInt(u64, out[i * 8 ..][0..8], @bitCast(w), .little);
    }
}

fn readWeights(bytes: []const u8) Net {
    var net: Net = undefined;
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&net)), 0..) |*w, i| {
        w.* = @bitCast(std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little));
    }
    return net;
}

pub fn encode(net: *const Net) [file_len]u8 {
    var out: [file_len]u8 = undefined;
    @memcpy(out[0..8], file_magic);
    std.mem.writeInt(u32, out[8..12], file_version, .little);
    std.mem.writeInt(u32, out[12..16], endian_marker, .little);
    writeWeights(out[header_len..], net);
    return out;
}

/// Diagnostic: how many times `parse` ran on this thread (issue #87 tests assert weights are
/// parsed once per thread, not once per move).
pub threadlocal var parse_count: usize = 0;

pub fn parse(bytes: []const u8) !Net {
    parse_count += 1;
    if (bytes.len != file_len) return error.WrongWeightFileSize;
    if (!std.mem.eql(u8, bytes[0..8], file_magic)) return error.BadWeightMagic;
    if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedWeightVersion;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongWeightEndianness;
    return readWeights(bytes[header_len..]);
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

test "deep3 keeps the value within 5% of the unpruned 3-ply search" {
    const net = try parse(@embedFile("nn_weights.bin"));
    var cache: ValueCache = .{};
    var full: Search = .{ .vfn = .{ .single = &net }, .cache = &cache };
    var pruned: Search = .{ .vfn = .{ .single = &net }, .cache = &cache, .cutoff = deep3_cutoff };
    const plain = full.value(late_board, 3, 1);
    const cut = pruned.value(late_board, 3, 1);
    try std.testing.expect(@abs(cut - plain) <= 0.05 * @abs(plain));
}

/// Modes that pick their own depth from the phase table (#157).
const phase_modes = [_]Mode{ .phase_h1, .phase_h2, .phase_h3 };

test "phase modes return only legal moves on random boards (#157)" {
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

test "phase modes are deterministic for a fixed board (#157)" {
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
    try std.testing.expectEqual(@as(?Direction, .left), bestMove(&net, mid_board, .greedy));
    try std.testing.expectEqual(@as(?Direction, .left), bestMove(&net, mid_board, .chance));
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
    try std.testing.expectEqual(@as(?Pick, null), pickSearch2Value(.{ .single = &net }, .{ .board = dead_board }, &cache_a));

    const first = pickSearch2Value(.{ .single = &net }, mid_board, &cache_a).?;
    const second = pickSearch2Value(.{ .single = &net }, mid_board, &cache_b).?;
    try std.testing.expectEqual(first.dir, second.dir);
    try std.testing.expectEqual(first.after, second.after);
    try std.testing.expectEqual(first.reward, second.reward);
    var moved = mid_board;
    try std.testing.expect(moved.slide(first.dir).changed);
    try std.testing.expectEqual(first.after, moved.board);
    // the self-play policy and the eval policy must look at the same tree, otherwise #158 compares apples to oranges
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

test "search2 bootstraps from the post-spawn board, td1 from V of the next afterstate (#158)" {
    // zero net: V is 0 everywhere, so the td1 bootstrap cannot see a single future reward
    const net = std.mem.zeroes(Net);
    const next: Pick = .{ .dir = .left, .after = post_spawn_board, .reward = 0, .value = 0 };
    const spawned: GameState = .{ .board = post_spawn_board };
    const td1 = tdBootstrap(&net, next, spawned, .{ .target = .td1 });
    const search2 = tdBootstrap(&net, next, spawned, .{ .target = .search2 });
    try std.testing.expectEqual(@as(f64, 0), td1);
    // merging the 2s is 4 points / reward_scale, and the 2-ply search starts on this post-spawn board
    try std.testing.expect(search2 >= 4 / reward_scale);
    // same target on a board with no legal move stays 0: the bootstrap is the search of the given board
    const dead_spawned: GameState = .{ .board = dead_board };
    try std.testing.expectEqual(@as(f64, 0), tdBootstrap(&net, next, dead_spawned, .{ .target = .search2 }));
    // with the committed weights the two targets also differ on a midgame board
    const weights = try parse(@embedFile("nn_weights.bin"));
    const trained_next: Pick = .{ .dir = .left, .after = mid_board.board, .reward = 0, .value = valueOf(&weights, mid_board.board) };
    const boot1 = tdBootstrap(&weights, trained_next, mid_board, .{ .target = .td1 });
    const boot2 = tdBootstrap(&weights, trained_next, mid_board, .{ .target = .search2 });
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
    const boot = search2Value(vfn, spawned, &cache);
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var train_rng = std.Random.DefaultPrng.init(12);
    for (0..60) |_| _ = trainGameOpts(&net, train_rng.random(), 0.02, null, .{ .death = true });

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

/// Mean and standard error (sample sd / sqrt(n)) of per-seed-range averages.
pub const MeanErr = struct { mean: f64, stderr: f64 };

pub fn meanStderr(xs: []const f64) MeanErr {
    const n: f64 = @floatFromInt(xs.len);
    var sum: f64 = 0;
    for (xs) |x| sum += x;
    const mean = sum / n;
    if (xs.len < 2) return .{ .mean = mean, .stderr = 0 };
    var ss: f64 = 0;
    for (xs) |x| ss += (x - mean) * (x - mean);
    return .{ .mean = mean, .stderr = @sqrt(ss / (n - 1) / n) };
}

/// Score per CPU second; 0 when no time elapsed (avoids division by zero).
pub fn scorePerSecond(total_score: f64, elapsed_ns: i96) f64 {
    if (elapsed_ns <= 0) return 0;
    return total_score / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
}

test "meanStderr gives mean and sample standard error" {
    const xs = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    const r = meanStderr(&xs);
    try std.testing.expectApproxEqAbs(@as(f64, 5), r.mean, 1e-12);
    // sample variance = 32 / 7, stderr = sqrt(32/7) / sqrt(8)
    try std.testing.expectApproxEqAbs(@sqrt(32.0 / 7.0) / @sqrt(8.0), r.stderr, 1e-12);
}

/// Linear correlation of two equal-length samples (issue #101 probe); 0 when either is constant.
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

test "meanStderr of one value has zero stderr" {
    const r = meanStderr(&[_]f64{3});
    try std.testing.expectApproxEqAbs(@as(f64, 3), r.mean, 1e-12);
    try std.testing.expectEqual(@as(f64, 0), r.stderr);
}

test "scorePerSecond divides total score by elapsed seconds" {
    // 3000 points in 1.5 s is 2000 points per second
    try std.testing.expectApproxEqAbs(@as(f64, 2000), scorePerSecond(3000, 1_500_000_000), 1e-9);
}

test "scorePerSecond of zero elapsed time is zero" {
    try std.testing.expectEqual(@as(f64, 0), scorePerSecond(3000, 0));
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    const plain = trainGameOpts(&a, rng_a.random(), 0.003, null, .{});
    var dump: Dump = .{};
    var c = b;
    var rng_c = std.Random.DefaultPrng.init(12);
    const recorded = trainGameOpts(&b, rng_b.random(), 0.003, null, .{ .dump_record = true, .dump = &dump });
    const again = trainGameOpts(&c, rng_c.random(), 0.003, null, .{ .dump_record = true, .dump = &dump });
    try std.testing.expectEqual(plain.score, recorded.score);
    try std.testing.expectEqual(plain.moves, recorded.moves);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    // recording is a pure side effect: a second identical game overwrites the same entries in the same order
    try std.testing.expectEqual(recorded.score, again.score);
    try std.testing.expectEqual(recorded.moves, again.moves);
    // a fresh net stays shallow, so most games record nothing; whatever it does record must be deep enough
    for (0..dump.len) |i| try std.testing.expect(core.maxTile(dump.boardAt(i)) >= dump_min_tile);
}

test "ft2 mixes one deep draw per self-play step and is deterministic" {
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
    const game = trainGameOpts(&a, rng_a.random(), 0.001, null, .{ .deep_mix = true, .dump = &dump });
    const again = trainGameOpts(&b, rng_b.random(), 0.001, null, .{ .deep_mix = true, .dump = &dump });
    _ = trainGameOpts(&plain_net, rng_plain.random(), 0.001, null, .{});

    // one deep draw per self-play step, and no draw at all without the dump: 50% self-play / 50% deep
    try std.testing.expectEqual(@as(u64, game.moves) * 2, dump.draws);
    // same seed -> same games and byte-identical weights
    try std.testing.expectEqual(game.moves, again.moves);
    try std.testing.expectEqual(game.score, again.score);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    // the deep half is not a no-op: it changes the weights (and with them the rest of the game)
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&plain_net)));
}

test "trainGame with diagnostic gives the same weights and records steps" {
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(12);
    for (0..5) |_| _ = trainGame(&net, rng.random(), 0.003, null);
    try std.testing.expectEqual(@as(u32, 1621370333), std.hash.Crc32.hash(std.mem.asBytes(&net)));
}

test "trainGameOpts with default options equals trainGame" {
    var init_rng = std.Random.DefaultPrng.init(11);
    var a = Net.init(init_rng.random());
    var b = a;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&a, rng_a.random(), 0.003, null);
    _ = trainGameOpts(&b, rng_b.random(), 0.003, null, .{});
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

test "td1 + greedy1 is the default self-play path, bit for bit (#158)" {
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var default_opts = base;
    var explicit = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    const plain = trainGame(&base, rng_a.random(), 0.003, null);
    const with_default = trainGameOpts(&default_opts, rng_b.random(), 0.003, null, .{}).score;
    const with_td1 = trainGameOpts(&explicit, rng_c.random(), 0.003, null, .{ .target = .td1, .selfplay = .greedy1 }).score;
    try std.testing.expectEqual(plain, with_default);
    try std.testing.expectEqual(plain, with_td1);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&default_opts));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&explicit));
}

test "search2, deep2 and tp2 train one game without NaN (#158)" {
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
        const result = trainGameOpts(&net, rng.random(), 0.003, null, opts);
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var one = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    const td0 = trainGameOpts(&base, rng_a.random(), 0.003, null, .{});
    const nstep = trainGameOpts(&one, rng_b.random(), 0.003, null, .{ .nstep = 1 });
    try std.testing.expectEqual(td0, nstep);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&one));
}

test "nstep = 4 changes the weights, stays finite and is deterministic" {
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var plain = base;
    var four = base;
    var again = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = trainGameOpts(&plain, rng_a.random(), 0.003, null, .{});
    _ = trainGameOpts(&four, rng_b.random(), 0.003, null, .{ .nstep = 4 });
    _ = trainGameOpts(&again, rng_c.random(), 0.003, null, .{ .nstep = 4 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&plain), std.mem.asBytes(&four)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&four), std.mem.asBytes(&again));
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&four))) |w| try std.testing.expect(std.math.isFinite(w));
}

test "nstep = 0 equals the default path" {
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var off = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGameOpts(&base, rng_a.random(), 0.003, null, .{});
    _ = trainGameOpts(&off, rng_b.random(), 0.003, null, .{ .nstep = 0 });
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&off));
}

test "backward sweep changes the weights" {
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var back = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = trainGameOpts(&back, rng_b.random(), 0.003, null, .{ .backward = true });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&base), std.mem.asBytes(&back)));
}

test "replay tail changes the weights and K=0 is off" {
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var replay = base;
    var off = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = trainGameOpts(&replay, rng_b.random(), 0.003, null, .{ .replay_k = 30 });
    _ = trainGameOpts(&off, rng_c.random(), 0.003, null, .{ .replay_k = 0 });
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&base), std.mem.asBytes(&replay)));
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&base), std.mem.asBytes(&off));
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var base = Net.init(init_rng.random());
    var prio = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var pr: Priority = .{};
    _ = trainGame(&base, rng_a.random(), 0.003, null);
    _ = trainGameOpts(&prio, rng_b.random(), 0.003, null, .{ .priority = &pr });
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
    _ = trainBatch(&par, 77, 0, games, 0.001, null);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&serial), std.mem.asBytes(&par));
}

test "parallel training batch with variant opts matches a serial run of the same game seeds" {
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
            _ = trainGameOpts(u, rng.random(), 0.001, null, opts);
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
        _ = trainBatchOpts(&par, 77, 0, games, 0.001, null, opts);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&serial), std.mem.asBytes(&par));
    }
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
    var init_rng = std.Random.DefaultPrng.init(11);
    var net = Net.init(init_rng.random());
    var rng = std.Random.DefaultPrng.init(12);
    for (0..5) |_| _ = trainGameOpts(&net, rng.random(), 0.003, null, .{ .symmetry = false, .lambda = 0 });
    try std.testing.expectEqual(@as(u32, 1621370333), std.hash.Crc32.hash(std.mem.asBytes(&net)));
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
    var init_rng = std.Random.DefaultPrng.init(11);
    const base = Net.init(init_rng.random());
    var plain = base;
    var sym = base;
    var symlam = base;
    var rng_a = std.Random.DefaultPrng.init(12);
    var rng_b = std.Random.DefaultPrng.init(12);
    var rng_c = std.Random.DefaultPrng.init(12);
    _ = trainGameOpts(&plain, rng_a.random(), 0.003, null, .{});
    _ = trainGameOpts(&sym, rng_b.random(), 0.003, null, .{ .symmetry = true });
    _ = trainGameOpts(&symlam, rng_c.random(), 0.003, null, .{ .symmetry = true, .lambda = 0.7 });
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

test "Staged.forward reads the net of the board's own stage" {
    var prng = std.Random.DefaultPrng.init(11);
    var staged = Staged.init(prng.random());
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
    var staged = Staged.init(prng.random());
    const after = boardWithEmpties(16); // stage 6, the net that gets the step
    const next = boardWithEmpties(1); // stage 0, read for the target only
    var pristine = staged;
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
    var staged = Staged.init(prng.random());
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
    const staged = Staged.init(prng.random());
    try staged.saveFile(io, std.testing.allocator, path);
    var loaded: Staged = undefined;
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
    var init_a = std.Random.DefaultPrng.init(19);
    var init_b = std.Random.DefaultPrng.init(19);
    var staged_a = Staged.init(init_a.random());
    var staged_b = Staged.init(init_b.random());
    const before = staged_a;
    var rng_a = std.Random.DefaultPrng.init(23);
    var rng_b = std.Random.DefaultPrng.init(23);
    const game_a = trainGameStaged(&staged_a, rng_a.random(), 0.003);
    const game_b = trainGameStaged(&staged_b, rng_b.random(), 0.003);
    try std.testing.expectEqual(game_a, game_b);
    for (staged_a.nets, staged_b.nets) |a, b| try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&before.nets), std.mem.asBytes(&staged_a.nets)));
}

test "bestMove and play accept the staged value" {
    var prng = std.Random.DefaultPrng.init(29);
    const staged = Staged.init(prng.random());
    const value: Value = .{ .staged = &staged };
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

