const std = @import("std");
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

/// Afterstate chosen by the 1-ply policy, with its scaled reward and value estimate.
const Pick = struct {
    dir: Direction,
    after: Board,
    reward: f64,
    value: f64,
};

/// Best legal move by reward + V(afterstate); null when no move changes the board.
fn pickMove(net: *const Net, state: GameState) ?Pick {
    var best: ?Pick = null;
    for ([_]Direction{ .up, .left, .down, .right }) |d| {
        var next = state;
        const moved = next.slide(d);
        if (!moved.changed) continue;
        var cache: Cache = .{};
        const reward = @as(f64, @floatFromInt(moved.score_delta)) / reward_scale;
        const pick: Pick = .{ .dir = d, .after = next.board, .reward = reward, .value = forward(net, next.board, &cache) };
        if (best == null or pick.reward + pick.value > best.?.reward + best.?.value) best = pick;
    }
    return best;
}

/// Value of a post-spawn board for the chance-node search: best reward + V over the legal moves (0 if none).
fn bestAfterValue(net: *const Net, board_state: GameState) f64 {
    const pick = pickMove(net, board_state) orelse return 0;
    return pick.reward + pick.value;
}

/// Expectimax with one spawn of lookahead: for each move, average over spawns of the best NN afterstate value.
fn chanceMove(net: *const Net, state: GameState) ?Direction {
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
                    total += spawn.p * bestAfterValue(net, child);
                }
            }
        }
        const chance = if (empties == 0) bestAfterValue(net, moved) else total / empties;
        const value = @as(f64, @floatFromInt(result.score_delta)) / reward_scale + chance;
        if (best == null or value > best_value) {
            best = d;
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
};

pub fn bestMove(net: *const Net, state: GameState, mode: Mode) ?Direction {
    return switch (mode) {
        .greedy => if (pickMove(net, state)) |p| p.dir else null,
        .chance => chanceMove(net, state),
    };
}

pub const PlayResult = struct {
    score: u32,
    max_tile: core.Tile,
    moves: u32,
};

/// Play one game to game over with the network policy.
pub fn play(net: *const Net, rng: std.Random, mode: Mode) PlayResult {
    var state = GameState.gameStart(rng);
    var moves: u32 = 0;
    while (bestMove(net, state, mode)) |d| {
        _ = state.move(d, rng);
        moves += 1;
    }
    var max_tile: core.Tile = 0;
    for (state.board) |row| for (row) |value| {
        max_tile = @max(max_tile, value);
    };
    return .{ .score = state.score, .max_tile = max_tile, .moves = moves };
}

/// Buckets for the training diagnostic (issue #32): empty cells 0..3, 4..7, 8+; moves left 0..9, 10..29, 30+.
pub fn emptyBucket(empties: usize) usize {
    return if (empties <= 3) 0 else if (empties <= 7) 1 else 2;
}

pub fn distBucket(moves_left: usize) usize {
    return if (moves_left < 10) 0 else if (moves_left < 30) 1 else 2;
}

/// Per-bucket sums over the training steps: TD error |target - V(s)| and value change V(s_{t+1}) - V(s_t).
pub const Diag = struct {
    pub const Stat = struct { count: u64 = 0, abs_err: f64 = 0, drop: f64 = 0 };
    /// Steps still inside the last-30 window; their distance to game over is not known until the game ends.
    const Pending = struct { empties: usize, err: f64, drop: f64 };
    const window = 30;

    stats: [3][3]Stat = @splat(@splat(.{})),
    ring: [window]Pending = undefined,
    pushed: usize = 0,

    pub fn add(self: *Diag, empties: usize, moves_left: usize, err: f64, drop: f64) void {
        const s = &self.stats[emptyBucket(empties)][distBucket(moves_left)];
        s.count += 1;
        s.abs_err += err;
        s.drop += drop;
    }

    /// Record one training step of the current game. Steps older than the window are at least 30 moves from the end.
    pub fn push(self: *Diag, empties: usize, err: f64, drop: f64) void {
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

pub fn countEmpty(board: Board) usize {
    var n: usize = 0;
    for (board) |row| for (row) |v| {
        if (v == 0) n += 1;
    };
    return n;
}

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

    fn isMidgame(board: Board) bool {
        const empty = countEmpty(board);
        return empty >= 4 and empty <= 7;
    }

    fn weight(self: *const Priority, err: f64) f64 {
        return std.math.pow(f64, @abs(err) + err_floor, self.alpha);
    }
};

/// Stored afterstates of one game for the variants (ring buffer; the last `step_cap` steps are kept).
/// shortcut: games longer than step_cap only replay/sweep their last step_cap afterstates, upgrade if games get that long.
const step_cap = 8192;
const Step = struct { after: Board, reward: f64 };

/// TD(0) on afterstates over one self-play game (greedy 1-ply moves, fixed rng).
/// Target for the previous afterstate = its scaled reward + V(next afterstate); 0 after game over.
/// `diag` (optional) records each step's TD error and value change; it does not touch the rng or the update.
pub fn trainGame(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag) u32 {
    return trainGameOpts(net, rng, lr, diag, .{}).score;
}

pub const GameResult = struct { score: u32, moves: u32 };

/// `trainGame` with options. `diag` is only recorded on the default path (`opts` all off).
pub fn trainGameOpts(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, opts: Options) GameResult {
    if (opts.priority) |pr| return trainGamePerMove(net, rng, lr, diag, pr, opts.midgame);
    if (!opts.backward and opts.replay_k == 0) return trainGamePerMove(net, rng, lr, diag, null, false);
    var steps: [step_cap]Step = undefined;
    var n: usize = 0;
    var state = GameState.gameStart(rng);
    var cache: Cache = .{};
    var pending: ?Pick = null;
    while (pickMove(net, state)) |pick| {
        if (pending) |p| if (!opts.backward) trainAfter(net, &cache, p.after, p.reward + pick.value, lr);
        steps[n % step_cap] = .{ .after = pick.after, .reward = pick.reward };
        n += 1;
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| if (!opts.backward) trainAfter(net, &cache, p.after, p.reward, lr);

    if (opts.backward) {
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

fn trainGamePerMove(net: *Net, rng: std.Random, lr: f64, diag: ?*Diag, pr: ?*Priority, midgame: bool) GameResult {
    var state = GameState.gameStart(rng);
    var pending: ?Pick = null;
    var cache: Cache = .{};
    var moves: u32 = 0;
    while (pickMove(net, state)) |pick| {
        moves += 1;
        if (pending) |p| {
            _ = forward(net, p.after, &cache);
            const target = p.reward + pick.value;
            if (diag) |d| d.push(countEmpty(p.after), @abs(target - p.value), pick.value - p.value);
            if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = pick.after, .terminal = false, .err = @abs(target - p.value) });
            train(net, &cache, target, lr);
        }
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| {
        _ = forward(net, p.after, &cache);
        if (diag) |d| d.push(countEmpty(p.after), @abs(p.reward - p.value), -p.value);
        if (pr) |b| b.push(.{ .after = p.after, .reward = p.reward, .next = p.after, .terminal = true, .err = @abs(p.reward - p.value) });
        train(net, &cache, p.reward, lr);
    }
    if (pr) |b| for (0..b.m) |_| {
        const e = &b.items[if (midgame) b.sampleMidgame(rng) else b.sample(rng)];
        const target = e.reward + if (e.terminal) 0 else valueOf(net, e.next);
        _ = forward(net, e.after, &cache);
        e.err = @abs(target - cache.out);
        train(net, &cache, target, lr);
    };
    if (diag) |d| d.finish();
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

const BatchGameCtx = struct { net: Net, seed: u64, first_game: usize, lr: f64, want_diag: bool };

const BatchGame = struct {
    net: Net,
    score: u32,
    diag: Diag = .{},
};

fn runBatchGame(ctx: BatchGameCtx, i: usize) BatchGame {
    const game = ctx.first_game + i;
    var result: BatchGame = .{ .net = ctx.net, .score = 0 };
    var rng = std.Random.DefaultPrng.init(trainingSeed(ctx.seed, game));
    result.score = trainGame(&result.net, rng.random(), ctx.lr, if (ctx.want_diag) &result.diag else null);
    return result;
}

fn mergeDiag(dst: *Diag, src: *const Diag) void {
    for (0..3) |e| for (0..3) |k| {
        dst.stats[e][k].count += src.stats[e][k].count;
        dst.stats[e][k].abs_err += src.stats[e][k].abs_err;
        dst.stats[e][k].drop += src.stats[e][k].drop;
    };
}

/// Runs `games` (<= max_threads) self-play games in parallel from the same starting net, then sets
/// `net` to net_start + (1/games) * sum_i (net_after_i - net_start), summed in game order.
/// Each game i uses `trainingSeed(seed, first_game + i)`, so the result is the same for any thread count.
pub fn trainBatch(net: *Net, seed: u64, first_game: usize, games: usize, lr: f64, diag: ?*Diag) u32 {
    std.debug.assert(games >= 1 and games <= parallel.max_threads);
    var units: [parallel.max_threads]BatchGame = undefined;
    parallel.map(BatchGame, units[0..games], BatchGameCtx{
        .net = net.*,
        .seed = seed,
        .first_game = first_game,
        .lr = lr,
        .want_diag = diag != null,
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

pub fn encode(net: *const Net) [file_len]u8 {
    var out: [file_len]u8 = undefined;
    @memcpy(out[0..8], file_magic);
    std.mem.writeInt(u32, out[8..12], file_version, .little);
    std.mem.writeInt(u32, out[12..16], endian_marker, .little);
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(net)), 0..) |w, i| {
        std.mem.writeInt(u64, out[header_len + i * 8 ..][0..8], @bitCast(w), .little);
    }
    return out;
}

pub fn parse(bytes: []const u8) !Net {
    if (bytes.len != file_len) return error.WrongWeightFileSize;
    if (!std.mem.eql(u8, bytes[0..8], file_magic)) return error.BadWeightMagic;
    if (std.mem.readInt(u32, bytes[8..12], .little) != file_version) return error.UnsupportedWeightVersion;
    if (std.mem.readInt(u32, bytes[12..16], .little) != endian_marker) return error.WrongWeightEndianness;
    var net: Net = undefined;
    for (std.mem.bytesAsSlice(f64, std.mem.asBytes(&net)), 0..) |*w, i| {
        w.* = @bitCast(std.mem.readInt(u64, bytes[header_len + i * 8 ..][0..8], .little));
    }
    return net;
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
    for (0..35) |_| diag.push(0, 1.0, 0);
    diag.finish();
    try std.testing.expectEqual(@as(u64, 10), diag.stats[0][0].count);
    try std.testing.expectEqual(@as(u64, 20), diag.stats[0][1].count);
    try std.testing.expectEqual(@as(u64, 5), diag.stats[0][2].count);
    try std.testing.expectApproxEqAbs(@as(f64, 35), diag.stats[0][0].abs_err + diag.stats[0][1].abs_err + diag.stats[0][2].abs_err, 1e-12);
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
