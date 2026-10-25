const std = @import("std");
const core = @import("core");

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

/// TD(0) on afterstates over one self-play game (greedy 1-ply moves, fixed rng).
/// Target for the previous afterstate = its scaled reward + V(next afterstate); 0 after game over.
pub fn trainGame(net: *Net, rng: std.Random, lr: f64) u32 {
    var state = GameState.gameStart(rng);
    var pending: ?Pick = null;
    var cache: Cache = .{};
    while (pickMove(net, state)) |pick| {
        if (pending) |p| {
            _ = forward(net, p.after, &cache);
            train(net, &cache, p.reward + pick.value, lr);
        }
        pending = pick;
        _ = state.move(pick.dir, rng);
    }
    if (pending) |p| {
        _ = forward(net, p.after, &cache);
        train(net, &cache, p.reward, lr);
    }
    return state.score;
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
    _ = trainGame(&a, rng_a.random(), 0.001);
    _ = trainGame(&b, rng_b.random(), 0.001);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
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
