//! Converts the value net from the f64 format to the f32 one (#348) and reports what that costs and
//! what it buys: per-leaf time, value agreement (including the near-tie positions where a small
//! change can pick a different move), and greedy play at the player level.
//! `zig build nn-f32 -- ai/solver/nn_weights.bin ai/solver/nn_weights.f32.bin [games] [seed]`

const std = @import("std");
const core = @import("core");
const nn = @import("nn");

const usage =
    \\usage: nn-f32 [in_net.bin] [out_net.f32.bin] [games=20] [seed=1]
    \\
    \\  in_net    a net in the f64 format (`zig build nn` output, ai/solver/nn_weights.bin)
    \\  out_net   where to write the f32 format (296,472 B -> 148,236 B)
    \\  games     games of greedy play to sample boards from, for the agreement and timing tables
    \\  seed      rng seed for those games
    \\
;

/// A move counts as a near tie when its value is within this of the best one: f32 rounding is far
/// below it, so the two nets should still agree on the move if the only difference is precision.
const near_tie: f64 = 1e-3;

const Summary = struct {
    count: u64 = 0,
    sum: f64 = 0,
    max: f64 = 0,

    fn add(self: *Summary, err: f64) void {
        self.count += 1;
        self.sum += err;
        self.max = @max(self.max, err);
    }

    fn mean(self: Summary) f64 {
        return if (self.count == 0) 0 else self.sum / @as(f64, @floatFromInt(self.count));
    }
};

const PlaySummary = struct {
    score: u64 = 0,
    best_tile: core.Tile = 0,

    fn meanScore(self: PlaySummary, games: usize) f64 {
        return if (games == 0) 0 else @as(f64, @floatFromInt(self.score)) / @as(f64, @floatFromInt(games));
    }
};

fn playAll(vfn: nn.Value, rng: std.Random, games: usize) PlaySummary {
    var out = PlaySummary{};
    for (0..games) |_| {
        const result = nn.playValue(vfn, rng, .greedy);
        out.score += result.score;
        out.best_tile = @max(out.best_tile, result.max_tile);
    }
    return out;
}

/// Nanoseconds per `forward`, both nets on the same board set, alternating so a drifting machine
/// affects both sides the same way.
fn timeForward(init: std.process.Init, net: *const nn.Net, net32: *const nn.Net32, boards: []const core.Board) [2]f64 {
    var cache: nn.Cache = .{};
    var sink: f64 = 0;
    const rounds = 200;
    var t64: i96 = 0;
    var t32: i96 = 0;
    for (0..rounds) |_| {
        const a = std.Io.Clock.awake.now(init.io);
        for (boards) |board| sink += nn.forward(net, board, &cache);
        t64 += a.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
        const b = std.Io.Clock.awake.now(init.io);
        for (boards) |board| sink += nn.forwardF32(net32, board);
        t32 += b.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
    }
    std.mem.doNotOptimizeAway(&sink);
    const calls: f64 = @floatFromInt(rounds * boards.len);
    return .{
        @as(f64, @floatFromInt(t64)) / calls,
        @as(f64, @floatFromInt(t32)) / calls,
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or args.len > 5) {
        std.debug.print("{s}", .{usage});
        return;
    }
    const games = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 20;
    const seed = if (args.len > 4) try std.fmt.parseInt(u64, args[4], 10) else 1;

    var net: nn.Net = undefined;
    try nn.load(&net, init.io, init.gpa, args[1]);
    const net32 = nn.toF32(&net);
    try nn.saveF32(&net32, init.io, args[2]);

    std.debug.print("{s}: f64 {d} B -> {s}: f32 {d} B ({d:.1}% of f64)\n", .{
        args[1],
        nn.file_len,
        args[2],
        nn.f32_file_len,
        100.0 * @as(f64, @floatFromInt(nn.f32_file_len)) / @as(f64, @floatFromInt(nn.file_len)),
    });

    // Read the file back: proves the payload parses and the bytes are the weights we expect.
    var back: nn.Net32 = undefined;
    try nn.loadF32(&back, init.io, init.gpa, args[2]);
    const f64_bytes = std.mem.bytesAsSlice(f64, std.mem.asBytes(&net)[0..@sizeOf(nn.Net)]);
    const f32_bytes = std.mem.bytesAsSlice(f32, std.mem.asBytes(&back)[0..@sizeOf(nn.Net32)]);
    var worst: f64 = 0;
    var wrong: usize = 0;
    for (f64_bytes, f32_bytes) |a, b| {
        const narrowed: f64 = @floatCast(b);
        if (narrowed != @as(f64, @floatCast(@as(f32, @floatCast(a))))) wrong += 1;
        worst = @max(worst, @abs(a - narrowed));
    }
    std.debug.print("round trip {d} weights: {d} mismatched, worst |w64 - w32| {e:.3}\n", .{ f64_bytes.len, wrong, worst });

    // Value agreement along greedy play, plus the near-tie count: a move within `near_tie` of the
    // best one is where a precision change could flip the choice.
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var cache: nn.Cache = .{};
    var boards: usize = 0;
    var values = Summary{};
    var ties: usize = 0;
    var ties_flipped: usize = 0;
    var moves: usize = 0;
    var moves_flipped: usize = 0;
    var tiles: core.Tile = 0;
    var sample = std.ArrayList(core.Board).empty;
    defer sample.deinit(init.gpa);
    for (0..games) |_| {
        var state = core.GameState.gameStart(rng);
        while (!state.isGameOver()) {
            var best: ?core.Direction = null;
            var best_value: f64 = -std.math.inf(f64);
            var second: f64 = -std.math.inf(f64);
            for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
                var next = state;
                const moved = next.slide(d);
                if (!moved.changed) continue;
                const value = @as(f64, @floatFromInt(moved.score_delta)) / nn.reward_scale +
                    nn.forward(&net, next.board, &cache);
                if (best == null or value > best_value) {
                    second = best_value;
                    best = d;
                    best_value = value;
                } else if (value > second) second = value;
            }
            const dir = best orelse break;
            // The policy's own choice with each net, from the same position.
            const pick64 = nn.bestMoveValue(.{ .single = &net }, state, .greedy);
            const pick32 = nn.bestMoveValue(.{ .f32 = &net32 }, state, .greedy);
            moves += 1;
            if (pick64 != pick32) moves_flipped += 1;
            if (best != null and second > -std.math.inf(f64) and best_value - second < near_tie) {
                ties += 1;
                if (pick64 != pick32) ties_flipped += 1;
            }
            _ = state.move(dir, rng);
            tiles = @max(tiles, core.maxTile(state.board));

            const want = nn.forward(&net, state.board, &cache);
            values.add(@abs(nn.forwardF32(&net32, state.board) - want));
            if (sample.items.len < 16 or boards % 97 == 0) try sample.append(init.gpa, state.board);
            boards += 1;
        }
    }
    std.debug.print("values over {d} boards ({d} games, max tile {d}): mean |dV| {e:.3} max {e:.3}\n", .{
        boards, games, tiles, values.mean(), values.max,
    });
    std.debug.print("moves where f32 picked a different move: {d}/{d}", .{ moves_flipped, moves });
    std.debug.print(" (near ties {d}, flipped {d})\n", .{ ties, ties_flipped });

    // Per-leaf cost on the sampled boards.
    const ns = timeForward(init, &net, &net32, sample.items);
    std.debug.print("forward on {d} boards x 200 rounds: f64 {d:.1} ns/leaf, f32 {d:.1} ns/leaf ({d:.2}x)\n", .{
        sample.items.len, ns[0], ns[1], ns[0] / ns[1],
    });

    // Player level: the same greedy policy over each `Value`, one rng reseeded per side.
    var prng64 = std.Random.DefaultPrng.init(seed);
    const play64 = playAll(.{ .single = &net }, prng64.random(), games);
    var prng32 = std.Random.DefaultPrng.init(seed);
    const play32 = playAll(.{ .f32 = &net32 }, prng32.random(), games);
    std.debug.print("greedy {d} games: f64 mean score {d:.0} best tile {d} | f32 mean score {d:.0} best tile {d}\n", .{
        games,
        play64.meanScore(games),
        play64.best_tile,
        play32.meanScore(games),
        play32.best_tile,
    });
}
