//! Converts the value net from the f32 format to the int16 quantized one and reports what that costs
//! in accuracy (#170). The quantized file is a shipping artefact (37,170 B instead of 148,504 B), so it
//! is written wherever the caller points and never committed.
//! `zig build nn-quant -- ai/solver/nn_weights.bin /tmp/nn_weights_q.bin [games] [seed]`

const std = @import("std");
const core = @import("core");
const nn = @import("nn");

const usage =
    \\usage: nn-quant [in_net.bin] [out_net.q16.bin] [games=20] [seed=1]
    \\
    \\  in_net    a net in the f32 format (`zig build nn` output, ai/solver/nn_weights.bin)
    \\  out_net   where to write the int16 quantized format (148,504 B -> 37,170 B)
    \\  games     games of greedy f32 play to sample boards from, for the accuracy table
    \\  seed      rng seed for those games
    \\
;

/// Weight blocks via `nn.quant_blocks`; `flat` is the same view of a `Net` that `nn.quantize` uses.
fn flat(net: *nn.Net) []f64 {
    return std.mem.bytesAsSlice(f64, std.mem.asBytes(net)[0..@sizeOf(nn.Net)]);
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
    const q = nn.quantize(&net);
    try nn.saveQuantized(&q, init.io, args[2]);

    std.debug.print("{s}: f32 {d} B -> {s}: int16 {d} B ({d} B, {d:.1}% of f32)\n", .{
        args[1],
        nn.file_len,
        args[2],
        nn.quant_file_len,
        nn.quant_file_len - @as(i64, @intCast(nn.file_len)),
        100.0 * @as(f64, @floatFromInt(nn.quant_file_len)) / @as(f64, @floatFromInt(nn.file_len)),
    });

    // Read the file back: this also proves the payload parses and every weight is within its step.
    var back: nn.Quantized = undefined;
    try nn.loadQuantized(&back, init.io, init.gpa, args[2]);

    const before = flat(&net);
    std.debug.print("blocks", .{});
    for (0..nn.quant_block_count) |b| std.debug.print(" {d}:{e:.3}", .{ b, back.steps[b] });
    std.debug.print("\nactivations a1 {e:.3} a2 {e:.3}", .{ back.a_steps[0], back.a_steps[1] });

    var qnet: nn.Net = undefined;
    nn.dequantize(&back, &qnet);
    const after = flat(&qnet);
    var worst_abs: f64 = 0;
    var worst_ratio: f64 = 0;
    var worst_block: usize = 0;
    for (nn.quant_blocks, 0..) |block, b| {
        const step = back.steps[b];
        for (block.offset..block.offset + block.len) |i| {
            const delta = @abs(before[i] - after[i]);
            if (delta > worst_abs) worst_abs = delta;
            if (step > 0 and delta / step > worst_ratio) {
                worst_ratio = delta / step;
                worst_block = b;
            }
        }
    }
    std.debug.print("; round trip {d} weights, worst |w - q*step| {e:.3} (block {d}, {d:.3} of its step)\n", .{
        nn.weight_count, worst_abs, worst_block, worst_ratio,
    });

    // Value agreement along greedy f32 play: weights-only (quantize then dequantize, same `forward`)
    // and the real thing (i16 activations via `forwardQuantized`).
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var cache: nn.Cache = .{};
    var boards: usize = 0;
    var weights_only = Summary{};
    var quantized = Summary{};
    var tiles: core.Tile = 0;
    for (0..games) |_| {
        var state = core.GameState.gameStart(rng);
        while (!state.isGameOver()) {
            var best: ?core.Direction = null;
            var best_value: f64 = -std.math.inf(f64);
            for ([_]core.Direction{ .up, .left, .down, .right }) |d| {
                var next = state;
                const moved = next.slide(d);
                if (!moved.changed) continue;
                const value = @as(f64, @floatFromInt(moved.score_delta)) / nn.reward_scale +
                    nn.forward(&net, next.board, &cache);
                if (best == null or value > best_value) {
                    best = d;
                    best_value = value;
                }
            }
            const dir = best orelse break;
            _ = state.move(dir, rng);
            tiles = @max(tiles, core.maxTile(state.board));

            const want = nn.forward(&net, state.board, &cache);
            weights_only.add(@abs(nn.forward(&qnet, state.board, &cache) - want));
            quantized.add(@abs(nn.forwardQuantized(&back, state.board) - want));
            boards += 1;
        }
    }
    std.debug.print("values over {d} boards ({d} games, max tile {d}):\n", .{ boards, games, tiles });
    std.debug.print("  weights only        mean |dV| {e:.3} max {e:.3}\n", .{ weights_only.mean(), weights_only.max });
    std.debug.print("  weights + int16 act mean |dV| {e:.3} max {e:.3}\n", .{ quantized.mean(), quantized.max });

    // Player-level check of the opt-in path (#170): the same policy, `Value.quantized` instead of
    // `Value.single`, through the normal play loop.
    const f32_play = playAll(.{ .single = &net }, rng, games);
    const q_play = playAll(.{ .quantized = &back }, rng, games);
    std.debug.print("greedy {d} games: f32 mean score {d:.0} best tile {d} | int16 mean score {d:.0} best tile {d}\n", .{
        games,
        f32_play.meanScore(games),
        f32_play.best_tile,
        q_play.meanScore(games),
        q_play.best_tile,
    });
}

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
