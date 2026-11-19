const std = @import("std");
const core = @import("core");
const nn = @import("nn");

pub fn main(init: std.process.Init) !void {
    const net = try nn.parse(@embedFile("nn_weights.bin"));
    var board: core.Board = .{
        .{ 1024, 512, 256, 128 }, .{ 64, 32, 16, 8 },
        .{ 4, 8, 16, 0 },         .{ 2, 0, 0, 0 },
    };
    var cache: nn.Cache = .{};
    var sum: f64 = 0;
    const iterations: usize = 1_000_000;
    const start = std.Io.Clock.awake.now(init.io);
    for (0..iterations) |i| {
        board[3][3] = @as(u32, 2) << @as(u5, @intCast(i % 12));
        sum += nn.forward(&net, board, &cache);
    }
    const ns = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
    std.debug.print("forward: {d} iterations, {d:.2} ns/call, sum={d:.6}\n", .{ iterations, @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iterations)), sum });
}
