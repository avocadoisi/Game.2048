const std = @import("std");
const core = @import("core");

pub fn main() !void {
    std.debug.print("Game.2048 CLI scaffold\n", .{});

    var prng = std.Random.DefaultPrng.init(0);
    _ = core.gameStart(prng.random());

    std.debug.print("Use zig build test to drive core with TDD.\n", .{});
}

test "main module compiles" {
    try std.testing.expect(true);
}
