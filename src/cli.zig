const std = @import("std");
const core = @import("core");
const render = @import("render.zig");

pub const Command = union(enum) {
    move: core.Direction,
    new_game,
    quit,
    unknown,
};

/// Accepts WASD, arrow-word names, and short commands (n = new, q = quit).
pub fn parseCommand(input: []const u8) Command {
    const line = std.mem.trim(u8, input, " \t\r\n");
    if (line.len == 0) return .unknown;

    if (std.ascii.eqlIgnoreCase(line, "w")) return .{ .move = .up };
    if (std.ascii.eqlIgnoreCase(line, "a")) return .{ .move = .left };
    if (std.ascii.eqlIgnoreCase(line, "s")) return .{ .move = .down };
    if (std.ascii.eqlIgnoreCase(line, "d")) return .{ .move = .right };
    if (std.ascii.eqlIgnoreCase(line, "n") or std.ascii.eqlIgnoreCase(line, "new")) return .new_game;
    if (std.ascii.eqlIgnoreCase(line, "q") or std.ascii.eqlIgnoreCase(line, "quit")) return .quit;

    if (core.Direction.parse(line)) |dir| return .{ .move = dir };
    return .unknown;
}

/// Interactive loop. Reads commands until EOF or quit.
pub fn run(
    input: *std.Io.Reader,
    output: *std.Io.Writer,
    prng: *std.Random.DefaultPrng,
) !void {
    var state = core.GameState.gameStart(prng.random());
    try output.writeAll("2048 CLI — w/a/s/d or up/left/down/right, n = new game, q = quit\n");

    while (true) {
        try render.writeBoard(output, state);
        if (state.isGameOver()) {
            try output.writeAll("game over. type n for a new game or q to quit\n");
        }
        try output.writeAll("> ");
        try output.flush();

        const line = (try input.takeDelimiter('\n')) orelse break;
        switch (parseCommand(line)) {
            .quit => break,
            .new_game => state = core.GameState.gameStart(prng.random()),
            .move => |dir| {
                const result = state.move(dir, prng.random());
                if (!result.changed) try output.writeAll("no change\n");
            },
            .unknown => try output.writeAll("unknown command\n"),
        }
    }
    try output.flush();
}

test "parseCommand maps WASD, words, and short commands" {
    try std.testing.expectEqual(core.Direction.up, parseCommand("w").move);
    try std.testing.expectEqual(core.Direction.left, parseCommand(" A \n").move);
    try std.testing.expectEqual(core.Direction.down, parseCommand("down").move);
    try std.testing.expectEqual(core.Direction.right, parseCommand("d").move);
    try std.testing.expectEqual(Command.new_game, parseCommand("n"));
    try std.testing.expectEqual(Command.quit, parseCommand("quit"));
    try std.testing.expectEqual(Command.unknown, parseCommand("x"));
    try std.testing.expectEqual(Command.unknown, parseCommand(""));
}

test "run plays scripted moves then quits" {
    var input = std.Io.Reader.fixed("a\nq\n");
    var aw = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer aw.deinit();
    var prng = std.Random.DefaultPrng.init(1);

    try run(&input, &aw.writer, &prng);

    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "score:") != null);
}

test "run reports unknown commands and exits on EOF" {
    var input = std.Io.Reader.fixed("zzz\n");
    var aw = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer aw.deinit();
    var prng = std.Random.DefaultPrng.init(1);

    try run(&input, &aw.writer, &prng);

    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "unknown command") != null);
}
