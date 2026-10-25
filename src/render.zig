const std = @import("std");
const core = @import("core");

/// Writes the board as a 4x4 text grid; empty cells show '.'.
pub fn writeBoard(w: *std.Io.Writer, state: core.GameState) std.Io.Writer.Error!void {
    for (state.board) |row| {
        for (row) |value| {
            if (value == 0) {
                try w.print("{s:>6}", .{"."});
            } else {
                try w.print("{d:>6}", .{value});
            }
        }
        try w.writeByte('\n');
    }
    try w.print("score: {d}\n", .{state.score});
}

test "writeBoard prints tiles, empty dots, and score" {
    const state: core.GameState = .{
        .board = .{
            .{ 2, 0, 0, 0 },
            .{ 0, 0, 0, 0 },
            .{ 0, 0, 0, 0 },
            .{ 0, 0, 0, 4 },
        },
        .score = 12,
    };

    var aw = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer aw.deinit();
    try writeBoard(&aw.writer, state);

    const expected =
        "     2     .     .     .\n" ++
        "     .     .     .     .\n" ++
        "     .     .     .     .\n" ++
        "     .     .     .     4\n" ++
        "score: 12\n";
    try std.testing.expectEqualStrings(expected, aw.written());
}
