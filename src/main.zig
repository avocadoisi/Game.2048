const std = @import("std");
const core = @import("core");
const cli = @import("cli.zig");
const server = @import("server.zig");

const usage =
    \\usage: game-2048 [command]
    \\
    \\  play            interactive terminal game (default)
    \\  serve [port]    HTTP server with browser UI and JSON API (default port 8080)
    \\  help            show this message
    \\
;

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    const command = if (args.len > 1) args[1] else "play";

    if (std.mem.eql(u8, command, "play")) {
        var seed = std.Random.DefaultPrng.init(try seedFromOs());
        var in_buf: [4096]u8 = undefined;
        var out_buf: [4096]u8 = undefined;
        var stdin_reader = std.fs.File.stdin().reader(&in_buf);
        var stdout_writer = std.fs.File.stdout().writer(&out_buf);
        try cli.run(&stdin_reader.interface, &stdout_writer.interface, &seed);
    } else if (std.mem.eql(u8, command, "serve")) {
        const port: u16 = if (args.len > 2) try std.fmt.parseInt(u16, args[2], 10) else 8080;
        try server.serve(gpa, port, try seedFromOs());
    } else {
        std.debug.print("{s}", .{usage});
    }
}

fn seedFromOs() !u64 {
    var buf: [8]u8 = undefined;
    try std.posix.getrandom(&buf);
    return std.mem.readInt(u64, &buf, .little);
}

test "main module references core" {
    _ = core.BoardSize;
}

test {
    std.testing.refAllDecls(@import("render.zig"));
    std.testing.refAllDecls(@import("cli.zig"));
    std.testing.refAllDecls(@import("server.zig"));
}
