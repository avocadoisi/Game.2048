const std = @import("std");
const solver = @import("solver");

const usage =
    \\usage: solve [games] [depth] [seed]
    \\
    \\  games   number of games to play (default 100)
    \\  depth   expectimax depth in player moves (default 2)
    \\  seed    PRNG seed for reproducible runs (default 1)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const games: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 100;
    const depth: u8 = if (args.len > 2) try std.fmt.parseInt(u8, args[2], 10) else 2;
    const seed: u64 = if (args.len > 3) try std.fmt.parseInt(u64, args[3], 10) else 1;
    if (args.len > 4 or games == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    var prng = std.Random.DefaultPrng.init(seed);
    var total_score: u64 = 0;
    var total_moves: u64 = 0;
    var best_tile: u32 = 0;
    // index = log2(max tile); 2^31 is far beyond any reachable tile
    var tile_counts: [32]u32 = @splat(0);

    for (0..games) |_| {
        const result = solver.playGame(prng.random(), depth);
        total_score += result.score;
        total_moves += result.moves;
        best_tile = @max(best_tile, result.max_tile);
        tile_counts[std.math.log2_int(u32, result.max_tile)] += 1;
    }

    std.debug.print("games={d} depth={d} seed={d}\n", .{ games, depth, seed });
    std.debug.print("avg score={d:.1} avg moves={d:.1} best tile={d}\n", .{
        @as(f64, @floatFromInt(total_score)) / @as(f64, @floatFromInt(games)),
        @as(f64, @floatFromInt(total_moves)) / @as(f64, @floatFromInt(games)),
        best_tile,
    });
    for (tile_counts, 0..) |count, exp| {
        if (count > 0) std.debug.print("  max tile {d}: {d} games\n", .{ @as(u32, 1) << @intCast(exp), count });
    }
}
