const std = @import("std");
const solver = @import("solver");
const bench = @import("bench.zig");

const usage =
    \\usage: solve [runs] [games_per_run] [depth] [seed] [policy]
    \\
    \\  runs            number of runs, one PRNG seed each (default 10)
    \\  games_per_run   games per run (default 20)
    \\  depth           expectimax depth in player moves (default 2); for
    \\                  expectimax_id it is the time budget in ms, 0 = the default
    \\  seed            base PRNG seed (default 1)
    \\  policy          policy name, e.g. expectimax or expectimax_id (default expectimax)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const runs: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 10;
    const games_per_run: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 20;
    const depth: u8 = if (args.len > 3) try std.fmt.parseInt(u8, args[3], 10) else 2;
    const seed: u64 = if (args.len > 4) try std.fmt.parseInt(u64, args[4], 10) else 1;
    const policy: solver.Policy = if (args.len > 5) std.meta.stringToEnum(solver.Policy, args[5]) orelse {
        std.debug.print("unknown policy: {s}\n{s}", .{ args[5], usage });
        return;
    } else .expectimax;
    if (args.len > 6 or runs == 0 or games_per_run == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    const s = try bench.run(init.io, init.arena.allocator(), policy, runs, games_per_run, depth, seed);
    std.debug.print("runs={d} games/run={d} depth={d} seed={d} policy={s} games={d}\n", .{ runs, games_per_run, depth, seed, @tagName(policy), s.games });
    std.debug.print("avg score={d:.1} avg moves={d:.1} best tile={d} max depth={d}\n", .{ s.avg_score, s.avg_moves, s.best_tile, s.max_depth });
    std.debug.print("speed: {d:.2} games/s, {d:.0} moves/s, elapsed {d:.1} s\n", .{
        s.gamesPerSec(),
        s.movesPerSec(),
        s.elapsedSec(),
    });
}
