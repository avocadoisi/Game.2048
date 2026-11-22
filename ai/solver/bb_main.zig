//! `zig build bb -- [runs] [games_per_run] [depth] [seed] [leaf]`: the packed bitboard expectimax
//! (#349). Same games and seeds as `zig build solve`, so its numbers line up with `expectimax_tt`.

const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const bitboard = @import("bitboard");
const parallel = @import("parallel");

const usage =
    \\usage: bb [runs] [games_per_run] [depth] [seed] [leaf]
    \\
    \\  runs            number of runs, one PRNG seed each (default 10)
    \\  games_per_run   games per run (default 20)
    \\  depth           search depth in player moves (default 3)
    \\  seed            base PRNG seed (default 9000)
    \\  leaf            `heur` (row heuristic, default) or `net` (shipped 128/32 net)
    \\
;

const Leaf = enum { heur, net };

const tt_len = 1 << 16;

const RunResult = struct {
    score: u64 = 0,
    moves: u64 = 0,
    best_tile: u32 = 0,
    nodes: u64 = 0,
    games: u32 = 0,
    reach_4096: u32 = 0,
    reach_8192: u32 = 0,
    reach_16384: u32 = 0,
    dead_below_2048: u32 = 0,
    elapsed_ns: i96 = 0,
};

const Ctx = struct {
    depth: u8,
    leaf: Leaf,
    games: u32,
    seed: u64,
    io: std.Io,
    tt: []bitboard.TTEntry,
};

fn playRun(c: Ctx, r: usize) RunResult {
    const start = std.Io.Clock.awake.now(c.io);
    var prng = std.Random.DefaultPrng.init(c.seed + r);
    var s: bitboard.Search = .{ .tt = c.tt[r * tt_len ..][0..tt_len], .depth_limit = c.depth };
    if (c.leaf == .net) s.useNet(solver.shippedNet());
    s.clearTable();
    s.resetNodes();
    var res: RunResult = .{};
    for (0..c.games) |_| {
        var state = core.GameState.gameStart(prng.random());
        var moves: u32 = 0;
        while (s.bestMove(bitboard.pack(state.board))) |d| {
            _ = state.move(d, prng.random());
            moves += 1;
        }
        res.score += state.score;
        res.moves += moves;
        res.games += 1;
        const tile = core.maxTile(state.board);
        res.best_tile = @max(res.best_tile, tile);
        if (tile >= 4096) res.reach_4096 += 1;
        if (tile >= 8192) res.reach_8192 += 1;
        if (tile >= 16384) res.reach_16384 += 1;
        if (tile < 2048) res.dead_below_2048 += 1;
    }
    res.nodes = s.nodes;
    res.elapsed_ns = start.durationTo(std.Io.Clock.awake.now(c.io)).toNanoseconds();
    return res;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    const runs: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 10;
    const games_per_run: u32 = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 20;
    const depth: u8 = if (args.len > 3) try std.fmt.parseInt(u8, args[3], 10) else 3;
    const seed: u64 = if (args.len > 4) try std.fmt.parseInt(u64, args[4], 10) else 9000;
    const leaf: Leaf = if (args.len > 5) std.meta.stringToEnum(Leaf, args[5]) orelse {
        std.debug.print("{s}", .{usage});
        return;
    } else .heur;
    if (args.len > 6 or runs == 0 or games_per_run == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    bitboard.init();
    const tt = try gpa.alloc(bitboard.TTEntry, @as(usize, runs) * tt_len);
    const out = try gpa.alloc(RunResult, runs);
    const start = std.Io.Clock.awake.now(init.io);
    parallel.map(RunResult, out, Ctx{ .depth = depth, .leaf = leaf, .games = games_per_run, .seed = seed, .io = init.io, .tt = tt }, playRun);

    var score: u64 = 0;
    var moves: u64 = 0;
    var nodes: u64 = 0;
    var games: u32 = 0;
    var best_tile: u32 = 0;
    var serial_ns: i96 = 0;
    var reach_4096: u32 = 0;
    var reach_8192: u32 = 0;
    var reach_16384: u32 = 0;
    var dead: u32 = 0;
    for (out) |r| {
        score += r.score;
        moves += r.moves;
        nodes += r.nodes;
        games += r.games;
        best_tile = @max(best_tile, r.best_tile);
        serial_ns += r.elapsed_ns;
        reach_4096 += r.reach_4096;
        reach_8192 += r.reach_8192;
        reach_16384 += r.reach_16384;
        dead += r.dead_below_2048;
    }
    const wall_s = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e9;
    const ms_per_move = @as(f64, @floatFromInt(serial_ns)) / 1e6 / @as(f64, @floatFromInt(@max(moves, 1)));
    const nodes_per_s = @as(f64, @floatFromInt(nodes)) * 1e9 / @as(f64, @floatFromInt(@max(serial_ns, 1)));

    std.debug.print("runs={d} games/run={d} depth={d} seed={d} policy=packed+{s} games={d}\n", .{ runs, games_per_run, depth, seed, @tagName(leaf), games });
    std.debug.print("avg score={d:.1} avg moves={d:.1} best tile={d}\n", .{
        @as(f64, @floatFromInt(score)) / @as(f64, @floatFromInt(games)),
        @as(f64, @floatFromInt(moves)) / @as(f64, @floatFromInt(games)),
        best_tile,
    });
    std.debug.print("tiles: 4096 {d}/{d}  8192 {d}/{d}  16384 {d}/{d}  dead<2048 {d}/{d}\n", .{
        reach_4096, games, reach_8192, games, reach_16384, games, dead, games,
    });
    std.debug.print("speed: {d:.0} ms/move, {d:.0} leaf evals/s, {d:.1} games/s, wall {d:.1} s\n", .{
        ms_per_move,
        nodes_per_s,
        @as(f64, @floatFromInt(games)) / wall_s,
        wall_s,
    });
}
