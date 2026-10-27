const std = @import("std");
const core = @import("core");
const solver = @import("solver");
const nn = @import("nn");
const ntuple = @import("ntuple");

const Board = core.Board;

const usage =
    \\usage: probe [ntuple_table_path]
    \\
    \\Scores the same tile multiset placed along different path shapes with the hand-made snake
    \\evaluation, the embedded NN, and the TD-trained n-tuple table (issue #115), then prints the
    \\per-board values with their ranks and the Spearman rank correlation between the scorers.
    \\  ntuple_table_path  n-tuple table to load (default ai/solver/ntuple_weights.bin)
    \\
;

/// Every probe board holds this multiset, so the scorers only see where the tiles sit (issue #115).
const tiles = [_]core.Tile{ 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4 };

/// One probe board: `tiles` placed along `layout` / `anchor`, or a literal `board` when set.
const BoardSpec = struct {
    name: []const u8,
    layout: solver.Layout = .snake,
    anchor: u8 = 0,
    board: ?Board = null,
};

const specs = [_]BoardSpec{
    .{ .name = "snake", .layout = .snake },
    .{ .name = "column_snake", .layout = .column_snake },
    .{ .name = "spiral", .layout = .spiral },
    .{ .name = "flex", .layout = .flex },
    .{ .name = "block2", .layout = .block2 },
    .{ .name = "snake anchor 7", .layout = .snake, .anchor = 7 },
    // worst case: the large tiles are scattered, so no path shape holds them in order
    .{ .name = "scrambled", .board = .{
        .{ 2048, 0, 512, 0 },
        .{ 8, 128, 0, 64 },
        .{ 1024, 0, 256, 0 },
        .{ 4, 32, 0, 16 },
    } },
};

const max_boards = 8;
comptime {
    if (specs.len > max_boards) @compileError("raise max_boards");
}

const scorer_names = [_][]const u8{ "snake", "nn", "ntuple" };

/// Scores of one board, one per `scorer_names` column; NaN means that scorer was not run.
const Row = struct {
    name: []const u8,
    values: [scorer_names.len]f64,
};

fn boardFor(spec: BoardSpec) Board {
    if (spec.board) |b| return b;
    var path: [core.BoardCellCount]core.Position = undefined;
    solver.pathFor(spec.layout, spec.anchor, &path);
    var board = std.mem.zeroes(Board);
    for (tiles, 0..) |value, i| board[path[i].row][path[i].col] = value;
    return board;
}

/// Score every board; `net` and `tuple` are optional so a test can skip the 68 MB n-tuple table.
fn scoreBoards(net: ?*const nn.Net, tuple: ?*const ntuple.TupleNet, rows: []Row) void {
    var cache: nn.Cache = .{};
    for (specs[0..rows.len], rows) |spec, *row| {
        const board = boardFor(spec);
        const nan = std.math.nan(f64);
        row.* = .{ .name = spec.name, .values = .{ solver.evaluateW(board, solver.baseline), nan, nan } };
        if (net) |n| row.values[1] = nn.forward(n, board, &cache);
        if (tuple) |t| row.values[2] = t.eval(board);
    }
}

/// Competition ranks, 1 = largest value; tied values share the average of their positions.
/// A column holding NaN (scorer not run) gets NaN ranks.
fn ranks(values: []const f64, out: []f64) void {
    for (values) |v| {
        if (std.math.isNan(v)) {
            for (out) |*o| o.* = std.math.nan(f64);
            return;
        }
    }
    const n = values.len;
    var idx: [max_boards]usize = undefined;
    for (0..n) |i| idx[i] = i;
    // insertion sort, descending value
    for (1..n) |i| {
        var j = i;
        while (j > 0 and values[idx[j - 1]] < values[idx[j]]) : (j -= 1) std.mem.swap(usize, &idx[j - 1], &idx[j]);
    }
    var i: usize = 0;
    while (i < n) {
        var k = i + 1;
        while (k < n and values[idx[k]] == values[idx[i]]) : (k += 1) {}
        const avg = (@as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(k))) / 2;
        for (idx[i..k]) |m| out[m] = avg;
        i = k;
    }
}

/// Spearman rank correlation of two scorer columns: 1 = same preference order, -1 = reversed.
/// Pearson over the ranks, which handles ties and equals 1 - 6·Σd²/(n(n²-1)) without them.
fn spearman(a: []const f64, b: []const f64) f64 {
    var ra: [max_boards]f64 = undefined;
    var rb: [max_boards]f64 = undefined;
    ranks(a, ra[0..a.len]);
    ranks(b, rb[0..b.len]);
    const nan = std.math.nan(f64);
    if (std.math.isNan(ra[0]) or std.math.isNan(rb[0])) return nan;
    // ranks are a permutation of 1..n, so the column mean is (n+1)/2 even with ties
    const mean = (@as(f64, @floatFromInt(a.len)) + 1) / 2;
    var num: f64 = 0;
    var sa: f64 = 0;
    var sb: f64 = 0;
    for (ra[0..a.len], rb[0..b.len]) |x, y| {
        num += (x - mean) * (y - mean);
        sa += (x - mean) * (x - mean);
        sb += (y - mean) * (y - mean);
    }
    if (sa == 0 or sb == 0) return nan;
    return num / @sqrt(sa * sb);
}

fn renderTable(w: *std.Io.Writer, rows: []const Row) !void {
    var cols: [scorer_names.len][max_boards]f64 = undefined;
    var rank_cols: [scorer_names.len][max_boards]f64 = undefined;
    for (0..scorer_names.len) |s| {
        for (rows, 0..) |row, i| cols[s][i] = row.values[s];
        ranks(cols[s][0..rows.len], &rank_cols[s]);
    }

    try w.print("{s:<16}", .{"board"});
    for (scorer_names) |name| try w.print(" {s:>14} {s:>5}", .{ name, "rank" });
    try w.writeAll("\n");
    for (rows, 0..) |row, i| {
        try w.print("{s:<16}", .{row.name});
        for (0..scorer_names.len) |s| {
            const value = row.values[s];
            if (std.math.isNan(value)) {
                try w.print(" {s:>14} {s:>5}", .{ "n/a", "-" });
            } else {
                try w.print(" {d:>14.4} {d:>5}", .{ value, rank_cols[s][i] });
            }
        }
        try w.writeAll("\n");
    }

    try w.writeAll("\nspearman rank correlation (1 = same preference order, -1 = reversed, 0 = unrelated)\n");
    const pairs = [_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } };
    for (pairs) |p| {
        try w.print("  {s:<6} vs {s:<6} {d:>7.3}\n", .{
            scorer_names[p[0]],                                         scorer_names[p[1]],
            spearman(cols[p[0]][0..rows.len], cols[p[1]][0..rows.len]),
        });
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    const table_path = if (args.len > 1) args[1] else "ai/solver/ntuple_weights.bin";
    if (args.len > 2) {
        std.debug.print("{s}", .{usage});
        return;
    }

    const net = try nn.parse(@embedFile("nn_weights.bin"));
    var tuple = try ntuple.TupleNet.init(gpa);
    try ntuple.load(&tuple, init.io, gpa, table_path);

    var rows: [specs.len]Row = undefined;
    scoreBoards(&net, &tuple, &rows);

    var buf: [8192]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    try renderTable(&out.interface, &rows);
    try out.interface.flush();
}

test "probe boards hold the same tile multiset" {
    var want = tiles;
    std.mem.sort(core.Tile, &want, {}, std.sort.asc(core.Tile));
    for (specs) |spec| {
        const board = boardFor(spec);
        var got: [tiles.len]core.Tile = undefined;
        var n: usize = 0;
        for (board) |line| {
            for (line) |value| {
                if (value == 0) continue;
                got[n] = value;
                n += 1;
            }
        }
        try std.testing.expectEqual(tiles.len, n);
        std.mem.sort(core.Tile, &got, {}, std.sort.asc(core.Tile));
        try std.testing.expectEqualSlices(core.Tile, &want, &got);
    }
}

test "probe table is deterministic" {
    const net = try nn.parse(@embedFile("nn_weights.bin"));
    var first: [specs.len]Row = undefined;
    var second: [specs.len]Row = undefined;
    scoreBoards(&net, null, &first);
    scoreBoards(&net, null, &second);

    var buf1: [8192]u8 = undefined;
    var w1 = std.Io.Writer.fixed(&buf1);
    try renderTable(&w1, &first);
    var buf2: [8192]u8 = undefined;
    var w2 = std.Io.Writer.fixed(&buf2);
    try renderTable(&w2, &second);
    try std.testing.expectEqualStrings(w1.buffered(), w2.buffered());
}

test "spearman matches known orders" {
    try std.testing.expectEqual(@as(f64, 1), spearman(&.{ 1, 2, 3, 4 }, &.{ 10, 20, 30, 40 }));
    try std.testing.expectEqual(@as(f64, -1), spearman(&.{ 1, 2, 3, 4 }, &.{ 40, 30, 20, 10 }));
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), spearman(&.{ 1, 2, 3, 4 }, &.{ 1, 3, 2, 4 }), 1e-12);
}

test "n-tuple indices are dihedral invariant (why the n-tuple column ties)" {
    for (specs) |spec| {
        const board = boardFor(spec);
        for (0..8) |t| {
            const rotated = ntuple.transformBoard(board, t);
            try std.testing.expectEqualSlices(u32, &ntuple.indices(board), &ntuple.indices(rotated));
        }
    }
}
