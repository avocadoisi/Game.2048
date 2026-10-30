//! Converts an n-tuple table from the f32 format to the int16 quantized one (#129).
//! `zig build ntuple-quant -- ai/solver/ntuple_wide.bin ai/solver/ntuple_wide_q16.bin`

const std = @import("std");
const ntuple = @import("ntuple");

const usage =
    \\usage: ntuple-quant [in_table.bin] [out_table.q16.bin]
    \\
    \\  in_table   a table in the f32 format (`zig build ntuple` output)
    \\  out_table  where to write the int16 quantized format (129 MiB -> 64.6 MiB for `.wide`)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("{s}", .{usage});
        return;
    }
    // The pattern count comes from the file, so read the header first: `.full` and `.wide` differ.
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.gpa, .limited(ntuple.fileLen(ntuple.pattern_count) + 1));
    defer init.gpa.free(bytes);
    var net = try ntuple.parse(init.gpa, bytes);
    defer net.deinit(init.gpa);

    var q = try ntuple.quantize(init.gpa, &net);
    defer q.deinit(init.gpa);
    try ntuple.saveQuantized(&q, init.io, init.gpa, args[2]);

    std.debug.print("patterns={d} entries={d} f32={d:.1} MiB -> int16={d:.1} MiB ({s})\n", .{
        q.count,
        q.storage.len,
        @as(f64, @floatFromInt(net.storage.len * @sizeOf(f32))) / (1024 * 1024),
        @as(f64, @floatFromInt(ntuple.quantizedFileLen(q.count))) / (1024 * 1024),
        args[2],
    });
    for (0..q.count) |p| std.debug.print("  pattern {d}: step {d:.6}\n", .{ p, q.scales[p] });
}
