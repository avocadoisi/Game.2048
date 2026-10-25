const std = @import("std");

/// Upper bound on worker threads (issue #40: min(cpu count, 8)).
pub const max_threads = 8;

/// Fills `out[i] = play(ctx, i)` for every i, split into contiguous chunks over threads.
/// `play` must depend only on `(ctx, i)` (e.g. seed a PRNG from i), so the result is the same for any thread count.
/// If a thread cannot be spawned, its chunk runs on the calling thread instead.
pub fn map(comptime R: type, out: []R, ctx: anytype, comptime play: anytype) void {
    const Ctx = @TypeOf(ctx);
    const Work = struct {
        fn run(c: Ctx, slots: []R, first: usize) void {
            for (slots, 0..) |*slot, k| slot.* = play(c, first + k);
        }
    };
    const n = out.len;
    const cpus = std.Thread.getCpuCount() catch 1;
    const threads = @max(1, @min(cpus, max_threads, n));
    if (threads == 1) return Work.run(ctx, out, 0);

    var handles: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;
    const per = (n + threads - 1) / threads;
    var first: usize = 0;
    while (first < n) : (first += per) {
        const end = @min(n, first + per);
        if (std.Thread.spawn(.{}, Work.run, .{ ctx, out[first..end], first })) |h| {
            handles[spawned] = h;
            spawned += 1;
        } else |_| {
            Work.run(ctx, out[first..end], first);
        }
    }
    for (handles[0..spawned]) |h| h.join();
}

fn square(_: void, i: usize) u64 {
    return @as(u64, i) * i;
}

test "map gives the same result as a serial loop" {
    var parallel_out: [37]u64 = undefined;
    map(u64, &parallel_out, {}, square);
    for (parallel_out, 0..) |v, i| try std.testing.expectEqual(square({}, i), v);
}

test "map handles an empty slice" {
    var out: [0]u64 = .{};
    map(u64, &out, {}, square);
}
