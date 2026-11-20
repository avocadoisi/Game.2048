const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");

/// Operation budgets for both learned policies (#336). A library default of no cap is retained;
/// the command-line runners inject these budgets for comparable training and evaluation runs.
pub const RunPhase = enum { train, evaluate };
pub const promising_partial_score: u32 = 100_000;
pub const promising_partial_tile: core.Tile = 8192;

/// The WASM gameplay wrapper has no clock and always calls the unlimited library APIs.
const clock_available = builtin.target.os.tag != .freestanding;

pub fn clockNow() std.Io.Timestamp {
    if (!clock_available) return .{ .nanoseconds = 0 };
    return std.Io.Clock.awake.now(std.Io.Threaded.global_single_threaded.io());
}

pub fn start(limit_ns: ?u64) std.Io.Timestamp {
    return if (limit_ns == null) .{ .nanoseconds = 0 } else clockNow();
}

pub fn limitNs(phase: RunPhase, extended: bool) u64 {
    return switch (phase) {
        .train => if (extended) 300 * std.time.ns_per_s else 60 * std.time.ns_per_s,
        .evaluate => if (extended) 120 * std.time.ns_per_min else 30 * std.time.ns_per_min,
    };
}

/// A partial score keeps a long game from being discarded solely because it is playing well.
/// Extension is deliberately unavailable on the second (already extended) attempt.
pub fn promising(score: u32, max_tile: core.Tile, extended: bool) bool {
    return !extended and (score >= promising_partial_score or max_tile >= promising_partial_tile);
}

/// Checked between moves; the current search is not preempted. The signed duration also tolerates
/// equal timestamps and a hypothetical earlier timestamp without an underflow.
pub fn reached(game_start: std.Io.Timestamp, limit_ns: ?u64) bool {
    const limit = limit_ns orelse return false;
    const elapsed = game_start.durationTo(clockNow()).nanoseconds;
    return elapsed >= 0 and elapsed >= @as(i96, limit);
}
