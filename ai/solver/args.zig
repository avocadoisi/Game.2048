//! Positional-argument parsing shared by the CLI runners in this directory (#139).
//!
//! Every runner reads its positional arguments the same way: `args[i]` when it was given, else a
//! named default. The order of those arguments is a compatibility surface (each runner documents it
//! in its `usage` string and the runner's tests inspect that text), so this module only removes the
//! repetition — never the order or the `usage` text.

const std = @import("std");

/// `args[i]` parsed as `T`, or `default` when the argument is absent.
/// `[]const u8` is taken verbatim (paths, variant names); floats and ints are parsed.
pub fn argOr(comptime T: type, args: []const []const u8, i: usize, default: T) !T {
    if (args.len <= i) return default;
    return switch (T) {
        []const u8 => args[i],
        f64, f32 => try std.fmt.parseFloat(T, args[i]),
        else => try std.fmt.parseInt(T, args[i], 10),
    };
}

test "argOr falls back to the default when the argument is absent" {
    const args = [_][]const u8{ "prog", "7" };
    try std.testing.expectEqual(@as(u32, 5), try argOr(u32, &args, 2, 5));
    try std.testing.expectEqual(@as(u8, 2), try argOr(u8, &args, 4, 2));
    try std.testing.expectEqual(@as(f64, 0.5), try argOr(f64, &args, 9, 0.5));
    try std.testing.expectEqualStrings("base", try argOr([]const u8, &args, 3, "base"));
}

test "argOr parses the given argument" {
    const args = [_][]const u8{ "prog", "7", "1000", "0.003", "resume" };
    try std.testing.expectEqual(@as(u32, 7), try argOr(u32, &args, 1, 2000));
    try std.testing.expectEqual(@as(u64, 1000), try argOr(u64, &args, 2, 1));
    try std.testing.expectEqual(@as(f64, 0.003), try argOr(f64, &args, 3, 0.01));
    try std.testing.expectEqualStrings("resume", try argOr([]const u8, &args, 4, "base"));
}

test "argOr propagates a parse error like the hand-written parsers did" {
    const args = [_][]const u8{ "prog", "nope" };
    try std.testing.expectError(error.InvalidCharacter, argOr(u32, &args, 1, 0));
}
