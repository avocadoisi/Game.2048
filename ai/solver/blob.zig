//! Cursor-style little-endian I/O for the weight and table files (#302).
//!
//! Every format used to carry its own offset arithmetic and bounds checks; this is the one place
//! that turns bytes into values and back. Formats keep their own layout, magic and error names —
//! only the cursor mechanics live here.
const std = @import("std");

/// The header every weight/table file shares: 8-byte magic, u32 version, u32 endianness marker.
pub const endian_marker: u32 = 0x01020304;

/// The caller's own names for the three header failures, so each format keeps its own errors.
pub const HeaderErrors = struct { magic: anyerror, version: anyerror, endian: anyerror };

pub const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn init(bytes: []const u8) Reader {
        return .{ .bytes = bytes };
    }

    pub fn remaining(self: *const Reader) usize {
        return self.bytes.len - self.pos;
    }

    /// Next `n` bytes, or `error.Truncated` without advancing.
    pub fn take(self: *Reader, n: usize) ![]const u8 {
        if (n > self.remaining()) return error.Truncated;
        defer self.pos += n;
        return self.bytes[self.pos..][0..n];
    }

    fn takeArray(self: *Reader, comptime n: usize) !*const [n]u8 {
        if (n > self.remaining()) return error.Truncated;
        defer self.pos += n;
        return self.bytes[self.pos..][0..n];
    }

    pub fn raw(self: *Reader, out: []u8) !void {
        @memcpy(out, try self.take(out.len));
    }

    pub fn readU32(self: *Reader) !u32 {
        return std.mem.readInt(u32, try self.takeArray(4), .little);
    }

    pub fn readU64(self: *Reader) !u64 {
        return std.mem.readInt(u64, try self.takeArray(8), .little);
    }

    pub fn readI16(self: *Reader) !i16 {
        return std.mem.readInt(i16, try self.takeArray(2), .little);
    }

    pub fn readF32(self: *Reader) !f32 {
        return @bitCast(try self.readU32());
    }

    pub fn readF64(self: *Reader) !f64 {
        return @bitCast(try self.readU64());
    }
};

/// Writes into a caller-sized buffer. Encode paths compute the exact length first, so the cursor
/// never runs past the end; reads are the bounds-checked side.
pub const Writer = struct {
    bytes: []u8,
    pos: usize = 0,

    pub fn init(bytes: []u8) Writer {
        return .{ .bytes = bytes };
    }

    pub fn raw(self: *Writer, src: []const u8) void {
        @memcpy(self.bytes[self.pos..][0..src.len], src);
        self.pos += src.len;
    }

    pub fn writeU32(self: *Writer, value: u32) void {
        std.mem.writeInt(u32, self.bytes[self.pos..][0..4], value, .little);
        self.pos += 4;
    }

    pub fn writeU64(self: *Writer, value: u64) void {
        std.mem.writeInt(u64, self.bytes[self.pos..][0..8], value, .little);
        self.pos += 8;
    }

    pub fn writeI16(self: *Writer, value: i16) void {
        std.mem.writeInt(i16, self.bytes[self.pos..][0..2], value, .little);
        self.pos += 2;
    }

    pub fn writeF32(self: *Writer, value: f32) void {
        self.writeU32(@bitCast(value));
    }

    pub fn writeF64(self: *Writer, value: f64) void {
        self.writeU64(@bitCast(value));
    }
};

pub fn writeHeader(w: *Writer, magic: []const u8, version: u32) void {
    w.raw(magic);
    w.writeU32(version);
    w.writeU32(endian_marker);
}

/// The header every file shares. `errs` names the three failures so each format keeps its own values.
pub fn readHeader(r: *Reader, magic: []const u8, version: u32, errs: HeaderErrors) !void {
    if (!std.mem.eql(u8, try r.take(8), magic)) return errs.magic;
    if (try r.readU32() != version) return errs.version;
    if (try r.readU32() != endian_marker) return errs.endian;
}

test "writer and reader round-trip every scalar little endian" {
    var bytes: [26]u8 = undefined;
    var w = Writer.init(&bytes);
    w.writeU32(0x01020304);
    w.writeU64(0x1122334455667788);
    w.writeI16(-1234);
    w.writeF32(1.5);
    w.writeF64(-2.25);
    try std.testing.expectEqual(@as(usize, bytes.len), w.pos);

    // spot-check the byte order itself, not just the round trip through this module
    try std.testing.expectEqualSlices(u8, &[_]u8{ 4, 3, 2, 1 }, bytes[0..4]);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x2e, 0xfb }, bytes[12..14]); // -1234

    var r = Reader.init(&bytes);
    try std.testing.expectEqual(@as(u32, 0x01020304), try r.readU32());
    try std.testing.expectEqual(@as(u64, 0x1122334455667788), try r.readU64());
    try std.testing.expectEqual(@as(i16, -1234), try r.readI16());
    try std.testing.expectEqual(@as(f32, 1.5), try r.readF32());
    try std.testing.expectEqual(@as(f64, -2.25), try r.readF64());
    try std.testing.expectEqual(@as(usize, 0), r.remaining());
}

test "raw bytes round-trip and stop at the end" {
    var bytes: [5]u8 = undefined;
    var w = Writer.init(&bytes);
    w.raw("ab");
    w.raw("cde");

    var back: [5]u8 = undefined;
    var r = Reader.init(&bytes);
    try r.raw(back[0..2]);
    try r.raw(back[2..]);
    try std.testing.expectEqualStrings("abcde", &back);
}

test "reader rejects a short buffer without consuming it" {
    const bytes = [_]u8{ 1, 2, 3 };
    var r = Reader.init(&bytes);
    try std.testing.expectError(error.Truncated, r.readU32());
    try std.testing.expectEqual(@as(usize, 0), r.pos);
    try std.testing.expectEqual(@as(usize, 3), (try r.take(3)).len);
    try std.testing.expectError(error.Truncated, r.take(1));
    try std.testing.expectEqual(@as(usize, 3), r.pos);

    var out: [4]u8 = undefined;
    var empty = Reader.init(&[_]u8{});
    try std.testing.expectError(error.Truncated, empty.raw(&out));
    try std.testing.expectError(error.Truncated, empty.readF64());
}

test "header round-trips and names each failure" {
    var bytes: [16]u8 = undefined;
    var w = Writer.init(&bytes);
    writeHeader(&w, "2048-XX\x00", 7);

    var r = Reader.init(&bytes);
    try readHeader(&r, "2048-XX\x00", 7, .{
        .magic = error.BadMagic,
        .version = error.BadVersion,
        .endian = error.BadEndian,
    });
    try std.testing.expectEqual(@as(usize, 0), r.remaining());

    var bad_magic = bytes;
    bad_magic[0] ^= 0xff;
    var rm = Reader.init(&bad_magic);
    try std.testing.expectError(error.BadMagic, readHeader(&rm, "2048-XX\x00", 7, .{
        .magic = error.BadMagic,
        .version = error.BadVersion,
        .endian = error.BadEndian,
    }));

    var bad_version = bytes;
    std.mem.writeInt(u32, bad_version[8..12], 8, .little);
    var rv = Reader.init(&bad_version);
    try std.testing.expectError(error.BadVersion, readHeader(&rv, "2048-XX\x00", 7, .{
        .magic = error.BadMagic,
        .version = error.BadVersion,
        .endian = error.BadEndian,
    }));

    var bad_endian = bytes;
    std.mem.writeInt(u32, bad_endian[12..16], 0x04030201, .little);
    var re = Reader.init(&bad_endian);
    try std.testing.expectError(error.BadEndian, readHeader(&re, "2048-XX\x00", 7, .{
        .magic = error.BadMagic,
        .version = error.BadVersion,
        .endian = error.BadEndian,
    }));
}
