const std = @import("std");

pub const BoardSize: usize = 4;
pub const BoardCellCount: usize = BoardSize * BoardSize;

/// u32 (not u16): a merge of two 32768 tiles produces 65536, which overflowed u16.
pub const Tile = u32;
pub const Board = [BoardSize][BoardSize]Tile;

pub const Direction = enum {
    up,
    left,
    down,
    right,

    pub fn parse(name: []const u8) ?Direction {
        return std.meta.stringToEnum(Direction, name);
    }
};

pub const MoveResult = struct {
    changed: bool,
    spawned: bool,
    score_delta: u32,
};

pub const MergeResult = struct {
    line: [BoardSize]Tile,
    changed: bool,
    score_delta: u32,
};

pub const SpawnError = error{
    OutOfBounds,
    Occupied,
    InvalidValue,
};

pub const Position = struct {
    row: usize,
    col: usize,
};

pub const GameState = struct {
    board: Board = std.mem.zeroes(Board),
    score: u32 = 0,

    pub fn initEmpty() GameState {
        return .{};
    }

    pub fn gameStart(rng: anytype) GameState {
        var state = GameState.initEmpty();
        _ = state.spawnRandomTile(rng);
        _ = state.spawnRandomTile(rng);
        return state;
    }

    pub fn move(self: *GameState, direction: Direction, rng: anytype) MoveResult {
        var result = self.slide(direction);
        result.spawned = result.changed and self.spawnRandomTile(rng);
        return result;
    }

    /// Slide and merge without spawning a tile (used by search).
    pub fn slide(self: *GameState, direction: Direction) MoveResult {
        var changed = false;
        var score_delta: u32 = 0;

        for (0..BoardSize) |line_index| {
            var line: [BoardSize]Tile = undefined;
            for (&line, 0..) |*cell, offset| {
                const p = cellAt(direction, line_index, offset);
                cell.* = self.board[p.row][p.col];
            }

            const merged = mergeLine(line);
            for (merged.line, 0..) |value, offset| {
                const p = cellAt(direction, line_index, offset);
                self.board[p.row][p.col] = value;
            }

            changed = changed or merged.changed;
            score_delta += merged.score_delta;
        }

        self.score += score_delta;
        return .{ .changed = changed, .spawned = false, .score_delta = score_delta };
    }

    pub fn moveUp(self: *GameState, rng: anytype) MoveResult {
        return self.move(.up, rng);
    }

    pub fn moveLeft(self: *GameState, rng: anytype) MoveResult {
        return self.move(.left, rng);
    }

    pub fn moveDown(self: *GameState, rng: anytype) MoveResult {
        return self.move(.down, rng);
    }

    pub fn moveRight(self: *GameState, rng: anytype) MoveResult {
        return self.move(.right, rng);
    }

    pub fn spawnRandomTile(self: *GameState, rng: anytype) bool {
        var empties: [BoardCellCount]Position = undefined;
        var empty_count: usize = 0;

        for (0..BoardSize) |row| {
            for (0..BoardSize) |col| {
                if (self.board[row][col] == 0) {
                    empties[empty_count] = .{ .row = row, .col = col };
                    empty_count += 1;
                }
            }
        }

        if (empty_count == 0) return false;

        const pos = empties[rng.uintLessThan(usize, empty_count)];
        const value: Tile = if (rng.uintLessThan(u8, 10) == 0) 4 else 2;
        self.board[pos.row][pos.col] = value;
        return true;
    }

    pub fn spawnAt(self: *GameState, pos: Position, value: Tile) SpawnError!void {
        if (pos.row >= BoardSize or pos.col >= BoardSize) return error.OutOfBounds;
        if (self.board[pos.row][pos.col] != 0) return error.Occupied;
        if (value != 2 and value != 4) return error.InvalidValue;

        self.board[pos.row][pos.col] = value;
    }

    /// True when `slide(direction)` would change the board. Does not modify the state.
    pub fn canSlide(self: GameState, direction: Direction) bool {
        for (0..BoardSize) |line_index| {
            var line: [BoardSize]Tile = undefined;
            for (&line, 0..) |*cell, offset| {
                const p = cellAt(direction, line_index, offset);
                cell.* = self.board[p.row][p.col];
            }
            if (mergeLine(line).changed) return true;
        }
        return false;
    }

    pub fn isGameOver(self: GameState) bool {
        for (0..BoardSize) |row| {
            for (0..BoardSize) |col| {
                const value = self.board[row][col];
                if (value == 0) return false;
                if (row + 1 < BoardSize and self.board[row + 1][col] == value) return false;
                if (col + 1 < BoardSize and self.board[row][col + 1] == value) return false;
            }
        }
        return true;
    }
};

pub fn gameStart(rng: anytype) GameState {
    return GameState.gameStart(rng);
}

pub fn moveUp(state: *GameState, rng: anytype) MoveResult {
    return state.moveUp(rng);
}

pub fn moveLeft(state: *GameState, rng: anytype) MoveResult {
    return state.moveLeft(rng);
}

pub fn moveDown(state: *GameState, rng: anytype) MoveResult {
    return state.moveDown(rng);
}

pub fn moveRight(state: *GameState, rng: anytype) MoveResult {
    return state.moveRight(rng);
}

pub fn spawn(state: *GameState, rng: anytype) bool {
    return state.spawnRandomTile(rng);
}

/// Slide tiles toward index 0 and merge each equal pair once (left-to-right).
/// Uses a precomputed table when every tile is a power of two below 65536; otherwise the
/// general path (`mergeLineSlow`) runs, so odd tile values still behave the same.
pub fn mergeLine(line: [BoardSize]Tile) MergeResult {
    var key: u16 = 0;
    for (line, 0..) |value, i| {
        if (value > 32768 or @popCount(value) > 1) return mergeLineSlow(line);
        const exp: u16 = if (value == 0) 0 else @ctz(value);
        key |= exp << @intCast(i * 4);
    }
    const entry = row_table[key];
    if (!entry.ok) return mergeLineSlow(line);
    var merged: [BoardSize]Tile = undefined;
    for (&merged, 0..) |*cell, i| {
        const exp: u4 = @truncate(entry.row >> @intCast(i * 4));
        cell.* = if (exp == 0) 0 else @as(Tile, 1) << exp;
    }
    return .{ .line = merged, .changed = entry.row != key, .score_delta = entry.score };
}

const RowEntry = struct {
    /// Merged line as 4 nibbles of log2 tile values (0 = empty).
    row: u16,
    score: u32,
    /// false when a merge would produce a tile above 32768 (nibble overflow).
    ok: bool,
};

/// row_table[key]: key packs 4 nibbles of log2 tile values, offset 0 in the low nibble.
const row_table: [65536]RowEntry = blk: {
    @setEvalBranchQuota(1_000_000_000);
    var t: [65536]RowEntry = undefined;
    for (0..65536) |key| {
        var line: [BoardSize]Tile = undefined;
        for (&line, 0..) |*cell, i| {
            const exp: u4 = @truncate(key >> @intCast(i * 4));
            cell.* = if (exp == 0) 0 else @as(Tile, 1) << exp;
        }
        const merged = mergeLineSlow(line);
        var row: u16 = 0;
        var ok = true;
        for (merged.line, 0..) |value, i| {
            const exp: u16 = if (value == 0) 0 else @ctz(value);
            if (exp > 15) ok = false;
            row |= exp << @intCast(i * 4);
        }
        t[key] = .{ .row = row, .score = merged.score_delta, .ok = ok };
    }
    break :blk t;
};

fn mergeLineSlow(line: [BoardSize]Tile) MergeResult {
    var packed_tiles: [BoardSize]Tile = @splat(0);
    var count: usize = 0;
    for (line) |value| {
        if (value != 0) {
            packed_tiles[count] = value;
            count += 1;
        }
    }

    var merged: [BoardSize]Tile = @splat(0);
    var score_delta: u32 = 0;
    var read: usize = 0;
    var write: usize = 0;
    while (read < count) : (write += 1) {
        const value = packed_tiles[read];
        if (read + 1 < count and packed_tiles[read + 1] == value) {
            merged[write] = value * 2;
            score_delta += value * 2;
            read += 2;
        } else {
            merged[write] = value;
            read += 1;
        }
    }

    return .{
        .line = merged,
        .changed = !std.mem.eql(Tile, &line, &merged),
        .score_delta = score_delta,
    };
}

/// Board cell for the `offset`-th cell of a line, where offset 0 is the cell
/// the tiles slide toward (left column, right column, top row, bottom row).
fn cellAt(direction: Direction, line_index: usize, offset: usize) Position {
    const far = BoardSize - 1 - offset;
    return switch (direction) {
        .left => .{ .row = line_index, .col = offset },
        .right => .{ .row = line_index, .col = far },
        .up => .{ .row = offset, .col = line_index },
        .down => .{ .row = far, .col = line_index },
    };
}

const SequenceRng = struct {
    values: []const usize,
    index: usize = 0,

    pub fn uintLessThan(self: *SequenceRng, comptime T: type, upper: T) T {
        std.debug.assert(self.index < self.values.len);
        const raw = self.values[self.index];
        self.index += 1;
        return @intCast(raw % @as(usize, upper));
    }
};

const NoCallRng = struct {
    pub fn uintLessThan(_: *NoCallRng, comptime T: type, _: T) T {
        @panic("rng should not be called");
    }
};

fn countTiles(board: Board) usize {
    var count: usize = 0;
    for (board) |row| {
        for (row) |value| {
            if (value != 0) count += 1;
        }
    }
    return count;
}

test "mergeLine merges each pair once" {
    const merged = mergeLine(.{ 2, 2, 2, 2 });
    try std.testing.expectEqual([BoardSize]Tile{ 4, 4, 0, 0 }, merged.line);
    try std.testing.expect(merged.changed);
    try std.testing.expectEqual(@as(u32, 8), merged.score_delta);
}

test "mergeLine does not chain a new merge onto an existing one" {
    const merged = mergeLine(.{ 2, 2, 4, 0 });
    try std.testing.expectEqual([BoardSize]Tile{ 4, 4, 0, 0 }, merged.line);
    try std.testing.expectEqual(@as(u32, 4), merged.score_delta);
}

test "mergeLine compacts gaps without merging" {
    const merged = mergeLine(.{ 0, 2, 0, 4 });
    try std.testing.expectEqual([BoardSize]Tile{ 2, 4, 0, 0 }, merged.line);
    try std.testing.expect(merged.changed);
    try std.testing.expectEqual(@as(u32, 0), merged.score_delta);
}

test "mergeLine reports unchanged for packed non-mergeable line" {
    const merged = mergeLine(.{ 2, 4, 8, 0 });
    try std.testing.expect(!merged.changed);
    try std.testing.expectEqual(@as(u32, 0), merged.score_delta);
}

test "mergeLine of empty line is unchanged" {
    const merged = mergeLine(.{ 0, 0, 0, 0 });
    try std.testing.expectEqual([BoardSize]Tile{ 0, 0, 0, 0 }, merged.line);
    try std.testing.expect(!merged.changed);
}

test "mergeLine does not overflow when tile reaches 65536" {
    const merged = mergeLine(.{ 32768, 32768, 0, 0 });
    try std.testing.expectEqual([BoardSize]Tile{ 65536, 0, 0, 0 }, merged.line);
    try std.testing.expectEqual(@as(u32, 65536), merged.score_delta);
}

test "moveLeft merges line vectors and then spawns" {
    var game = GameState{
        .board = .{
            .{ 2, 0, 2, 0 },
            .{ 0, 0, 0, 0 },
            .{ 2, 2, 4, 0 },
            .{ 0, 0, 0, 0 },
        },
        .score = 0,
    };

    var rng = SequenceRng{ .values = &[_]usize{ 0, 1 } };
    const result = game.moveLeft(&rng);

    const expected_board: Board = .{
        .{ 4, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 4, 4, 0, 0 },
        .{ 0, 0, 0, 0 },
    };

    try std.testing.expectEqual(expected_board, game.board);
    try std.testing.expect(result.changed);
    try std.testing.expect(result.spawned);
    try std.testing.expectEqual(@as(u32, 8), result.score_delta);
    try std.testing.expectEqual(@as(u32, 8), game.score);
}

test "moveLeft does not spawn on unchanged board" {
    const board_before: Board = .{
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 512, 1024, 2, 4 },
        .{ 8, 16, 32, 64 },
    };

    var game = GameState{ .board = board_before, .score = 100 };
    var rng = NoCallRng{};
    const result = game.moveLeft(&rng);

    try std.testing.expect(!result.changed);
    try std.testing.expect(!result.spawned);
    try std.testing.expectEqual(@as(u32, 0), result.score_delta);
    try std.testing.expectEqual(@as(u32, 100), game.score);
    try std.testing.expectEqual(board_before, game.board);
}

test "moveUp slides and merges toward the top row" {
    var game = GameState{ .board = .{
        .{ 2, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 4, 0, 0, 0 },
    } };
    var rng = SequenceRng{ .values = &[_]usize{ 0, 0 } };
    _ = game.moveUp(&rng);

    try std.testing.expectEqual(@as(Tile, 4), game.board[0][0]);
    try std.testing.expectEqual(@as(Tile, 4), game.board[1][0]);
    try std.testing.expectEqual(@as(Tile, 0), game.board[2][0]);
    try std.testing.expectEqual(@as(Tile, 0), game.board[3][0]);
}

test "moveDown slides and merges toward the bottom row" {
    var game = GameState{ .board = .{
        .{ 2, 0, 0, 0 },
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 4, 0, 0, 0 },
    } };
    // after the move column 0 is {0,0,4,4}; the spawn lands in the first empty cell (0,0)
    var rng = SequenceRng{ .values = &[_]usize{ 0, 1 } };
    _ = game.moveDown(&rng);

    try std.testing.expectEqual(@as(Tile, 2), game.board[0][0]);
    try std.testing.expectEqual(@as(Tile, 0), game.board[1][0]);
    try std.testing.expectEqual(@as(Tile, 4), game.board[2][0]);
    try std.testing.expectEqual(@as(Tile, 4), game.board[3][0]);
}

test "moveRight slides and merges toward the right column" {
    var game = GameState{ .board = .{
        .{ 2, 2, 0, 4 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    // after the move row 0 is {0,0,4,4}; spawn of 2 lands in first empty cell (0,0)
    var rng = SequenceRng{ .values = &[_]usize{ 0, 1 } };
    const result = game.moveRight(&rng);

    try std.testing.expectEqual([BoardSize]Tile{ 2, 0, 4, 4 }, game.board[0]);
    try std.testing.expect(result.changed);
    try std.testing.expectEqual(@as(u32, 4), result.score_delta);
}

test "move spawns into a cell freed by a merge" {
    var game = GameState{ .board = .{
        .{ 2, 4, 8, 16 },
        .{ 2, 4, 8, 16 },
        .{ 32, 64, 128, 256 },
        .{ 32, 64, 128, 256 },
    } };
    // moveUp merges every column pair: 8 tiles remain, rows 2-3 are empty.
    // rng index 0 -> first empty (2,0); value roll 1 -> 2.
    var rng = SequenceRng{ .values = &[_]usize{ 0, 1 } };
    const result = game.moveUp(&rng);

    try std.testing.expect(result.changed);
    try std.testing.expect(result.spawned);
    try std.testing.expectEqual(@as(Tile, 2), game.board[2][0]);
    try std.testing.expectEqual(@as(usize, 9), countTiles(game.board));
}

test "spawnRandomTile returns false on full board" {
    var game = GameState{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };
    var rng = NoCallRng{};
    try std.testing.expect(!game.spawnRandomTile(&rng));
}

test "spawnRandomTile picks the rng-chosen empty cell and spawns 4 on a 1-in-10 roll" {
    var game = GameState{ .board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    // empties in row-major order: (0,1)(0,2)(0,3)(1,0)...; index 1 -> (0,2)
    // value roll 0 -> 4
    var rng = SequenceRng{ .values = &[_]usize{ 1, 0 } };
    try std.testing.expect(game.spawnRandomTile(&rng));
    try std.testing.expectEqual(@as(Tile, 4), game.board[0][2]);
}

test "spawnAt rejects out of bounds, occupied, and invalid values" {
    var game = GameState{ .board = .{
        .{ 2, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    try std.testing.expectError(error.OutOfBounds, game.spawnAt(.{ .row = 4, .col = 0 }, 2));
    try std.testing.expectError(error.Occupied, game.spawnAt(.{ .row = 0, .col = 0 }, 2));
    try std.testing.expectError(error.InvalidValue, game.spawnAt(.{ .row = 1, .col = 1 }, 8));
    try game.spawnAt(.{ .row = 1, .col = 1 }, 4);
    try std.testing.expectEqual(@as(Tile, 4), game.board[1][1]);
}

test "Direction.parse accepts only the four direction names" {
    try std.testing.expectEqual(Direction.up, Direction.parse("up").?);
    try std.testing.expectEqual(Direction.right, Direction.parse("right").?);
    try std.testing.expectEqual(@as(?Direction, null), Direction.parse("w"));
    try std.testing.expectEqual(@as(?Direction, null), Direction.parse(""));
}

test "gameStart spawns two deterministic tiles" {
    var rng = SequenceRng{ .values = &[_]usize{ 0, 1, 0, 1 } };
    const game = GameState.gameStart(&rng);

    const expected_board: Board = .{
        .{ 2, 2, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };

    try std.testing.expectEqual(expected_board, game.board);
    try std.testing.expectEqual(@as(usize, 2), countTiles(game.board));
    try std.testing.expectEqual(@as(u32, 0), game.score);
}

test "same seed reproduces the same game" {
    var a = std.Random.DefaultPrng.init(12345);
    var b = std.Random.DefaultPrng.init(12345);
    var ga = GameState.gameStart(a.random());
    var gb = GameState.gameStart(b.random());
    for ([_]Direction{ .left, .up, .right, .down, .left }) |d| {
        _ = ga.move(d, a.random());
        _ = gb.move(d, b.random());
    }
    try std.testing.expectEqual(ga.board, gb.board);
    try std.testing.expectEqual(ga.score, gb.score);
}

test "isGameOver detects playable and terminal boards" {
    const terminal: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
    } };

    const playable: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 2, 4, 8 },
        .{ 16, 32, 64, 128 },
    } };

    try std.testing.expect(terminal.isGameOver());
    try std.testing.expect(!playable.isGameOver());
}

test "isGameOver is false while an empty cell remains" {
    const state: GameState = .{ .board = .{
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 2 },
        .{ 2, 4, 2, 4 },
        .{ 4, 2, 4, 0 },
    } };
    try std.testing.expect(!state.isGameOver());
}

test "slide merges like move but never spawns" {
    var game = GameState{ .board = .{
        .{ 2, 0, 2, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    } };
    const result = game.slide(.left);

    try std.testing.expectEqual([BoardSize]Tile{ 4, 0, 0, 0 }, game.board[0]);
    try std.testing.expect(result.changed);
    try std.testing.expect(!result.spawned);
    try std.testing.expectEqual(@as(u32, 4), result.score_delta);
    try std.testing.expectEqual(@as(u32, 4), game.score);
}

test "mergeLine table matches the general path for every power-of-two row" {
    for (0..65536) |key| {
        var line: [BoardSize]Tile = undefined;
        for (&line, 0..) |*cell, i| {
            const exp: u4 = @truncate(key >> @intCast(i * 4));
            cell.* = if (exp == 0) 0 else @as(Tile, 1) << exp;
        }
        const fast = mergeLine(line);
        const slow = mergeLineSlow(line);
        try std.testing.expectEqual(slow.line, fast.line);
        try std.testing.expectEqual(slow.changed, fast.changed);
        try std.testing.expectEqual(slow.score_delta, fast.score_delta);
    }
}

test "mergeLine falls back for tiles the table cannot hold" {
    const big = mergeLine(.{ 65536, 65536, 3, 0 });
    try std.testing.expectEqual([BoardSize]Tile{ 131072, 3, 0, 0 }, big.line);
    try std.testing.expectEqual(@as(u32, 131072), big.score_delta);
    const odd = mergeLine(.{ 3, 3, 0, 0 });
    try std.testing.expectEqual([BoardSize]Tile{ 6, 0, 0, 0 }, odd.line);
}

test "canSlide matches slide(...).changed without changing the board" {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    for (0..20_000) |_| {
        var board: Board = undefined;
        for (&board) |*row| {
            for (row) |*cell| {
                cell.* = if (rng.uintLessThan(u8, 3) == 0) 0 else @as(Tile, 1) << @intCast(1 + rng.uintLessThan(u8, 10));
            }
        }
        const state = GameState{ .board = board, .score = 5 };
        for ([_]Direction{ .up, .left, .down, .right }) |d| {
            var copy = state;
            try std.testing.expectEqual(copy.slide(d).changed, state.canSlide(d));
        }
        try std.testing.expectEqual(board, state.board);
    }
}
