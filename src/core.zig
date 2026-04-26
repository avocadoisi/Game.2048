const std = @import("std");

pub const BoardSize: usize = 4;
pub const BoardCellCount: usize = BoardSize * BoardSize;
pub const Board = [BoardSize][BoardSize]u16;

pub const Direction = enum {
    up,
    left,
    down,
    right,
};

pub const MoveResult = struct {
    changed: bool,
    spawned: bool,
    score_delta: u32,
};

pub const MergeResult = struct {
    line: [BoardSize]u16,
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
        var changed = false;
        var total_score_delta: u32 = 0;

        for (0..BoardSize) |line_index| {
            const extracted = extractLine(self.board, direction, line_index);
            const merged = mergeLineLeft(extracted);
            injectLine(&self.board, direction, line_index, merged.line);

            changed = changed or merged.changed;
            total_score_delta += merged.score_delta;
        }

        self.score += total_score_delta;

        var spawned = false;
        if (changed) {
            spawned = self.spawnRandomTile(rng);
        }

        return .{
            .changed = changed,
            .spawned = spawned,
            .score_delta = total_score_delta,
        };
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

        const empty_pick = rng.uintLessThan(usize, empty_count);
        const value_pick = rng.uintLessThan(u8, 10);
        const value: u16 = if (value_pick == 0) 4 else 2;

        const pos = empties[empty_pick];
        self.board[pos.row][pos.col] = value;
        return true;
    }

    pub fn spawnAt(self: *GameState, pos: Position, value: u16) SpawnError!void {
        if (pos.row >= BoardSize or pos.col >= BoardSize) {
            return error.OutOfBounds;
        }
        if (self.board[pos.row][pos.col] != 0) {
            return error.Occupied;
        }
        if (value != 2 and value != 4) {
            return error.InvalidValue;
        }

        self.board[pos.row][pos.col] = value;
    }

    pub fn isGameOver(self: GameState) bool {
        for (0..BoardSize) |row| {
            for (0..BoardSize) |col| {
                const value = self.board[row][col];
                if (value == 0) return false;

                if (row + 1 < BoardSize and self.board[row + 1][col] == value) {
                    return false;
                }
                if (col + 1 < BoardSize and self.board[row][col + 1] == value) {
                    return false;
                }
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

pub fn mergeLine(line: [BoardSize]u16) MergeResult {
    return mergeLineLeft(line);
}

pub fn spawn(state: *GameState, rng: anytype) bool {
    return state.spawnRandomTile(rng);
}

pub fn mergeLineLeft(line: [BoardSize]u16) MergeResult {
    var compacted: [BoardSize]u16 = undefined;
    var compact_len: usize = 0;

    for (line) |value| {
        if (value == 0) continue;
        compacted[compact_len] = value;
        compact_len += 1;
    }

    var merged = [_]u16{0} ** BoardSize;
    var merged_index: usize = 0;
    var read_index: usize = 0;
    var score_delta: u32 = 0;

    while (read_index < compact_len) {
        const current = compacted[read_index];
        if (read_index + 1 < compact_len and compacted[read_index + 1] == current) {
            merged[merged_index] = current * 2;
            score_delta += @as(u32, current) * 2;
            read_index += 2;
        } else {
            merged[merged_index] = current;
            read_index += 1;
        }

        merged_index += 1;
    }

    const changed = !std.mem.eql(u16, line[0..], merged[0..]);
    return .{
        .line = merged,
        .changed = changed,
        .score_delta = score_delta,
    };
}

fn extractLine(board: Board, direction: Direction, line_index: usize) [BoardSize]u16 {
    var line: [BoardSize]u16 = undefined;

    switch (direction) {
        .left => for (0..BoardSize) |offset| {
            line[offset] = board[line_index][offset];
        },
        .right => for (0..BoardSize) |offset| {
            line[offset] = board[line_index][BoardSize - 1 - offset];
        },
        .up => for (0..BoardSize) |offset| {
            line[offset] = board[offset][line_index];
        },
        .down => for (0..BoardSize) |offset| {
            line[offset] = board[BoardSize - 1 - offset][line_index];
        },
    }

    return line;
}

fn injectLine(board: *Board, direction: Direction, line_index: usize, line: [BoardSize]u16) void {
    switch (direction) {
        .left => for (0..BoardSize) |offset| {
            board[line_index][offset] = line[offset];
        },
        .right => for (0..BoardSize) |offset| {
            board[line_index][BoardSize - 1 - offset] = line[offset];
        },
        .up => for (0..BoardSize) |offset| {
            board[offset][line_index] = line[offset];
        },
        .down => for (0..BoardSize) |offset| {
            board[BoardSize - 1 - offset][line_index] = line[offset];
        },
    }
}

const SequenceRng = struct {
    values: []const usize,
    index: usize = 0,

    pub fn uintLessThan(self: *SequenceRng, comptime T: type, upper: T) T {
        std.debug.assert(upper > 0);
        std.debug.assert(self.index < self.values.len);

        const raw = self.values[self.index];
        self.index += 1;

        const upper_usize: usize = @intCast(upper);
        const picked = raw % upper_usize;
        return @as(T, @intCast(picked));
    }
};

const NoCallRng = struct {
    pub fn uintLessThan(_: *NoCallRng, comptime T: type, _: T) T {
        @panic("rng should not be called");
    }
};

fn countTiles(board: Board) usize {
    var count: usize = 0;
    for (0..BoardSize) |row| {
        for (0..BoardSize) |col| {
            if (board[row][col] != 0) count += 1;
        }
    }
    return count;
}

test "mergeLineLeft merges each pair once" {
    const merged = mergeLineLeft(.{ 2, 2, 2, 2 });

    try std.testing.expectEqual([BoardSize]u16{ 4, 4, 0, 0 }, merged.line);
    try std.testing.expect(merged.changed);
    try std.testing.expectEqual(@as(u32, 8), merged.score_delta);
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

    var game = GameState{
        .board = board_before,
        .score = 100,
    };

    var rng = NoCallRng{};
    const result = game.moveLeft(&rng);

    try std.testing.expect(!result.changed);
    try std.testing.expect(!result.spawned);
    try std.testing.expectEqual(@as(u32, 0), result.score_delta);
    try std.testing.expectEqual(@as(u32, 100), game.score);
    try std.testing.expectEqual(board_before, game.board);
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

test "isGameOver detects playable and terminal boards" {
    const terminal: GameState = .{
        .board = .{
            .{ 2, 4, 2, 4 },
            .{ 4, 2, 4, 2 },
            .{ 2, 4, 2, 4 },
            .{ 4, 2, 4, 2 },
        },
        .score = 0,
    };

    const playable: GameState = .{
        .board = .{
            .{ 2, 4, 2, 4 },
            .{ 4, 2, 4, 2 },
            .{ 2, 2, 4, 8 },
            .{ 16, 32, 64, 128 },
        },
        .score = 0,
    };

    try std.testing.expect(terminal.isGameOver());
    try std.testing.expect(!playable.isGameOver());
}
