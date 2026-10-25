const std = @import("std");
const core = @import("core");

const index_html = @embedFile("web/index.html");

/// One shared game per server process. Requests are handled one at a time,
/// so no locking is needed.
pub const Session = struct {
    state: core.GameState,
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Session {
        var prng = std.Random.DefaultPrng.init(seed);
        const state = core.GameState.gameStart(prng.random());
        return .{ .state = state, .prng = prng };
    }

    pub fn newGame(self: *Session) void {
        self.state = core.GameState.gameStart(self.prng.random());
    }

    pub fn move(self: *Session, dir: core.Direction) core.MoveResult {
        return self.state.move(dir, self.prng.random());
    }
};

const StateJson = struct {
    board: core.Board,
    score: u32,
    game_over: bool,
    changed: bool,
    spawned: bool,
};

fn stateJson(session: *const Session, changed: bool, spawned: bool) StateJson {
    return .{
        .board = session.state.board,
        .score = session.state.score,
        .game_over = session.state.isGameOver(),
        .changed = changed,
        .spawned = spawned,
    };
}

/// Routes a request. Returns status, content type, and body.
pub fn route(
    gpa: std.mem.Allocator,
    session: *Session,
    method: std.http.Method,
    target: []const u8,
) !Response {
    const path, const query = if (std.mem.indexOfScalar(u8, target, '?')) |i|
        .{ target[0..i], target[i + 1 ..] }
    else
        .{ target, "" };

    if (method == .GET and std.mem.eql(u8, path, "/")) {
        return .{ .status = .ok, .content_type = "text/html; charset=utf-8", .body = index_html };
    }
    if (method == .GET and std.mem.eql(u8, path, "/api/state")) {
        return jsonResponse(gpa, .ok, stateJson(session, false, false));
    }
    if (method == .POST and std.mem.eql(u8, path, "/api/new")) {
        session.newGame();
        return jsonResponse(gpa, .ok, stateJson(session, false, false));
    }
    if (method == .POST and std.mem.eql(u8, path, "/api/move")) {
        const dir = directionFromQuery(query) orelse
            return .{ .status = .bad_request, .content_type = "text/plain", .body = "dir must be up|left|down|right\n" };
        const result = session.move(dir);
        return jsonResponse(gpa, .ok, stateJson(session, result.changed, result.spawned));
    }
    return .{ .status = .not_found, .content_type = "text/plain", .body = "not found\n" };
}

pub const Response = struct {
    status: std.http.Status,
    content_type: []const u8,
    body: []const u8,
    owned: bool = false,
};

fn jsonResponse(gpa: std.mem.Allocator, status: std.http.Status, value: anytype) !Response {
    const body = try std.json.Stringify.valueAlloc(gpa, value, .{});
    return .{ .status = status, .content_type = "application/json", .body = body, .owned = true };
}

fn directionFromQuery(query: []const u8) ?core.Direction {
    var it = std.mem.splitScalar(u8, query, '&');
    while (it.next()) |pair| {
        if (std.mem.startsWith(u8, pair, "dir=")) return core.Direction.parse(pair["dir=".len..]);
    }
    return null;
}

pub fn serve(io: std.Io, gpa: std.mem.Allocator, port: u16, seed: u64) !void {
    const address: std.Io.net.IpAddress = try .parse("127.0.0.1", port);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    var session = Session.init(seed);
    std.debug.print("2048 server: http://127.0.0.1:{d}/\n", .{port});

    while (true) {
        const stream = try listener.accept(io);
        defer stream.close(io);
        handleConnection(io, gpa, &session, stream) catch |err| {
            std.debug.print("connection error: {s}\n", .{@errorName(err)});
        };
    }
}

fn handleConnection(io: std.Io, gpa: std.mem.Allocator, session: *Session, stream: std.Io.net.Stream) !void {
    var read_buf: [8192]u8 = undefined;
    var write_buf: [8192]u8 = undefined;
    var net_reader = stream.reader(io, &read_buf);
    var net_writer = stream.writer(io, &write_buf);
    var http_server = std.http.Server.init(&net_reader.interface, &net_writer.interface);

    var req = try http_server.receiveHead();
    const res = try route(gpa, session, req.head.method, req.head.target);
    defer if (res.owned) gpa.free(res.body);

    try req.respond(res.body, .{
        .status = res.status,
        // one request per connection: bodyless POSTs would trip std's keep-alive body assert
        .keep_alive = false,
        .extra_headers = &.{.{ .name = "content-type", .value = res.content_type }},
    });
}

test "route serves the page and rejects unknown paths" {
    const gpa = std.testing.allocator;
    var session = Session.init(1);

    const page = try route(gpa, &session, .GET, "/");
    try std.testing.expectEqual(std.http.Status.ok, page.status);
    try std.testing.expect(std.mem.indexOf(u8, page.body, "2048") != null);

    const missing = try route(gpa, &session, .GET, "/nope");
    try std.testing.expectEqual(std.http.Status.not_found, missing.status);
}

test "route move validates direction and returns JSON state" {
    const gpa = std.testing.allocator;
    var session = Session.init(1);

    const bad = try route(gpa, &session, .POST, "/api/move?dir=sideways");
    try std.testing.expectEqual(std.http.Status.bad_request, bad.status);

    const ok = try route(gpa, &session, .POST, "/api/move?dir=left");
    defer gpa.free(ok.body);
    try std.testing.expectEqual(std.http.Status.ok, ok.status);
    try std.testing.expect(std.mem.indexOf(u8, ok.body, "\"board\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, ok.body, "\"game_over\":false") != null);
}

test "route move rejects GET and new resets the board" {
    const gpa = std.testing.allocator;
    var session = Session.init(7);

    const get_move = try route(gpa, &session, .GET, "/api/move?dir=up");
    try std.testing.expectEqual(std.http.Status.not_found, get_move.status);

    session.state = .{};
    const new = try route(gpa, &session, .POST, "/api/new");
    defer gpa.free(new.body);
    try std.testing.expectEqual(@as(usize, 2), countTiles(session.state.board));
}

fn countTiles(board: core.Board) usize {
    var n: usize = 0;
    for (board) |row| for (row) |v| {
        if (v != 0) n += 1;
    };
    return n;
}
