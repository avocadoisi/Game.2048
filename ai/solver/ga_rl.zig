const std = @import("std");
const core = @import("core");
const solver = @import("solver");

const Weights = solver.Weights;
const gene_count = 4;

/// Search range per gene, in the field order of `Weights`.
const lo = [gene_count]f64{ 0, 0, 0, 0 };
const hi = [gene_count]f64{ 50, 100, 50, 500 };

fn toGenes(w: Weights) [gene_count]f64 {
    return .{ w.empty, w.order_penalty, w.merge, w.forced_down };
}

fn fromGenes(g: [gene_count]f64) Weights {
    // the late-phase penalty tracks the same gene so evolution stays phase-invariant
    return .{ .empty = g[0], .order_penalty = g[1], .order_penalty_late = g[1], .merge = g[2], .forced_down = g[3] };
}

/// Tournament selection: the fittest of `tournament_size` random individuals.
pub fn select(rng: std.Random, scores: []const f64) usize {
    var best = rng.uintLessThan(usize, scores.len);
    for (1..tournament_size) |_| {
        const challenger = rng.uintLessThan(usize, scores.len);
        if (scores[challenger] > scores[best]) best = challenger;
    }
    return best;
}
const tournament_size = 3;

/// Uniform crossover: each gene comes from one parent at random.
pub fn crossover(rng: std.Random, a: [gene_count]f64, b: [gene_count]f64) [gene_count]f64 {
    var child = a;
    for (&child, b) |*gene, other| {
        if (rng.boolean()) gene.* = other;
    }
    return child;
}

/// Each gene moves with probability `mutation_rate` by up to `mutation_step` of its range; result clamped to range.
pub fn mutate(rng: std.Random, genes: *[gene_count]f64) void {
    for (genes, lo, hi) |*gene, low, high| {
        if (rng.float(f64) >= mutation_rate) continue;
        const delta = (rng.float(f64) * 2 - 1) * mutation_step * (high - low);
        gene.* = std.math.clamp(gene.* + delta, low, high);
    }
}
const mutation_rate = 0.2;
const mutation_step = 0.2;

/// Average score of `weights` over the fixed training games (game i uses PRNG seed `first_seed + i`).
fn fitness(weights: Weights, games: u32, depth: u8, first_seed: u64) f64 {
    var total: u64 = 0;
    for (0..games) |i| {
        var prng = std.Random.DefaultPrng.init(first_seed + i);
        const rng = prng.random();
        var state = core.GameState.gameStart(rng);
        while (solver.bestMoveW(state, depth, weights)) |d| _ = state.move(d, rng);
        total += state.score;
    }
    return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(games));
}

const usage =
    \\usage: ga-rl [population] [generations] [train_games] [depth] [seed] [holdout_seed] [holdout_games]
    \\
    \\Evolves the snake evaluation weights on training seeds 1..train_games, then compares
    \\the baseline and the best weights on holdout seeds holdout_seed.. (same seeds for both).
    \\  defaults: 16 15 8 2 1 500 100
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const population: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 16;
    const generations: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 15;
    const train_games: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 8;
    const depth: u8 = if (args.len > 4) try std.fmt.parseInt(u8, args[4], 10) else 2;
    const seed: u64 = if (args.len > 5) try std.fmt.parseInt(u64, args[5], 10) else 1;
    const holdout_seed: u64 = if (args.len > 6) try std.fmt.parseInt(u64, args[6], 10) else 500;
    const holdout_games: u32 = if (args.len > 7) try std.fmt.parseInt(u32, args[7], 10) else 100;
    if (args.len > 8 or population < 2 or train_games == 0 or holdout_games == 0) {
        std.debug.print("{s}", .{usage});
        return;
    }

    const start = std.Io.Clock.awake.now(init.io);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // generation 0: the baseline plus random individuals inside the range
    var pop: []([gene_count]f64) = try init.arena.allocator().alloc([gene_count]f64, population);
    const fit = try init.arena.allocator().alloc(f64, population);
    pop[0] = toGenes(solver.baseline);
    for (pop[1..]) |*ind| {
        for (ind, lo, hi) |*gene, low, high| gene.* = low + rng.float(f64) * (high - low);
    }

    var next = try init.arena.allocator().alloc([gene_count]f64, population);
    var best: usize = 0;
    for (0..generations + 1) |gen| {
        for (pop, fit) |ind, *f| f.* = fitness(fromGenes(ind), train_games, depth, 1);
        best = 0;
        for (fit, 0..) |f, i| {
            if (f > fit[best]) best = i;
        }
        std.debug.print("gen {d:>2}: best train avg {d:.0}  weights {any}\n", .{ gen, fit[best], pop[best] });
        if (gen == generations) break;

        // elitism: the best individual survives unchanged
        next[0] = pop[best];
        for (next[1..]) |*child| {
            const a = pop[select(rng, fit)];
            const b = pop[select(rng, fit)];
            child.* = crossover(rng, a, b);
            mutate(rng, child);
        }
        std.mem.swap([]([gene_count]f64), &pop, &next);
    }
    const evolved = fromGenes(pop[best]);

    const base_avg = fitness(solver.baseline, holdout_games, depth, holdout_seed);
    const evo_avg = fitness(evolved, holdout_games, depth, holdout_seed);
    const elapsed = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e9;
    std.debug.print("evolved: empty={d:.3} order_penalty={d:.3} merge={d:.3} forced_down={d:.3}\n", .{ evolved.empty, evolved.order_penalty, evolved.merge, evolved.forced_down });
    std.debug.print("holdout seeds {d}..{d} ({d} games, depth {d}): baseline avg {d:.0}, evolved avg {d:.0}\n", .{ holdout_seed, holdout_seed + holdout_games - 1, holdout_games, depth, base_avg, evo_avg });
    std.debug.print("elapsed {d:.1} s\n", .{elapsed});
}

test "select returns an index inside the population" {
    var prng = std.Random.DefaultPrng.init(3);
    const fit = [_]f64{ 1, 5, 2 };
    for (0..100) |_| try std.testing.expect(select(prng.random(), &fit) < fit.len);
}

test "crossover and mutate keep genes inside the search range" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const a = [gene_count]f64{ lo[0], lo[1], lo[2], lo[3] };
    const b = [gene_count]f64{ hi[0], hi[1], hi[2], hi[3] };
    for (0..200) |_| {
        var child = crossover(rng, a, b);
        for (child, a, b) |g, x, y| try std.testing.expect(g == x or g == y);
        for (0..20) |_| mutate(rng, &child);
        for (child, lo, hi) |g, low, high| {
            try std.testing.expect(g >= low and g <= high);
        }
    }
}
