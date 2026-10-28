//! Small CMA-ES for tuning real-valued weights (issue #105).
//!
//! shortcut: CSA step size + rank-one covariance update only (no rank-mu), and samples are clipped
//! into `bounds` instead of penalised; good enough for the 4-dim weight search. Upgrade to the full
//! rank-mu update if a higher-dimensional search stalls.
const std = @import("std");

/// Search settings; `bounds` is per dimension, a degenerate `[v, v]` range pins that dimension.
pub fn Opts(comptime n: usize) type {
    return struct {
        /// initial mean, clipped into `bounds`
        x0: [n]f64,
        sigma: f64,
        pop: usize,
        generations: usize,
        seed: u64,
        bounds: [n][2]f64,
        /// print one line per evaluation
        verbose: bool = false,
    };
}

const max_pop = 64;
/// Score given to a sample the objective rejects (non-finite), so the ranking stays finite.
const rejected_score = 1e300;

/// Minimise `objective` and return the best point evaluated. Deterministic for a fixed `seed`.
/// A sample outside `bounds` is clipped before it is scored; a generation in which every sample is
/// rejected (the objective returned a non-finite value) ends the run early.
pub fn minimise(comptime n: usize, ctx: anytype, opts: Opts(n), objective: fn (@TypeOf(ctx), [n]f64) f64) [n]f64 {
    std.debug.assert(opts.pop >= 2 and opts.pop % 2 == 0 and opts.pop <= max_pop);
    std.debug.assert(opts.sigma > 0);

    const lambda = opts.pop;
    const mu = lambda / 2;

    // log-decreasing recombination weights, normalised to sum 1
    var weights: [max_pop]f64 = undefined;
    var weight_sum: f64 = 0;
    for (0..mu) |i| {
        weights[i] = @log(@as(f64, @floatFromInt(mu)) + 0.5) - @log(@as(f64, @floatFromInt(i + 1)));
        weight_sum += weights[i];
    }
    for (0..mu) |i| weights[i] /= weight_sum;
    var mueff: f64 = 0;
    for (0..mu) |i| mueff += weights[i] * weights[i];
    mueff = 1 / mueff;

    const nf: f64 = @floatFromInt(n);
    const cc = 4 / (nf + 4);
    const cs = (mueff + 2) / (nf + mueff + 5);
    const c1 = 2 / ((nf + 1.3) * (nf + 1.3) + mueff);
    const damps = 1 + 2 * @max(0, @sqrt((mueff - 1) / (nf + 1)) - 1) + cs;
    // E||N(0, I)||, the reference norm of the sigma path
    const chi_n = @sqrt(nf) * (1 - 1 / (4 * nf) + 1 / (21 * nf * nf));

    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();

    var mean: [n]f64 = undefined;
    for (0..n) |i| mean[i] = std.math.clamp(opts.x0[i], opts.bounds[i][0], opts.bounds[i][1]);
    var sigma = opts.sigma;
    var cov: [n][n]f64 = undefined;
    for (0..n) |i| {
        for (0..n) |j| cov[i][j] = if (i == j) 1 else 0;
    }
    var p_sigma: [n]f64 = @splat(0);
    var p_c: [n]f64 = @splat(0);

    var best_x = mean;
    var best_score = std.math.inf(f64);

    for (0..opts.generations) |gen| {
        const l = decompose(n, &cov, &p_c);

        var pop: [max_pop][n]f64 = undefined;
        var score: [max_pop]f64 = undefined;
        var rejected: usize = 0;
        for (0..lambda) |k| {
            var z: [n]f64 = undefined;
            for (&z) |*v| v.* = rng.floatNorm(f64);
            var x = mean;
            // x = mean + sigma * L z, then clipped into the bounds
            for (0..n) |i| {
                var step: f64 = 0;
                for (0..i + 1) |j| step += l[i][j] * z[j];
                x[i] = std.math.clamp(x[i] + sigma * step, opts.bounds[i][0], opts.bounds[i][1]);
            }
            pop[k] = x;
            const s = objective(ctx, x);
            if (std.math.isFinite(s)) {
                score[k] = s;
                if (s < best_score) {
                    best_score = s;
                    best_x = x;
                }
            } else {
                rejected += 1;
                score[k] = rejected_score;
            }
            if (opts.verbose) {
                std.debug.print("gen={d}/{d} f={d:.1} best={d:.1} x=[", .{ gen + 1, opts.generations, score[k], best_score });
                for (x, 0..) |v, j| std.debug.print("{s}{d:.2}", .{ if (j == 0) "" else ", ", v });
                std.debug.print("]\n", .{});
            }
        }
        if (rejected == lambda) break;

        // rank the samples, best first
        var order: [max_pop]usize = undefined;
        for (0..lambda) |k| order[k] = k;
        for (1..lambda) |i| {
            const o = order[i];
            var j = i;
            while (j > 0 and score[order[j - 1]] > score[o]) : (j -= 1) order[j] = order[j - 1];
            order[j] = o;
        }

        // weighted recombination of the best half
        var new_mean: [n]f64 = @splat(0);
        for (0..mu) |i| {
            const child = pop[order[i]];
            for (0..n) |j| new_mean[j] += weights[i] * child[j];
        }
        var step_mean: [n]f64 = undefined;
        for (0..n) |i| step_mean[i] = (new_mean[i] - mean[i]) / sigma;
        mean = new_mean;

        // CSA: the C^-1/2 norm of the mean step drives the step size; `forwardSolve` gives that norm
        const inv_step = forwardSolve(n, l, step_mean);
        const ps_scale = @sqrt(cs * (2 - cs) * mueff);
        var ps_norm: f64 = 0;
        for (0..n) |i| {
            p_sigma[i] = (1 - cs) * p_sigma[i] + ps_scale * inv_step[i];
            ps_norm += p_sigma[i] * p_sigma[i];
        }
        ps_norm = @sqrt(ps_norm);

        const hsig: f64 = if (ps_norm / @sqrt(1 - std.math.pow(f64, 1 - cs, 2 * (@as(f64, @floatFromInt(gen)) + 1))) / chi_n < 1.4 + 2 / (nf + 1)) 1 else 0;
        const pc_scale = @sqrt(cc * (2 - cc) * mueff);
        for (0..n) |i| p_c[i] = (1 - cc) * p_c[i] + hsig * pc_scale * step_mean[i];

        // rank-one covariance update (the rank-mu term is dropped)
        for (0..n) |i| {
            for (0..n) |j| {
                cov[i][j] = (1 - c1) * cov[i][j] + c1 * (p_c[i] * p_c[j] + (1 - hsig) * cc * (2 - cc) * cov[i][j]);
            }
        }

        sigma *= @exp((cs / damps) * (ps_norm / chi_n - 1));
        // shortcut: keep the step size within 1e-6..1e3 of the initial value; clipping already
        // bounds every sample, this only guards against a runaway after a rejected generation.
        sigma = std.math.clamp(sigma, opts.sigma * 1e-6, opts.sigma * 1e3);
    }
    return best_x;
}

/// Lower Cholesky factor of the covariance; a numerically broken matrix is reset to the identity.
fn decompose(comptime n: usize, cov: *[n][n]f64, p_c: *[n]f64) [n][n]f64 {
    if (cholesky(n, cov.*)) |l| return l;
    for (0..n) |i| {
        for (0..n) |j| cov[i][j] = if (i == j) 1 else 0;
    }
    p_c.* = @splat(0);
    return cholesky(n, cov.*).?;
}

fn cholesky(comptime n: usize, cov: [n][n]f64) ?[n][n]f64 {
    var l: [n][n]f64 = undefined;
    for (0..n) |i| {
        for (0..n) |j| l[i][j] = 0;
    }
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var s = cov[i][j];
            for (0..j) |k| s -= l[i][k] * l[j][k];
            if (i == j) {
                if (!(s > 0)) return null;
                l[i][j] = @sqrt(s);
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    return l;
}

/// Solve `L y = v` for the lower triangular `L` (forward substitution).
fn forwardSolve(comptime n: usize, l: [n][n]f64, v: [n]f64) [n]f64 {
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        var acc = v[i];
        for (0..i) |k| acc -= l[i][k] * y[k];
        y[i] = acc / l[i][i];
    }
    return y;
}

fn sphere(_: void, x: [2]f64) f64 {
    return x[0] * x[0] + x[1] * x[1];
}

test "cmaes minimises a 2-d sphere" {
    const best = minimise(2, {}, .{
        .x0 = .{ 3, -4 },
        .sigma = 0.5,
        .pop = 8,
        .generations = 200,
        .seed = 7,
        .bounds = .{ .{ -10, 10 }, .{ -10, 10 } },
    }, sphere);
    try std.testing.expect(@abs(best[0]) < 1e-3);
    try std.testing.expect(@abs(best[1]) < 1e-3);
}

/// Records any sample outside [0, 1]; both dimensions are pinned to that range.
fn insideUnit(flag: *bool, x: [2]f64) f64 {
    for (x) |v| {
        if (v < 0 or v > 1) flag.* = true;
    }
    return x[0] + x[1];
}

test "cmaes clips samples into the bounds" {
    var outside = false;
    const best = minimise(2, &outside, .{
        .x0 = .{ 2, 2 },
        .sigma = 4,
        .pop = 8,
        .generations = 20,
        .seed = 3,
        .bounds = .{ .{ 0, 1 }, .{ 0, 1 } },
    }, insideUnit);
    try std.testing.expect(!outside);
    for (best) |v| try std.testing.expect(v >= 0 and v <= 1);
}

test "cmaes returns the same result for the same seed" {
    const opts = Opts(2){
        .x0 = .{ 3, -4 },
        .sigma = 0.5,
        .pop = 6,
        .generations = 40,
        .seed = 11,
        .bounds = .{ .{ -10, 10 }, .{ -10, 10 } },
    };
    const a = minimise(2, {}, opts, sphere);
    const b = minimise(2, {}, opts, sphere);
    try std.testing.expectEqual(a, b);
    var other = opts;
    other.seed = 12;
    try std.testing.expect(!std.mem.eql(f64, &a, &minimise(2, {}, other, sphere)));
}
