const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    const solver_mod = b.addModule("solver", .{
        .root_source_file = b.path("ai/solver/solver.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
        },
    });

    const exe = b.addExecutable(.{
        .name = "game-2048",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "solver", .module = solver_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    // forwards `zig build run -- <args>` to the app (replaces removed `b.args`)
    run_cmd.addPassthruArgs();
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the CLI app");
    run_step.dependOn(&run_cmd.step);

    const core_tests = b.addTest(.{
        .root_module = core_mod,
    });
    const run_core_tests = b.addRunArtifact(core_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const solver_tests = b.addTest(.{
        .root_module = solver_mod,
    });
    const bench_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "solver", .module = solver_mod },
            },
        }),
    });
    const run_bench_tests = b.addRunArtifact(bench_tests);
    const run_solver_tests = b.addRunArtifact(solver_tests);

    const solve_exe = b.addExecutable(.{
        .name = "solve",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "solver", .module = solver_mod },
            },
        }),
    });
    const solve_cmd = b.addRunArtifact(solve_exe);
    solve_cmd.addPassthruArgs();
    solve_cmd.step.dependOn(b.getInstallStep());

    const solve_step = b.step("solve", "Run the solver and report game statistics");
    solve_step.dependOn(&solve_cmd.step);

    const ranking_exe = b.addExecutable(.{
        .name = "ranking",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/ranking.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "solver", .module = solver_mod },
            },
        }),
    });
    const ranking_cmd = b.addRunArtifact(ranking_exe);
    ranking_cmd.addPassthruArgs();
    ranking_cmd.step.dependOn(b.getInstallStep());

    const ranking_step = b.step("ranking", "Compare policies and print JSON ranking");
    ranking_step.dependOn(&ranking_cmd.step);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_solver_tests.step);
    test_step.dependOn(&run_bench_tests.step);
}
