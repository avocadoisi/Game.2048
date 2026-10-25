const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    const parallel_mod = b.addModule("parallel", .{
        .root_source_file = b.path("ai/solver/parallel.zig"),
        .target = target,
        .optimize = optimize,
    });

    const nn_mod = b.addModule("nn", .{
        .root_source_file = b.path("ai/solver/nn.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "parallel", .module = parallel_mod },
        },
    });

    const solver_mod = b.addModule("solver", .{
        .root_source_file = b.path("ai/solver/solver.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "nn", .module = nn_mod },
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
                .{ .name = "parallel", .module = parallel_mod },
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
                .{ .name = "parallel", .module = parallel_mod },
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
                .{ .name = "parallel", .module = parallel_mod },
            },
        }),
    });
    const ranking_cmd = b.addRunArtifact(ranking_exe);
    ranking_cmd.addPassthruArgs();
    ranking_cmd.step.dependOn(b.getInstallStep());

    const ranking_step = b.step("ranking", "Compare policies and print JSON ranking");
    ranking_step.dependOn(&ranking_cmd.step);

    const parallel_tests = b.addTest(.{ .root_module = parallel_mod });
    const run_parallel_tests = b.addRunArtifact(parallel_tests);

    const nn_tests = b.addTest(.{ .root_module = nn_mod });
    const run_nn_tests = b.addRunArtifact(nn_tests);

    const nn_exe = b.addExecutable(.{
        .name = "nn",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/nn_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "solver", .module = solver_mod },
                .{ .name = "nn", .module = nn_mod },
                .{ .name = "parallel", .module = parallel_mod },
            },
        }),
    });
    const nn_cmd = b.addRunArtifact(nn_exe);
    nn_cmd.addPassthruArgs();
    nn_cmd.step.dependOn(b.getInstallStep());

    const nn_main_tests = b.addTest(.{ .root_module = nn_exe.root_module });
    const run_nn_main_tests = b.addRunArtifact(nn_main_tests);

    const nn_step = b.step("nn", "Train the value MLP (TD(0) self-play) and evaluate it against expectimax depth 2");
    nn_step.dependOn(&nn_cmd.step);

    const ga_exe = b.addExecutable(.{
        .name = "ga-rl",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/ga_rl.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "solver", .module = solver_mod },
            },
        }),
    });
    const ga_cmd = b.addRunArtifact(ga_exe);
    ga_cmd.addPassthruArgs();
    ga_cmd.step.dependOn(b.getInstallStep());

    const ga_step = b.step("ga-rl", "Evolve evaluation weights with a GA and compare on holdout seeds");
    ga_step.dependOn(&ga_cmd.step);

    const ga_tests = b.addTest(.{
        .root_module = ga_exe.root_module,
    });
    const run_ga_tests = b.addRunArtifact(ga_tests);

    // WASM build for static hosting: `zig build wasm -Doptimize=ReleaseFast`
    const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const wasm_core_mod = b.addModule("core-wasm", .{
        .root_source_file = b.path("src/core.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    const wasm_parallel_mod = b.addModule("parallel-wasm", .{
        .root_source_file = b.path("ai/solver/parallel.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    const wasm_nn_mod = b.addModule("nn-wasm", .{
        .root_source_file = b.path("ai/solver/nn.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = wasm_core_mod },
            .{ .name = "parallel", .module = wasm_parallel_mod },
        },
    });
    const wasm_solver_mod = b.addModule("solver-wasm", .{
        .root_source_file = b.path("ai/solver/solver.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = wasm_core_mod },
            .{ .name = "nn", .module = wasm_nn_mod },
        },
    });
    const wasm_exe = b.addExecutable(.{
        .name = "game2048",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm.zig"),
            .target = wasm_target,
            .optimize = optimize,
            .strip = true,
            .imports = &.{
                .{ .name = "core", .module = wasm_core_mod },
                .{ .name = "solver", .module = wasm_solver_mod },
            },
        }),
    });
    // No `_start`: JS instantiates and calls the exports directly; memory is exported.
    wasm_exe.entry = .disabled;
    wasm_exe.import_memory = false;
    wasm_exe.rdynamic = true;
    wasm_exe.export_memory = true;

    const install_wasm = b.addInstallFileWithDir(wasm_exe.getEmittedBin(), .{ .custom = "web" }, "game2048.wasm");
    const install_static_html = b.addInstallFileWithDir(b.path("src/web/index.static.html"), .{ .custom = "web" }, "index.html");

    const wasm_step = b.step("wasm", "Build the WebAssembly module and static site into zig-out/web");
    wasm_step.dependOn(&install_wasm.step);
    wasm_step.dependOn(&install_static_html.step);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_solver_tests.step);
    test_step.dependOn(&run_bench_tests.step);
    test_step.dependOn(&run_nn_tests.step);
    test_step.dependOn(&run_nn_main_tests.step);
    test_step.dependOn(&run_parallel_tests.step);
    test_step.dependOn(&run_ga_tests.step);
}
