const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Width of the value MLP (#205): the hidden layers are compile-time sized, so a capacity sweep is a
    // build option instead of three edited copies (`zig build nn -Dh1=32 -Dh2=16`).
    const width_opts = b.addOptions();
    width_opts.addOption(u32, "h1", b.option(u32, "h1", "hidden layer 1 width (default 64)") orelse 64);
    width_opts.addOption(u32, "h2", b.option(u32, "h2", "hidden layer 2 width (default 32)") orelse 32);

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
            .{ .name = "nn_width", .module = width_opts.createModule() },
        },
    });

    const ntuple_mod = b.addModule("ntuple", .{
        .root_source_file = b.path("ai/solver/ntuple.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
        },
    });

    const ntuple_quant_exe = b.addExecutable(.{
        .name = "ntuple-quant",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/ntuple_quant.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "ntuple", .module = ntuple_mod },
            },
        }),
    });
    b.installArtifact(ntuple_quant_exe);
    const run_ntuple_quant = b.addRunArtifact(ntuple_quant_exe);
    run_ntuple_quant.addPassthruArgs();
    const ntuple_quant_step = b.step("ntuple-quant", "Quantize an n-tuple table to int16 (#129)");
    ntuple_quant_step.dependOn(&run_ntuple_quant.step);

    const solver_mod = b.addModule("solver", .{
        .root_source_file = b.path("ai/solver/solver.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "nn", .module = nn_mod },
            .{ .name = "ntuple", .module = ntuple_mod },
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
    // Run steps depend on their own executable only. Depending on `b.getInstallStep()` made
    // `zig build nn -- 2000 ...` recompile every other artifact first (game-2048, solve, ranking,
    // ...) because the install step pulls all of them; a one-minute training run paid a full
    // ReleaseFast rebuild of the whole set. `zig build` (the default step) still installs everything.
    // forwards `zig build run -- <args>` to the app (replaces removed `b.args`)
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run the CLI app");
    run_step.dependOn(&run_cmd.step);

    const test_filters: []const []const u8 = if (b.option([]const u8, "test-filter", "only run tests whose name contains this substring")) |f| &.{f} else &.{};
    const core_tests = b.addTest(.{ .filters = test_filters,
        .root_module = core_mod,
    });
    const run_core_tests = b.addRunArtifact(core_tests);

    const exe_tests = b.addTest(.{ .filters = test_filters,
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const solver_tests = b.addTest(.{ .filters = test_filters,
        .root_module = solver_mod,
    });
    const bench_tests = b.addTest(.{ .filters = test_filters,
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

    const ranking_tests = b.addTest(.{ .filters = test_filters,
        .root_module = ranking_exe.root_module,
    });
    const run_ranking_tests = b.addRunArtifact(ranking_tests);

    const ranking_step = b.step("ranking", "Compare policies and print JSON ranking");
    ranking_step.dependOn(&ranking_cmd.step);

    const parallel_tests = b.addTest(.{ .filters = test_filters, .root_module = parallel_mod });    const run_parallel_tests = b.addRunArtifact(parallel_tests);

    const evalab_exe = b.addExecutable(.{
        .name = "evalab",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/evalab.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "solver", .module = solver_mod },
                .{ .name = "parallel", .module = parallel_mod },
            },
        }),
    });
    const evalab_cmd = b.addRunArtifact(evalab_exe);
    evalab_cmd.addPassthruArgs();

    const evalab_tests = b.addTest(.{ .filters = test_filters, .root_module = evalab_exe.root_module });
    const run_evalab_tests = b.addRunArtifact(evalab_tests);

    const evalab_step = b.step("evalab", "Compare evaluation variants (path shape, anchor, extra terms) on the same seeds");
    evalab_step.dependOn(&evalab_cmd.step);

    const probe_exe = b.addExecutable(.{
        .name = "probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/probe.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "solver", .module = solver_mod },
                .{ .name = "nn", .module = nn_mod },
                .{ .name = "ntuple", .module = ntuple_mod },
            },
        }),
    });
    const probe_cmd = b.addRunArtifact(probe_exe);
    probe_cmd.addPassthruArgs();

    const probe_tests = b.addTest(.{ .filters = test_filters, .root_module = probe_exe.root_module });
    const run_probe_tests = b.addRunArtifact(probe_tests);

    const probe_step = b.step("probe", "Score one tile multiset on probe board shapes with snake / nn / n-tuple evaluations");
    probe_step.dependOn(&probe_cmd.step);

    const nn_tests = b.addTest(.{ .filters = test_filters, .root_module = nn_mod });
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

    const nn_main_tests = b.addTest(.{ .filters = test_filters, .root_module = nn_exe.root_module });
    const run_nn_main_tests = b.addRunArtifact(nn_main_tests);

    const nn_step = b.step("nn", "Train the value MLP (TD(0) self-play) and evaluate it against expectimax depth 2");
    nn_step.dependOn(&nn_cmd.step);

    // The int16 file is a shipping artefact: write it where you like, do not commit it (#170).
    const nn_quant_exe = b.addExecutable(.{
        .name = "nn-quant",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/nn_quant.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "nn", .module = nn_mod },
            },
        }),
    });
    const nn_quant_cmd = b.addRunArtifact(nn_quant_exe);
    nn_quant_cmd.addPassthruArgs();
    const nn_quant_step = b.step("nn-quant", "Quantize the value net to int16 and measure the error (#170)");
    nn_quant_step.dependOn(&nn_quant_cmd.step);

    const ntuple_tests = b.addTest(.{ .filters = test_filters, .root_module = ntuple_mod });
    const run_ntuple_tests = b.addRunArtifact(ntuple_tests);

    const ntuple_exe = b.addExecutable(.{
        .name = "ntuple",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/ntuple_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "core", .module = core_mod },
                .{ .name = "solver", .module = solver_mod },
                .{ .name = "ntuple", .module = ntuple_mod },
                .{ .name = "parallel", .module = parallel_mod },
            },
        }),
    });
    const ntuple_cmd = b.addRunArtifact(ntuple_exe);
    ntuple_cmd.addPassthruArgs();

    const ntuple_main_tests = b.addTest(.{ .filters = test_filters, .root_module = ntuple_exe.root_module });
    const run_ntuple_main_tests = b.addRunArtifact(ntuple_main_tests);

    const ntuple_step = b.step("ntuple", "Train the n-tuple TD value function and evaluate it against expectimax depth 2");
    ntuple_step.dependOn(&ntuple_cmd.step);

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

    const ga_step = b.step("ga-rl", "Evolve evaluation weights with a GA and compare on holdout seeds");
    ga_step.dependOn(&ga_cmd.step);

    const ga_tests = b.addTest(.{ .filters = test_filters,
        .root_module = ga_exe.root_module,
    });
    const run_ga_tests = b.addRunArtifact(ga_tests);

    const tune_exe = b.addExecutable(.{
        .name = "tune",
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/tune.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "solver", .module = solver_mod },
                .{ .name = "parallel", .module = parallel_mod },
            },
        }),
    });
    const tune_cmd = b.addRunArtifact(tune_exe);
    tune_cmd.addPassthruArgs();

    const tune_tests = b.addTest(.{ .filters = test_filters, .root_module = tune_exe.root_module });
    const run_tune_tests = b.addRunArtifact(tune_tests);

    const tune_step = b.step("tune", "Optimise the evaluation weights with CMA-ES");
    tune_step.dependOn(&tune_cmd.step);

    const cmaes_tests = b.addTest(.{ .filters = test_filters,
        .root_module = b.createModule(.{
            .root_source_file = b.path("ai/solver/cmaes.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_cmaes_tests = b.addRunArtifact(cmaes_tests);

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
            .{ .name = "nn_width", .module = width_opts.createModule() },
        },
    });
    const wasm_ntuple_mod = b.addModule("ntuple-wasm", .{
        .root_source_file = b.path("ai/solver/ntuple.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = wasm_core_mod },
        },
    });
    const wasm_solver_mod = b.addModule("solver-wasm", .{
        .root_source_file = b.path("ai/solver/solver.zig"),
        .target = wasm_target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = wasm_core_mod },
            .{ .name = "nn", .module = wasm_nn_mod },
            .{ .name = "ntuple", .module = wasm_ntuple_mod },
        },
    });
    // Which policies the browser module ships is deploy config, not code: the list lives in
    // `wasm.policies` next to this file and `src/wasm.zig` parses it at compile time (#154).
    // `@embedFile` makes editing that file rebuild the module, and the parse (with its
    // "unknown policy" / "needs the n-tuple table" errors) stays in one place.
    const wasm_options = b.addOptions();
    wasm_options.addOption([]const u8, "policies", @embedFile("wasm.policies"));

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
                .{ .name = "options", .module = wasm_options.createModule() },
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

    // Per-target test steps: `zig build test` compiles all 15 targets in Debug (5-10 min), which is
    // too slow to verify one module and makes an agent look hung. These build one target only.
    b.step("test-nn", "Run the nn unit tests only").dependOn(&run_nn_tests.step);
    b.step("test-nn-main", "Run the nn CLI unit tests only").dependOn(&run_nn_main_tests.step);
    b.step("test-core", "Run the core unit tests only").dependOn(&run_core_tests.step);
    b.step("test-solver", "Run the solver unit tests only").dependOn(&run_solver_tests.step);
    b.step("test-bench", "Run the bench unit tests only").dependOn(&run_bench_tests.step);
    b.step("test-ranking", "Run the ranking unit tests only").dependOn(&run_ranking_tests.step);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_solver_tests.step);
    test_step.dependOn(&run_bench_tests.step);
    test_step.dependOn(&run_nn_tests.step);
    test_step.dependOn(&run_nn_main_tests.step);
    test_step.dependOn(&run_ntuple_tests.step);
    test_step.dependOn(&run_ntuple_main_tests.step);
    test_step.dependOn(&run_parallel_tests.step);
    test_step.dependOn(&run_ga_tests.step);
    test_step.dependOn(&run_tune_tests.step);
    test_step.dependOn(&run_cmaes_tests.step);
    test_step.dependOn(&run_ranking_tests.step);
    test_step.dependOn(&run_evalab_tests.step);
    test_step.dependOn(&run_probe_tests.step);
}
