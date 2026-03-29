const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const turboquant_mod = b.addModule("turboquant", .{
        .root_source_file = b.path("src/turboquant.zig"),
        .target = target,
    });
    turboquant_mod.addImport("matrix", b.addModule("matrix", .{
        .root_source_file = b.path("src/matrix.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("polar", b.addModule("polar", .{
        .root_source_file = b.path("src/polar.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("qjl", b.addModule("qjl", .{
        .root_source_file = b.path("src/qjl.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("format", b.addModule("format", .{
        .root_source_file = b.path("src/format.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("rotation", b.addModule("rotation", .{
        .root_source_file = b.path("src/rotation.zig"),
        .target = target,
    }));
    turboquant_mod.addImport("math", b.addModule("math", .{
        .root_source_file = b.path("src/math.zig"),
        .target = target,
    }));

    const tests = b.addTest(.{
        .root_module = turboquant_mod,
    });

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // Shared library for C/Python bindings (zig build lib)
    const optimize = b.standardOptimizeOption(.{});
    const c_api_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });

    const shared_lib = b.addLibrary(.{
        .name = "turboquant",
        .root_module = c_api_mod,
        .linkage = .dynamic,
    });
    shared_lib.linkLibC();

    const install_lib = b.addInstallArtifact(shared_lib, .{});
    const install_header = b.addInstallFile(b.path("include/turboquant.h"), "include/turboquant.h");

    const lib_step = b.step("lib", "Build shared library for C/Python bindings");
    lib_step.dependOn(&install_lib.step);
    lib_step.dependOn(&install_header.step);

    const bench_mod = b.addModule("bench", .{
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
    });
    bench_mod.addImport("turboquant", turboquant_mod);

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = bench_mod,
    });

    const bench_run = b.addRunArtifact(bench_exe);
    if (b.args) |args| bench_run.addArgs(args);

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_run.step);
}
