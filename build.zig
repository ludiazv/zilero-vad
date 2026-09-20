const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The weight generator runs on the host so cross-compiling the library
    // still works.
    const gen_exe = b.addExecutable(.{
        .name = "gen_weights",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_weights.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    const gen = b.addRunArtifact(gen_exe);
    gen.addFileArg(b.path("model/silero_vad_16k.safetensors"));
    const weights_zig = gen.addOutputFileArg("weights.zig");

    // The library module is the public "zilero" module; the CLI and the tests
    // use the same object.
    const zilero_mod = b.addModule("zilero", .{
        .root_source_file = b.path("src/zilero.zig"),
        .target = target,
        .optimize = optimize,
    });
    zilero_mod.addAnonymousImport("weights", .{ .root_source_file = weights_zig });

    const cli = b.addExecutable(.{
        .name = "zilero-cli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zilero", .module = zilero_mod }},
        }),
    });
    b.installArtifact(cli);

    const tests = b.addTest(.{ .root_module = zilero_mod });
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    const run_step = b.step("run", "Run the CLI (reads PCM from stdin)");
    const run_cmd = b.addRunArtifact(cli);
    run_step.dependOn(&run_cmd.step);
}
