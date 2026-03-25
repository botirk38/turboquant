const std = @import("std");
const turboquant = @import("turboquant.zig");

const ProfileError = error{
    MissingArgs,
    InvalidOp,
    InvalidDim,
    InvalidIterations,
    OddDimension,
};

const Operation = enum {
    encode,
    decode,
    dot,
};

const Config = struct {
    op: Operation,
    dim: usize,
    iterations: usize,
    seed: u32,
};

fn parseArgs(args: []const [:0]u8) ProfileError!Config {
    if (args.len < 3) return ProfileError.MissingArgs;

    const op_str = std.mem.sliceTo(args[1], 0);
    const op: Operation = if (std.mem.eql(u8, op_str, "encode")) Operation.encode else if (std.mem.eql(u8, op_str, "decode")) Operation.decode else if (std.mem.eql(u8, op_str, "dot")) Operation.dot else return ProfileError.InvalidOp;

    const dim_str = std.mem.sliceTo(args[2], 0);
    const dim = std.fmt.parseInt(usize, dim_str, 10) catch {
        return ProfileError.InvalidDim;
    };
    if (dim == 0) return ProfileError.InvalidDim;
    if (dim % 2 != 0) return ProfileError.OddDimension;

    const iterations: usize = if (args.len > 3) blk: {
        const iter_str = std.mem.sliceTo(args[3], 0);
        break :blk std.fmt.parseInt(usize, iter_str, 10) catch {
            return ProfileError.InvalidIterations;
        };
    } else 1000;

    return .{
        .op = op,
        .dim = dim,
        .iterations = iterations,
        .seed = 12345,
    };
}

fn generateVector(allocator: std.mem.Allocator, dim: usize, seed: u32) ![]f32 {
    const data = try allocator.alloc(f32, dim);
    errdefer allocator.free(data);

    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (data) |*v| {
        v.* = r.float(f32) * 10 - 5;
    }

    return data;
}

fn runEncode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
        for (compressed) |b| {
            checksum += @as(f32, @floatFromInt(b));
        }
        allocator.free(compressed);
    }
    return checksum;
}

fn runDecode(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        const decoded = try turboquant.decode(allocator, compressed, seed);
        for (decoded) |v| {
            checksum += v;
        }
        allocator.free(decoded);
    }
    return checksum;
}

fn runDot(allocator: std.mem.Allocator, dim: usize, iterations: usize, seed: u32) !f32 {
    const data = try generateVector(allocator, dim, seed);
    defer allocator.free(data);

    const query = try generateVector(allocator, dim, seed + 1);
    defer allocator.free(query);

    const compressed = try turboquant.encode(allocator, data, .{ .seed = seed });
    defer allocator.free(compressed);

    var checksum: f32 = 0;
    for (0..iterations) |_| {
        checksum += turboquant.dot(query, compressed, seed);
    }
    return checksum;
}

pub fn main() void {
    const args = std.process.argsAlloc(std.heap.page_allocator) catch {
        std.debug.print("error: out of memory parsing args\n", .{});
        return;
    };
    defer std.process.argsFree(std.heap.page_allocator, args);

    const config = parseArgs(args) catch |err| {
        switch (err) {
            ProfileError.MissingArgs => {
                std.debug.print("Usage: profile <op> <dim> [iterations]\n", .{});
                std.debug.print("  op: encode, decode, dot\n", .{});
                std.debug.print("  dim: vector dimension (must be even)\n", .{});
                std.debug.print("  iterations: default 1000\n", .{});
            },
            ProfileError.InvalidOp => {
                std.debug.print("error: invalid operation '{s}'\n", .{std.mem.sliceTo(args[1], 0)});
            },
            ProfileError.InvalidDim => {
                std.debug.print("error: invalid dimension '{s}'\n", .{std.mem.sliceTo(args[2], 0)});
            },
            ProfileError.InvalidIterations => {
                std.debug.print("error: invalid iterations '{s}'\n", .{std.mem.sliceTo(args[3], 0)});
            },
            ProfileError.OddDimension => {
                std.debug.print("error: dimension must be even\n", .{});
            },
        }
        return;
    };

    var timer = std.time.Timer.start() catch {
        std.debug.print("error: could not start timer\n", .{});
        return;
    };

    const result: f32 = switch (config.op) {
        .encode => runEncode(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("encode error: {}\n", .{err});
            return;
        },
        .decode => runDecode(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("decode error: {}\n", .{err});
            return;
        },
        .dot => runDot(std.heap.page_allocator, config.dim, config.iterations, config.seed) catch |err| {
            std.debug.print("dot error: {}\n", .{err});
            return;
        },
    };

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const per_op_us = elapsed_ms * 1000.0 / @as(f64, @floatFromInt(config.iterations));

    std.debug.print("checksum: {e}\n", .{result});
    std.debug.print("time: {d:.2}ms total, {d:.2}us/op ({d} iterations)\n", .{ elapsed_ms, per_op_us, config.iterations });
}
