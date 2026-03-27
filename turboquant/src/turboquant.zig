const std = @import("std");
const log = std.log.scoped(.turboquant);

pub const format = @import("format.zig");
const rotation = @import("rotation.zig");
pub const math = @import("math.zig");
pub const polar = @import("polar.zig");
const qjl = @import("qjl.zig");

pub const EncodeError = error{ InvalidDimension, OutOfMemory };
pub const DecodeError = error{ InvalidHeader, InvalidPayload, OutOfMemory };

pub const EngineConfig = struct {
    dim: usize,
    seed: u32,
};

pub const Engine = struct {
    dim: usize,
    seed: u32,
    rot_op: rotation.RotationOperator,
    qjl_workspace: qjl.Workspace,
    scratch_rotated: []f32,
    scratch_residual: []f32,
    scratch_polar_decoded: []f32,
    scratch_qjl_decoded: []f32,

    pub fn init(allocator: std.mem.Allocator, config: EngineConfig) !Engine {
        const dim = config.dim;
        if (dim == 0 or dim % 2 != 0) return EncodeError.InvalidDimension;

        var rot_op = try rotation.RotationOperator.prepare(allocator, dim, config.seed);
        errdefer rot_op.destroy(allocator);

        var qjl_workspace = try qjl.Workspace.init(allocator, dim);
        errdefer qjl_workspace.deinit(allocator);

        const scratch_rotated = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_rotated);

        const scratch_residual = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_residual);

        const scratch_polar_decoded = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_polar_decoded);

        const scratch_qjl_decoded = try allocator.alloc(f32, dim);
        errdefer allocator.free(scratch_qjl_decoded);

        return .{
            .dim = dim,
            .seed = config.seed,
            .rot_op = rot_op,
            .qjl_workspace = qjl_workspace,
            .scratch_rotated = scratch_rotated,
            .scratch_residual = scratch_residual,
            .scratch_polar_decoded = scratch_polar_decoded,
            .scratch_qjl_decoded = scratch_qjl_decoded,
        };
    }

    pub fn deinit(e: *Engine, allocator: std.mem.Allocator) void {
        e.rot_op.destroy(allocator);
        e.qjl_workspace.deinit(allocator);
        allocator.free(e.scratch_rotated);
        allocator.free(e.scratch_residual);
        allocator.free(e.scratch_polar_decoded);
        allocator.free(e.scratch_qjl_decoded);
        e.* = undefined;
    }

    pub fn encode(e: *Engine, allocator: std.mem.Allocator, x: []const f32) ![]u8 {
        const dim = e.dim;
        if (x.len != dim) return EncodeError.InvalidDimension;

        e.rot_op.rotate(x, e.scratch_rotated);

        var max_r: f32 = 0;
        for (0..dim / 2) |i| {
            const r = math.norm(e.scratch_rotated[i * 2 .. i * 2 + 2]);
            if (r > max_r) max_r = r;
        }
        if (max_r == 0) max_r = 1.0;

        const polar_encoded = try polar.encode(allocator, e.scratch_rotated, max_r);
        errdefer allocator.free(polar_encoded);

        computeResidualFromPolar(polar_encoded, e.scratch_rotated, max_r, e.scratch_residual);

        const gamma = math.norm(e.scratch_residual);
        const qjl_encoded = try qjl.encodeWithWorkspace(allocator, e.scratch_residual, &e.rot_op, &e.qjl_workspace);
        errdefer allocator.free(qjl_encoded);

        const polar_bytes = @as(u32, @intCast(polar_encoded.len));
        const qjl_bytes = @as(u32, @intCast(qjl_encoded.len));
        const total_size = format.HEADER_SIZE + polar_encoded.len + qjl_encoded.len;

        const result = try allocator.alloc(u8, total_size);
        errdefer allocator.free(result);

        format.writeHeader(result, @intCast(dim), polar_bytes, qjl_bytes, max_r, gamma);
        @memcpy(result[format.HEADER_SIZE..][0..polar_encoded.len], polar_encoded);
        @memcpy(result[format.HEADER_SIZE + polar_encoded.len ..], qjl_encoded);
        allocator.free(polar_encoded);
        allocator.free(qjl_encoded);

        const bpd = (total_size - format.HEADER_SIZE) * 8 / dim;
        log.debug("encoded: dim={}, bytes={}, bits/dim={}", .{ dim, total_size, bpd });

        return result;
    }

    pub fn decode(e: *Engine, allocator: std.mem.Allocator, compressed: []const u8) ![]f32 {
        const header = format.readHeader(compressed) catch |err| switch (err) {
            error.InvalidHeader => return DecodeError.InvalidHeader,
            error.OutOfMemory => return DecodeError.OutOfMemory,
            error.InvalidPayload => return DecodeError.InvalidPayload,
        };
        const dim = e.dim;
        if (header.dim != dim) return DecodeError.InvalidPayload;

        const payload = format.slicePayload(compressed, header) catch |err| switch (err) {
            error.InvalidHeader => return DecodeError.InvalidHeader,
            error.OutOfMemory => return DecodeError.OutOfMemory,
            error.InvalidPayload => return DecodeError.InvalidPayload,
        };

        const polar_decoded = try allocator.alloc(f32, e.dim);
        errdefer allocator.free(polar_decoded);

        polar.decodeInto(polar_decoded, payload.polar, header.max_r) catch |err| switch (err) {
            error.InvalidDimension => return DecodeError.InvalidPayload,
            error.OutOfMemory => return DecodeError.OutOfMemory,
        };

        qjl.decodeInto(e.scratch_qjl_decoded, payload.qjl, header.gamma, &e.rot_op, &e.qjl_workspace);

        math.addInPlace(polar_decoded, e.scratch_qjl_decoded);

        return polar_decoded;
    }

    pub fn dot(e: *Engine, q: []const f32, compressed: []const u8) f32 {
        const header = format.readHeader(compressed) catch return 0;
        if (q.len != e.dim or header.dim != e.dim) return 0;

        const payload = format.slicePayload(compressed, header) catch return 0;

        const polar_sum = polar.dotProduct(q, payload.polar, header.max_r);
        const qjl_sum = qjl.estimateDotWithWorkspace(q, payload.qjl, header.gamma, &e.rot_op, &e.qjl_workspace);

        return polar_sum + qjl_sum;
    }
};

fn computeResidualFromPolar(polar_encoded: []const u8, rotated: []const f32, max_r: f32, residual: []f32) void {
    const dim = rotated.len;
    const num_pairs = dim / 2;

    var bit_pos: usize = 0;
    for (0..num_pairs) |i| {
        const pair = polar.reconstructPair(polar_encoded, bit_pos, max_r);
        bit_pos += 7;

        residual[i * 2] = rotated[i * 2] - pair.dx;
        residual[i * 2 + 1] = rotated[i * 2 + 1] - pair.dy;
    }
}

test "roundtrip" {
    const allocator = std.testing.allocator;
    const seed: u32 = 12345;
    const dim: usize = 8;

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const x: [8]f32 = .{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const q: [8]f32 = .{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

    var true_dot: f32 = 0;
    for (x, q) |xv, qv| true_dot += xv * qv;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    log.info("{} bytes ({} bits/dim)", .{ compressed.len, (compressed.len - format.HEADER_SIZE) * 8 / dim });

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    var decoded_dot: f32 = 0;
    for (decoded, q) |dv, qv| decoded_dot += dv * qv;

    const cdot = engine.dot(&q, compressed);
    log.info("true={e}, decoded_dot={e}, direct_dot={e}", .{ true_dot, decoded_dot, cdot });
    try std.testing.expect(@abs(true_dot - cdot) < 50.0);
}

test "compression ratio" {
    const allocator = std.testing.allocator;
    const seed: u32 = 12345;
    const dim: usize = 128;

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(1234);
    const r = rng.random();

    var x: [128]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const bpd = (compressed.len - format.HEADER_SIZE) * 8 / 128;
    log.info("dim=128, bytes={}, bits/dim={}", .{ compressed.len, bpd });
    try std.testing.expect(bpd <= 4);
}

test "init rejects zero dimension" {
    const allocator = std.testing.allocator;
    const result = Engine.init(allocator, .{ .dim = 0, .seed = 12345 });
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "init rejects odd dimension" {
    const allocator = std.testing.allocator;
    const result = Engine.init(allocator, .{ .dim = 7, .seed = 12345 });
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "encode rejects wrong dimension" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    const result = engine.encode(allocator, &x);
    try std.testing.expectError(EncodeError.InvalidDimension, result);
}

test "decode rejects truncated header" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 128, .seed = 12345 });
    defer engine.deinit(allocator);

    const short: [5]u8 = .{ 1, 0, 0, 0, 0 };
    const result = engine.decode(allocator, &short);
    try std.testing.expectError(DecodeError.InvalidHeader, result);
}

test "decode rejects truncated payload" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 128, .seed = 12345 });
    defer engine.deinit(allocator);

    var buf: [118]u8 = undefined;
    format.writeHeader(&buf, 128, 1000, 100, 1.0, 0.5);
    const result = engine.decode(allocator, &buf);
    try std.testing.expectError(DecodeError.InvalidPayload, result);
}

test "dot returns zero on dimension mismatch" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 8, .seed = 12345 });
    defer engine.deinit(allocator);

    const x: [8]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const wrong_dim: [16]f32 = .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    const result = engine.dot(&wrong_dim, compressed);
    try std.testing.expectEqual(0.0, result);
}

test "roundtrip correct length and finite" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 64, .seed = 9999 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(9999);
    const r = rng.random();

    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    try std.testing.expectEqual(x.len, decoded.len);
    for (decoded) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "roundtrip multiple dims" {
    const allocator = std.testing.allocator;
    const seed: u32 = 8888;

    const dims = [_]usize{ 8, 16, 32, 64, 128 };

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(allocator);

        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();

        var x: [128]f32 = undefined;
        for (0..dim) |i| x[i] = r.float(f32) * 10 - 5;

        const compressed = try engine.encode(allocator, x[0..dim]);
        defer allocator.free(compressed);

        const decoded = try engine.decode(allocator, compressed);
        defer allocator.free(decoded);

        try std.testing.expectEqual(dim, decoded.len);
    }
}

test "dot close to decoded dot" {
    const allocator = std.testing.allocator;
    var engine = try Engine.init(allocator, .{ .dim = 64, .seed = 7777 });
    defer engine.deinit(allocator);

    var rng = std.Random.DefaultPrng.init(7777);
    const r = rng.random();

    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = r.float(f32) * 10 - 5;

    var q: [64]f32 = undefined;
    for (&q) |*v| v.* = r.float(f32);

    const compressed = try engine.encode(allocator, &x);
    defer allocator.free(compressed);

    const decoded = try engine.decode(allocator, compressed);
    defer allocator.free(decoded);

    var decoded_dot: f32 = 0;
    for (decoded, q) |dv, qv| decoded_dot += dv * qv;

    const direct_dot = engine.dot(&q, compressed);

    const rel_err = @abs(decoded_dot - direct_dot) / (@abs(decoded_dot) + 1e-10);
    log.info("decoded_dot={e}, direct_dot={e}, rel_err={e}", .{ decoded_dot, direct_dot, rel_err });
    try std.testing.expect(rel_err < 0.5);
}

test "benchmark encode" {
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };
    const seed: u32 = 12345;

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [1024]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
        }

        var timer = std.time.Timer.start() catch unreachable;
        const iterations = 100;
        for (0..iterations) |_| {
            const compressed = engine.encode(std.testing.allocator, data[0..dim]) catch unreachable;
            std.testing.allocator.free(compressed);
        }
        const ns = timer.read();
        const ns_per_op = ns / iterations;
        const bytes = (dim * 4) / 2 + (dim + 7) / 8 + 23;
        std.debug.print("encode/dim={:4}: {:9} ns/op  ({} bytes)\n", .{ dim, ns_per_op, bytes });
    }
}

test "benchmark decode" {
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };
    const seed: u32 = 12345;

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [1024]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
        }

        const compressed = try engine.encode(std.testing.allocator, data[0..dim]);
        defer std.testing.allocator.free(compressed);

        var timer = std.time.Timer.start() catch unreachable;
        const iterations = 100;
        for (0..iterations) |_| {
            const decoded = engine.decode(std.testing.allocator, compressed) catch unreachable;
            std.testing.allocator.free(decoded);
        }
        const ns = timer.read();
        const ns_per_op = ns / iterations;
        std.debug.print("decode/dim={:4}: {:9} ns/op\n", .{ dim, ns_per_op });
    }
}

test "benchmark dot" {
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };
    const seed: u32 = 12345;

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [1024]f32 = undefined;
        var query: [1024]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
            query[i] = r.float(f32);
        }

        const compressed = try engine.encode(std.testing.allocator, data[0..dim]);
        defer std.testing.allocator.free(compressed);

        var timer = std.time.Timer.start() catch unreachable;
        const iterations = 100;
        for (0..iterations) |_| {
            _ = engine.dot(query[0..dim], compressed);
        }
        const ns = timer.read();
        const ns_per_op = ns / iterations;
        std.debug.print("dot/dim={:4}: {:9} ns/op\n", .{ dim, ns_per_op });
    }
}

test "benchmark dot decoded" {
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };
    const seed: u32 = 12345;

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [1024]f32 = undefined;
        var query: [1024]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
            query[i] = r.float(f32);
        }

        const compressed = try engine.encode(std.testing.allocator, data[0..dim]);
        defer std.testing.allocator.free(compressed);

        var timer = std.time.Timer.start() catch unreachable;
        const iterations = 100;
        for (0..iterations) |_| {
            const decoded = engine.decode(std.testing.allocator, compressed) catch unreachable;
            var dot_prod: f32 = 0;
            for (0..dim) |i| {
                dot_prod += decoded[i] * query[i];
            }
            std.testing.allocator.free(decoded);
        }
        const ns = timer.read();
        const ns_per_op = ns / iterations;
        std.debug.print("dot_decoded/dim={:4}: {:9} ns/op\n", .{ dim, ns_per_op });
    }
}

test "benchmark compression" {
    const dims = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    const seed: u32 = 12345;

    std.debug.print("\n=== COMPRESSION RATIOS ===\n", .{});
    std.debug.print("{s:>4} | {s:>6} | {s:>6} | {s:>6} | {s:>8} | {s:>8}\n", .{ "dim", "raw(f32)", "compressed", "ratio", "bits/dim", "target" });
    std.debug.print("------|----------|----------|----------|----------|----------\n", .{});

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [4096]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
        }

        const compressed = try engine.encode(std.testing.allocator, data[0..dim]);
        defer std.testing.allocator.free(compressed);

        const raw_bytes = dim * 4;
        const ratio = @as(f64, @floatFromInt(raw_bytes)) / @as(f64, @floatFromInt(compressed.len));
        const bits_per_dim = @as(f64, @floatFromInt(compressed.len * 8)) / @as(f64, @floatFromInt(dim));

        const target_ratio = 6.0;
        const target_met: []const u8 = if (ratio >= target_ratio) "OK" else "LOW";

        std.debug.print("{:>4} | {:>6} | {:>6} | {:>6.2}x | {:>8.2} | {s:>8}\n", .{ dim, raw_bytes, compressed.len, ratio, bits_per_dim, target_met });
    }
}

test "compression breakdown" {
    const dims = [_]usize{ 128, 256, 512, 1024 };
    const seed: u32 = 12345;

    std.debug.print("\n=== COMPRESSION BREAKDOWN ===\n", .{});

    for (dims) |dim| {
        var engine = try Engine.init(std.testing.allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(std.testing.allocator);

        var data: [4096]f32 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();
        for (0..dim) |i| {
            data[i] = r.float(f32) * 10 - 5;
        }

        const compressed = try engine.encode(std.testing.allocator, data[0..dim]);
        defer std.testing.allocator.free(compressed);

        const header = format.HEADER_SIZE;
        const polar_expected = (dim / 2 * 7 + 7) / 8;
        const qjl_expected = (dim + 7) / 8;
        const total_expected = header + polar_expected + qjl_expected;
        const total_actual = compressed.len;
        const overhead = total_actual - total_expected;

        std.debug.print("dim={}: header={}, polar~={}, qjl~={}, expected={}, actual={}, overhead={}\n", .{ dim, header, polar_expected, qjl_expected, total_expected, total_actual, overhead });
    }
}

test "quality MSE distortion" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 64, 128, 256, 512, 1024 };
    const num_vectors: usize = 200;
    const seed: u32 = 42;

    std.debug.print("\n=== MSE DISTORTION (unit-sphere vectors) ===\n", .{});
    std.debug.print("{s:>6} | {s:>12} | {s:>12} | {s:>12} | {s:>12}\n", .{ "dim", "raw_mse", "norm_mse", "polar_only", "paper_bound" });
    std.debug.print("-------|--------------|--------------|--------------|-------------\n", .{});

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(allocator);

        const rotated_buf = try allocator.alloc(f32, dim);
        defer allocator.free(rotated_buf);
        const diff_buf = try allocator.alloc(f32, dim);
        defer allocator.free(diff_buf);
        const polar_buf = try allocator.alloc(f32, dim);
        defer allocator.free(polar_buf);

        var sum_raw_mse: f64 = 0;
        var sum_norm_mse: f64 = 0;
        var sum_polar_norm_mse: f64 = 0;

        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();

        for (0..num_vectors) |_| {
            var x_buf = try allocator.alloc(f32, dim);
            defer allocator.free(x_buf);

            // Generate unit-sphere vector
            var norm_sq: f32 = 0;
            for (0..dim) |i| {
                x_buf[i] = r.float(f32) * 2 - 1;
                norm_sq += x_buf[i] * x_buf[i];
            }
            const inv_norm = 1.0 / @sqrt(norm_sq);
            for (0..dim) |i| x_buf[i] *= inv_norm;

            engine.rot_op.rotate(x_buf, rotated_buf);
            const rotated_norm_sq: f64 = @as(f64, math.dot(rotated_buf, rotated_buf));

            const compressed = try engine.encode(allocator, x_buf);
            defer allocator.free(compressed);

            // Full decode MSE (polar + QJL)
            const decoded = try engine.decode(allocator, compressed);
            defer allocator.free(decoded);
            math.sub(rotated_buf, decoded, diff_buf);
            const raw_mse: f64 = @as(f64, math.dot(diff_buf, diff_buf)) / @as(f64, @floatFromInt(dim));
            const norm_mse: f64 = @as(f64, math.dot(diff_buf, diff_buf)) / rotated_norm_sq;

            // Polar-only normalized MSE
            const header = format.readHeader(compressed) catch unreachable;
            const payload = format.slicePayload(compressed, header) catch unreachable;
            polar.decodeInto(polar_buf, payload.polar, header.max_r) catch unreachable;
            math.sub(rotated_buf, polar_buf, diff_buf);
            const polar_norm_mse: f64 = @as(f64, math.dot(diff_buf, diff_buf)) / rotated_norm_sq;

            sum_raw_mse += raw_mse;
            sum_norm_mse += norm_mse;
            sum_polar_norm_mse += polar_norm_mse;
        }

        const n: f64 = @floatFromInt(num_vectors);
        // Paper bound: D_mse ≤ 2.7 * (1/4^b), b ≈ 3.5 bits/dim
        const paper_bound: f64 = 2.7 * (1.0 / std.math.pow(f64, 4.0, 3.5));

        std.debug.print("{:>6} | {:>12.6} | {:>12.6} | {:>12.6} | {:>12.6}\n", .{
            dim,
            @as(f32, @floatCast(sum_raw_mse / n)),
            @as(f32, @floatCast(sum_norm_mse / n)),
            @as(f32, @floatCast(sum_polar_norm_mse / n)),
            @as(f32, @floatCast(paper_bound)),
        });
    }
}

test "quality inner product distortion" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 64, 128, 256, 512, 1024 };
    const num_pairs: usize = 200;
    const seed: u32 = 42;

    std.debug.print("\n=== INNER PRODUCT DISTORTION (unit-sphere vectors) ===\n", .{});
    std.debug.print("{s:>6} | {s:>12} | {s:>12} | {s:>12} | {s:>12}\n", .{ "dim", "mean_sq_err", "mean_rel_err", "polar_sq_err", "polar_rel" });
    std.debug.print("-------|--------------|--------------|--------------|-------------\n", .{});

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(allocator);

        const rotated_buf = try allocator.alloc(f32, dim);
        defer allocator.free(rotated_buf);
        const polar_buf = try allocator.alloc(f32, dim);
        defer allocator.free(polar_buf);

        var sum_sq_err: f64 = 0;
        var sum_rel_err: f64 = 0;
        var sum_polar_sq_err: f64 = 0;
        var sum_polar_rel_err: f64 = 0;

        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();

        for (0..num_pairs) |_| {
            var x_buf = try allocator.alloc(f32, dim);
            defer allocator.free(x_buf);
            var q_buf = try allocator.alloc(f32, dim);
            defer allocator.free(q_buf);

            // Generate unit-sphere x
            var norm_sq: f32 = 0;
            for (0..dim) |i| {
                x_buf[i] = r.float(f32) * 2 - 1;
                norm_sq += x_buf[i] * x_buf[i];
            }
            var inv_norm = 1.0 / @sqrt(norm_sq);
            for (0..dim) |i| x_buf[i] *= inv_norm;

            // Generate unit-sphere q
            norm_sq = 0;
            for (0..dim) |i| {
                q_buf[i] = r.float(f32) * 2 - 1;
                norm_sq += q_buf[i] * q_buf[i];
            }
            inv_norm = 1.0 / @sqrt(norm_sq);
            for (0..dim) |i| q_buf[i] *= inv_norm;

            // Ground truth dot in rotated space
            engine.rot_op.rotate(x_buf, rotated_buf);
            const true_dot: f64 = @as(f64, math.dot(q_buf, rotated_buf));

            const compressed = try engine.encode(allocator, x_buf);
            defer allocator.free(compressed);

            // Full (polar + QJL) dot estimate
            const est_dot: f64 = @as(f64, engine.dot(q_buf, compressed));
            const sq_err = (true_dot - est_dot) * (true_dot - est_dot);
            const rel_err = @abs(true_dot - est_dot) / (@abs(true_dot) + 1e-10);

            // Polar-only dot estimate
            const header = format.readHeader(compressed) catch unreachable;
            const payload = format.slicePayload(compressed, header) catch unreachable;
            const polar_dot: f64 = @as(f64, polar.dotProduct(q_buf, payload.polar, header.max_r));
            const polar_sq = (true_dot - polar_dot) * (true_dot - polar_dot);
            const polar_rel = @abs(true_dot - polar_dot) / (@abs(true_dot) + 1e-10);

            sum_sq_err += sq_err;
            sum_rel_err += rel_err;
            sum_polar_sq_err += polar_sq;
            sum_polar_rel_err += polar_rel;
        }

        const n: f64 = @floatFromInt(num_pairs);
        std.debug.print("{:>6} | {:>12.6} | {:>12.6} | {:>12.6} | {:>12.6}\n", .{
            dim,
            @as(f32, @floatCast(sum_sq_err / n)),
            @as(f32, @floatCast(sum_rel_err / n)),
            @as(f32, @floatCast(sum_polar_sq_err / n)),
            @as(f32, @floatCast(sum_polar_rel_err / n)),
        });
    }
}

test "quality unbiasedness" {
    const allocator = std.testing.allocator;
    const dim: usize = 256;
    const num_vectors: usize = 500;
    const seed: u32 = 42;

    std.debug.print("\n=== UNBIASEDNESS CHECK (dim=256) ===\n", .{});

    var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
    defer engine.deinit(allocator);

    const rotated_buf = try allocator.alloc(f32, dim);
    defer allocator.free(rotated_buf);

    // Fixed query vector (unit-sphere)
    var q_buf = try allocator.alloc(f32, dim);
    defer allocator.free(q_buf);
    var rng = std.Random.DefaultPrng.init(seed + 1);
    const r = rng.random();
    var norm_sq: f32 = 0;
    for (0..dim) |i| {
        q_buf[i] = r.float(f32) * 2 - 1;
        norm_sq += q_buf[i] * q_buf[i];
    }
    var inv_norm = 1.0 / @sqrt(norm_sq);
    for (0..dim) |i| q_buf[i] *= inv_norm;

    var sum_signed_err: f64 = 0;
    var sum_abs_err: f64 = 0;

    var rng2 = std.Random.DefaultPrng.init(seed + 2);
    const r2 = rng2.random();

    for (0..num_vectors) |_| {
        var x_buf = try allocator.alloc(f32, dim);
        defer allocator.free(x_buf);

        norm_sq = 0;
        for (0..dim) |i| {
            x_buf[i] = r2.float(f32) * 2 - 1;
            norm_sq += x_buf[i] * x_buf[i];
        }
        inv_norm = 1.0 / @sqrt(norm_sq);
        for (0..dim) |i| x_buf[i] *= inv_norm;

        engine.rot_op.rotate(x_buf, rotated_buf);
        const true_dot: f64 = @as(f64, math.dot(q_buf, rotated_buf));

        const compressed = try engine.encode(allocator, x_buf);
        defer allocator.free(compressed);

        const est_dot: f64 = @as(f64, engine.dot(q_buf, compressed));
        sum_signed_err += (est_dot - true_dot);
        sum_abs_err += @abs(est_dot - true_dot);
    }

    const mean_bias = sum_signed_err / @as(f64, num_vectors);
    const mean_abs_err = sum_abs_err / @as(f64, num_vectors);

    std.debug.print("vectors:        {}\n", .{num_vectors});
    std.debug.print("mean bias:      {e} (should be ~0)\n", .{@as(f32, @floatCast(mean_bias))});
    std.debug.print("mean |error|:   {e}\n", .{@as(f32, @floatCast(mean_abs_err))});
}

test "quality component analysis" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 128, 256, 512, 1024 };
    const num_vectors: usize = 100;
    const seed: u32 = 42;

    std.debug.print("\n=== COMPONENT ANALYSIS (polar vs polar+QJL) ===\n", .{});
    std.debug.print("{s:>6} | {s:>12} | {s:>12} | {s:>12} | {s:>12}\n", .{ "dim", "polar_mse", "full_mse", "improvement", "gamma/norm" });
    std.debug.print("-------|--------------|--------------|--------------|-------------\n", .{});

    for (dims) |dim| {
        var engine = try Engine.init(allocator, .{ .dim = dim, .seed = seed });
        defer engine.deinit(allocator);

        const rotated_buf = try allocator.alloc(f32, dim);
        defer allocator.free(rotated_buf);
        const diff_buf = try allocator.alloc(f32, dim);
        defer allocator.free(diff_buf);
        const polar_decoded_buf = try allocator.alloc(f32, dim);
        defer allocator.free(polar_decoded_buf);

        var sum_polar_mse: f64 = 0;
        var sum_full_mse: f64 = 0;
        var sum_gamma_ratio: f64 = 0;

        var rng = std.Random.DefaultPrng.init(seed);
        const r = rng.random();

        for (0..num_vectors) |_| {
            var x_buf = try allocator.alloc(f32, dim);
            defer allocator.free(x_buf);

            var norm_sq: f32 = 0;
            for (0..dim) |i| {
                x_buf[i] = r.float(f32) * 2 - 1;
                norm_sq += x_buf[i] * x_buf[i];
            }
            const inv_norm = 1.0 / @sqrt(norm_sq);
            for (0..dim) |i| x_buf[i] *= inv_norm;

            engine.rot_op.rotate(x_buf, rotated_buf);

            const compressed = try engine.encode(allocator, x_buf);
            defer allocator.free(compressed);

            const header = format.readHeader(compressed) catch unreachable;
            const payload = format.slicePayload(compressed, header) catch unreachable;

            // Polar-only MSE
            polar.decodeInto(polar_decoded_buf, payload.polar, header.max_r) catch unreachable;
            math.sub(rotated_buf, polar_decoded_buf, diff_buf);
            const polar_mse: f64 = @as(f64, math.dot(diff_buf, diff_buf)) / @as(f64, @floatFromInt(dim));

            // Full decode MSE
            const decoded = try engine.decode(allocator, compressed);
            defer allocator.free(decoded);
            math.sub(rotated_buf, decoded, diff_buf);
            const full_mse: f64 = @as(f64, math.dot(diff_buf, diff_buf)) / @as(f64, @floatFromInt(dim));

            // Gamma / norm ratio
            const rotated_norm: f64 = @as(f64, math.norm(rotated_buf));
            const gamma_ratio: f64 = @as(f64, header.gamma) / (rotated_norm + 1e-10);

            sum_polar_mse += polar_mse;
            sum_full_mse += full_mse;
            sum_gamma_ratio += gamma_ratio;
        }

        const n: f64 = @floatFromInt(num_vectors);
        const mean_polar = sum_polar_mse / n;
        const mean_full = sum_full_mse / n;
        const improvement = (1.0 - mean_full / mean_polar) * 100.0;

        std.debug.print("{:>6} | {:>12.6} | {:>12.6} | {:>11.1}% | {:>12.6}\n", .{
            dim,
            @as(f32, @floatCast(mean_polar)),
            @as(f32, @floatCast(mean_full)),
            @as(f32, @floatCast(improvement)),
            @as(f32, @floatCast(sum_gamma_ratio / n)),
        });
    }
}
