const std = @import("std");
const rotation = @import("rotation.zig");
const math = @import("math.zig");

pub const QjlError = error{ InvalidDimension, OutOfMemory };

const SQRT_PI_OVER_2: f32 = 1.2533141373155003;

pub const Workspace = struct {
    projected: []f32,
    sign_vec: []f32,
    st_sign: []f32,

    pub fn init(allocator: std.mem.Allocator, dim: usize) !Workspace {
        return .{
            .projected = try allocator.alloc(f32, dim),
            .sign_vec = try allocator.alloc(f32, dim),
            .st_sign = try allocator.alloc(f32, dim),
        };
    }

    pub fn deinit(w: *Workspace, allocator: std.mem.Allocator) void {
        allocator.free(w.projected);
        allocator.free(w.sign_vec);
        allocator.free(w.st_sign);
        w.* = .{ .projected = &.{}, .sign_vec = &.{}, .st_sign = &.{} };
    }
};

pub fn encode(allocator: std.mem.Allocator, residual: []const f32, seed: u32) QjlError![]u8 {
    const d = residual.len;
    if (d == 0) return QjlError.InvalidDimension;

    const projected = try allocator.alloc(f32, d);
    defer allocator.free(projected);

    rotation.matVecMul(residual, projected, seed);

    const bits_bytes = (d + 7) / 8;
    const result = try allocator.alloc(u8, bits_bytes);
    @memset(result, 0);

    for (projected, 0..) |val, i| {
        if (val > 0) {
            result[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }
    }
    return result;
}

pub fn encodeWithWorkspace(
    allocator: std.mem.Allocator,
    residual: []const f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) QjlError![]u8 {
    const d = residual.len;
    if (d == 0) return QjlError.InvalidDimension;

    rot_op.matVecMul(residual, workspace.projected);

    const bits_bytes = (d + 7) / 8;
    const result = try allocator.alloc(u8, bits_bytes);
    @memset(result, 0);

    for (workspace.projected, 0..) |val, i| {
        if (val > 0) {
            result[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }
    }
    return result;
}

pub fn decode(
    allocator: std.mem.Allocator,
    qjl_bits: []const u8,
    gamma: f32,
    dim: usize,
    seed: u32,
) QjlError![]f32 {
    if (dim == 0) return QjlError.InvalidDimension;

    const sign_vec = try allocator.alloc(f32, dim);
    errdefer allocator.free(sign_vec);

    const st_sign = try allocator.alloc(f32, dim);
    errdefer allocator.free(st_sign);

    rotation.signToVector(qjl_bits, dim, sign_vec);
    rotation.matVecMulTransposed(sign_vec, st_sign, seed);

    const scale = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(dim)) * gamma;

    const result = try allocator.alloc(f32, dim);
    for (0..dim) |i| {
        result[i] = st_sign[i] * scale;
    }

    allocator.free(sign_vec);
    allocator.free(st_sign);
    return result;
}

pub fn decodeInto(
    out: []f32,
    qjl_bits: []const u8,
    gamma: f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) void {
    const dim = out.len;
    if (dim == 0) return;

    rotation.signToVector(qjl_bits, dim, workspace.sign_vec);
    rot_op.matVecMulTransposed(workspace.sign_vec, workspace.st_sign);

    const scale = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(dim)) * gamma;
    for (0..dim) |i| {
        out[i] = workspace.st_sign[i] * scale;
    }
}

pub fn estimateDot(
    q: []const f32,
    qjl_bits: []const u8,
    gamma: f32,
    seed: u32,
) f32 {
    const d = q.len;
    if (d == 0) return 0;

    const projected = std.heap.page_allocator.alloc(f32, d) catch unreachable;
    defer std.heap.page_allocator.free(projected);

    rotation.matVecMul(q, projected, seed);

    var dot_sum: f32 = 0;
    for (0..d) |i| {
        const bit = (qjl_bits[i / 8] >> @intCast(i % 8)) & 1;
        const sign: f32 = if (bit == 1) 1.0 else -1.0;
        dot_sum += projected[i] * sign;
    }

    const scale = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(d)) * gamma;
    return dot_sum * scale;
}

pub fn estimateDotWithWorkspace(
    q: []const f32,
    qjl_bits: []const u8,
    gamma: f32,
    rot_op: *const rotation.RotationOperator,
    workspace: *Workspace,
) f32 {
    const d = q.len;
    if (d == 0) return 0;

    rot_op.matVecMul(q, workspace.projected);

    for (0..d) |i| {
        const bit = (qjl_bits[i / 8] >> @intCast(i % 8)) & 1;
        workspace.sign_vec[i] = if (bit == 1) 1.0 else -1.0;
    }

    const dot_sum = math.dot(workspace.projected, workspace.sign_vec);

    const scale = SQRT_PI_OVER_2 / @as(f32, @floatFromInt(d)) * gamma;
    return dot_sum * scale;
}

test "encode rejects zero dimension" {
    const allocator = std.testing.allocator;
    const data: [0]f32 = .{};
    const result = encode(allocator, &data, 12345);
    try std.testing.expectError(QjlError.InvalidDimension, result);
}

test "decode rejects zero dimension" {
    const allocator = std.testing.allocator;
    const bits = [_]u8{0};
    const result = decode(allocator, &bits, 1.0, 0, 0);
    try std.testing.expectError(QjlError.InvalidDimension, result);
}

test "encoded bit length" {
    const allocator = std.testing.allocator;
    const residual = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const encoded = try encode(allocator, &residual, 99999);
    defer allocator.free(encoded);

    const expected_len = (residual.len + 7) / 8;
    try std.testing.expectEqual(expected_len, encoded.len);
}

test "decode uses seed consistently with encode" {
    const allocator = std.testing.allocator;
    const residual = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const gamma: f32 = 2.0;

    const encoded = try encode(allocator, &residual, 11111);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded, gamma, residual.len, 11111);
    defer allocator.free(decoded);

    for (decoded) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "estimateDot deterministic" {
    const q = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const residual = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const gamma: f32 = 1.5;
    const seed: u32 = 54321;

    const allocator = std.testing.allocator;
    const encoded = try encode(allocator, &residual, seed);
    defer allocator.free(encoded);

    const est1 = estimateDot(&q, encoded, gamma, seed);
    const est2 = estimateDot(&q, encoded, gamma, seed);

    try std.testing.expectEqual(est1, est2);
}

test "zero gamma gives zero decoded residual" {
    const allocator = std.testing.allocator;
    const bits = [_]u8{ 0xFF, 0xFF };

    const decoded = try decode(allocator, &bits, 0.0, 16, 0);
    defer allocator.free(decoded);

    for (decoded) |v| {
        try std.testing.expectEqual(0.0, v);
    }
}

test "zero gamma gives zero estimated dot" {
    const q = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bits = [_]u8{ 0xAA, 0x55 };

    const est = estimateDot(&q, &bits, 0.0, 12345);
    try std.testing.expectEqual(0.0, est);
}

test "encode decode reconstruction" {
    const allocator = std.testing.allocator;
    const residual = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    const gamma = std.math.sqrt(1.0 + 4.0 + 9.0 + 16.0);
    const seed: u32 = 99999;

    const encoded = try encode(allocator, &residual, seed);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded, gamma, residual.len, seed);
    defer allocator.free(decoded);

    for (decoded) |d| {
        try std.testing.expect(std.math.isFinite(d));
    }
}

test "Workspace init and deinit" {
    const allocator = std.testing.allocator;
    var ws = try Workspace.init(allocator, 128);
    defer ws.deinit(allocator);

    try std.testing.expectEqual(128, ws.projected.len);
    try std.testing.expectEqual(128, ws.sign_vec.len);
    try std.testing.expectEqual(128, ws.st_sign.len);
}
