const std = @import("std");
const tq = @import("turboquant.zig");

const allocator = std.heap.c_allocator;

// Version
const VERSION_MAJOR: u32 = 0;
const VERSION_MINOR: u32 = 1;
const VERSION_PATCH: u32 = 0;

// Error codes
const TQ_OK: c_int = 0;
const TQ_ERR_INVALID_DIM: c_int = -1;
const TQ_ERR_OUT_OF_MEMORY: c_int = -2;
const TQ_ERR_INVALID_DATA: c_int = -3;
const TQ_ERR_NULL_PTR: c_int = -4;

const TqBuffer = extern struct {
    data: ?[*]u8,
    len: usize,
};

export fn tq_version() u32 {
    return (VERSION_MAJOR << 16) | (VERSION_MINOR << 8) | VERSION_PATCH;
}

export fn tq_engine_create(dim: u32, seed: u32) ?*anyopaque {
    const engine = allocator.create(tq.Engine) catch return null;
    engine.* = tq.Engine.init(allocator, .{
        .dim = @as(usize, dim),
        .seed = seed,
    }) catch {
        allocator.destroy(engine);
        return null;
    };
    return @ptrCast(engine);
}

export fn tq_engine_destroy(handle: ?*anyopaque) void {
    const engine = castEngine(handle) orelse return;
    engine.deinit(allocator);
    allocator.destroy(engine);
}

export fn tq_engine_dim(handle: ?*anyopaque) u32 {
    const engine = castEngine(handle) orelse return 0;
    return @intCast(engine.dim);
}

export fn tq_encode(handle: ?*anyopaque, data: ?[*]const f32, out: ?*TqBuffer) c_int {
    const engine = castEngine(handle) orelse return TQ_ERR_NULL_PTR;
    const data_ptr = data orelse return TQ_ERR_NULL_PTR;
    const out_ptr = out orelse return TQ_ERR_NULL_PTR;

    const input = data_ptr[0..engine.dim];
    const compressed = engine.encode(allocator, input) catch |err| return switch (err) {
        error.InvalidDimension => TQ_ERR_INVALID_DIM,
        error.OutOfMemory => TQ_ERR_OUT_OF_MEMORY,
    };

    out_ptr.data = compressed.ptr;
    out_ptr.len = compressed.len;
    return TQ_OK;
}

export fn tq_decode(handle: ?*anyopaque, compressed: ?[*]const u8, compressed_len: usize, out_data: ?[*]f32) c_int {
    const engine = castEngine(handle) orelse return TQ_ERR_NULL_PTR;
    const comp_ptr = compressed orelse return TQ_ERR_NULL_PTR;
    const out_ptr = out_data orelse return TQ_ERR_NULL_PTR;

    const input = comp_ptr[0..compressed_len];
    const decoded = engine.decode(allocator, input) catch |err| return switch (err) {
        error.InvalidHeader, error.InvalidPayload => TQ_ERR_INVALID_DATA,
        error.OutOfMemory => TQ_ERR_OUT_OF_MEMORY,
    };
    defer allocator.free(decoded);

    @memcpy(out_ptr[0..engine.dim], decoded);
    return TQ_OK;
}

export fn tq_dot(handle: ?*anyopaque, query: ?[*]const f32, compressed: ?[*]const u8, compressed_len: usize) f32 {
    const engine = castEngine(handle) orelse return 0.0;
    const q_ptr = query orelse return 0.0;
    const c_ptr = compressed orelse return 0.0;

    return engine.dot(q_ptr[0..engine.dim], c_ptr[0..compressed_len]);
}

export fn tq_free_buffer(buf: ?*TqBuffer) void {
    const b = buf orelse return;
    if (b.data) |ptr| {
        allocator.free(ptr[0..b.len]);
        b.data = null;
        b.len = 0;
    }
}

fn castEngine(handle: ?*anyopaque) ?*tq.Engine {
    return @ptrCast(@alignCast(handle orelse return null));
}
