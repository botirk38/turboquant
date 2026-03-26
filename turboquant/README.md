# TurboQuant

A Zig implementation of Google's TurboQuant vector compression library.

## Installation

Add to your `build.zig.zon`:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/botirk38/turboquant/archive/refs/tags/v0.1.0.tar.gz",
            .hash = "sha256-...",
        },
    },
}
```

To get the hash, run `zig fetch --save https://github.com/botirk38/turboquant/archive/refs/tags/v0.1.0.tar.gz` after adding the URL, and Zig will provide the correct hash.

Or use the latest version from the main branch:

```zig
.{
    .name = "your-project",
    .dependencies = .{
        .turboquant = .{
            .url = "https://github.com/botirk38/turboquant/archive/refs/heads/master.tar.gz",
            .hash = "sha256-...",
        },
    },
}
```

## Usage

```zig
const turboquant = @import("turboquant");

// Create an engine for repeated operations
var engine = try turboquant.Engine.init(allocator, .{ .dim = 1024, .seed = 12345 });
defer engine.deinit(allocator);

// Encode
const compressed = try engine.encode(allocator, my_vector);
defer allocator.free(compressed);

// Decode
const decoded = try engine.decode(allocator, compressed);
defer allocator.free(decoded);

// Fast dot without decode
const score = engine.dot(query_vector, compressed);
```

## API

- `Engine.init(allocator, .{ .dim, .seed })` - Create engine
- `engine.deinit(allocator)` - Destroy engine
- `engine.encode(allocator, vector)` - Compress vector
- `engine.decode(allocator, compressed)` - Decompress
- `engine.dot(query, compressed)` - Dot product without full decode

## Performance

![Performance](docs/assets/performance.png)

At dim=1024: encode 2105µs, decode 1032µs, dot 997µs

## Compression

![Compression Ratio](docs/assets/compression-ratio.png)

![Bits per Dimension](docs/assets/bits-per-dimension.png)

- ~6x compression ratio at dim=1024
- ~3 bits/dim

## Building

```bash
cd turboquant
zig build-exe -O ReleaseFast -target aarch64-macos src/profile.zig
```

## License

MIT
