#!/usr/bin/env python3
"""Build the TurboQuant shared library and copy to the Python package."""

import platform
import shutil
import subprocess
from pathlib import Path


def build() -> None:
    root = Path(__file__).parent.parent
    zig_dir = root / "turboquant"

    print("Building shared library...")
    subprocess.check_call(["zig", "build", "lib"], cwd=zig_dir)

    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    src = zig_dir / "zig-out" / "lib" / f"libturboquant{ext}"
    if not src.exists():
        raise FileNotFoundError(f"Build succeeded but library not found at {src}")

    dst_dir = Path(__file__).parent / "turboquant" / "lib"
    dst_dir.mkdir(exist_ok=True)
    shutil.copy2(src, dst_dir)
    print(f"Copied {src.name} -> {dst_dir / src.name}")


if __name__ == "__main__":
    build()
