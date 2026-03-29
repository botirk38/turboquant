"""Low-level ctypes bindings to libturboquant. Private module."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path


class TqBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


def _load_lib() -> ctypes.CDLL:
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    name = f"libturboquant{ext}"

    # Search order: package lib/, zig-out relative to repo root
    search_paths = [
        Path(__file__).parent / "lib" / name,
        Path(__file__).parent.parent.parent / "turboquant" / "zig-out" / "lib" / name,
    ]

    for path in search_paths:
        if path.exists():
            return ctypes.CDLL(str(path))

    raise OSError(
        f"libturboquant not found. Build it first:\n"
        f"  cd turboquant && zig build lib\n"
        f"Then copy or run: python build_lib.py\n"
        f"Searched: {[str(p) for p in search_paths]}"
    )


_lib = _load_lib()

# --- Function signatures ---

_lib.tq_version.argtypes = []
_lib.tq_version.restype = ctypes.c_uint32

_lib.tq_engine_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_lib.tq_engine_create.restype = ctypes.c_void_p

_lib.tq_engine_destroy.argtypes = [ctypes.c_void_p]
_lib.tq_engine_destroy.restype = None

_lib.tq_engine_dim.argtypes = [ctypes.c_void_p]
_lib.tq_engine_dim.restype = ctypes.c_uint32

_lib.tq_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(TqBuffer),
]
_lib.tq_encode.restype = ctypes.c_int

_lib.tq_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
]
_lib.tq_decode.restype = ctypes.c_int

_lib.tq_dot.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
]
_lib.tq_dot.restype = ctypes.c_float

_lib.tq_free_buffer.argtypes = [ctypes.POINTER(TqBuffer)]
_lib.tq_free_buffer.restype = None


# --- Error code mapping ---

TQ_OK = 0
TQ_ERR_INVALID_DIM = -1
TQ_ERR_OUT_OF_MEMORY = -2
TQ_ERR_INVALID_DATA = -3
TQ_ERR_NULL_PTR = -4

ERROR_MESSAGES = {
    TQ_ERR_INVALID_DIM: "Invalid dimension (must be even and > 0)",
    TQ_ERR_OUT_OF_MEMORY: "Out of memory",
    TQ_ERR_INVALID_DATA: "Invalid compressed data",
    TQ_ERR_NULL_PTR: "Null pointer",
}


def check_rc(rc: int) -> None:
    if rc != TQ_OK:
        msg = ERROR_MESSAGES.get(rc, f"Unknown error ({rc})")
        raise RuntimeError(f"TurboQuant error: {msg}")
