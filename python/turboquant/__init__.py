"""TurboQuant: Near-optimal vector compression with fast dot product estimation."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from . import _ffi

if TYPE_CHECKING:
    from numpy.typing import NDArray

__version__ = "0.1.0"
__all__ = ["Engine"]


class Engine:
    """TurboQuant vector compression engine.

    Args:
        dim: Vector dimension (must be a positive even integer).
        seed: Random seed for the rotation matrix.

    Usage::

        engine = Engine(dim=128, seed=42)
        compressed = engine.encode(vector)
        decoded = engine.decode(compressed)
        score = engine.dot(query, compressed)
        engine.close()

    Or as a context manager::

        with Engine(dim=128) as engine:
            compressed = engine.encode(vector)
    """

    def __init__(self, dim: int, seed: int = 42) -> None:
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim must be a positive even integer, got {dim}")
        self._dim = dim
        self._handle = _ffi._lib.tq_engine_create(dim, seed)
        if not self._handle:
            raise MemoryError("Failed to create TurboQuant engine")

    @property
    def dim(self) -> int:
        """Vector dimension this engine was created with."""
        return self._dim

    def encode(self, vector: NDArray[np.float32]) -> bytes:
        """Compress a float32 vector.

        Args:
            vector: 1-D float32 array of length ``dim``.

        Returns:
            Compressed bytes (~6x smaller than the input).

        Raises:
            ValueError: If vector shape doesn't match engine dim.
            RuntimeError: On compression failure.
        """
        self._check_open()
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        if vector.shape != (self._dim,):
            raise ValueError(
                f"Expected shape ({self._dim},), got {vector.shape}"
            )

        buf = _ffi.TqBuffer()
        data_ptr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = _ffi._lib.tq_encode(self._handle, data_ptr, ctypes.byref(buf))
        _ffi.check_rc(rc)

        result = bytes(
            ctypes.cast(
                buf.data, ctypes.POINTER(ctypes.c_uint8 * buf.len)
            ).contents
        )
        _ffi._lib.tq_free_buffer(ctypes.byref(buf))
        return result

    def decode(self, compressed: bytes) -> NDArray[np.float32]:
        """Decompress bytes back to a float32 vector.

        Args:
            compressed: Bytes returned by :meth:`encode`.

        Returns:
            1-D float32 array of length ``dim``.

        Raises:
            RuntimeError: On decompression failure.
        """
        self._check_open()
        out = np.empty(self._dim, dtype=np.float32)
        cbuf = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = _ffi._lib.tq_decode(
            self._handle, cbuf, len(compressed), out_ptr
        )
        _ffi.check_rc(rc)
        return out

    def dot(self, query: NDArray[np.float32], compressed: bytes) -> float:
        """Estimate dot product without full decompression.

        Args:
            query: 1-D float32 array of length ``dim``.
            compressed: Bytes returned by :meth:`encode`.

        Returns:
            Estimated dot product (float).

        Raises:
            ValueError: If query shape doesn't match engine dim.
        """
        self._check_open()
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self._dim,):
            raise ValueError(
                f"Expected shape ({self._dim},), got {query.shape}"
            )

        cbuf = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        q_ptr = query.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return float(
            _ffi._lib.tq_dot(self._handle, q_ptr, cbuf, len(compressed))
        )

    def close(self) -> None:
        """Release engine resources. Safe to call multiple times."""
        if self._handle:
            _ffi._lib.tq_engine_destroy(self._handle)
            self._handle = None

    def _check_open(self) -> None:
        if not self._handle:
            raise RuntimeError("Engine is closed")

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self._handle else "closed"
        return f"Engine(dim={self._dim}, {state})"
