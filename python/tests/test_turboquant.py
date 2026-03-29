"""Tests for the TurboQuant Python bindings."""

import numpy as np
import pytest

from turboquant import Engine


class TestEngineLifecycle:
    def test_create_and_close(self):
        engine = Engine(dim=64, seed=42)
        assert engine.dim == 64
        assert repr(engine) == "Engine(dim=64, open)"
        engine.close()
        assert repr(engine) == "Engine(dim=64, closed)"

    def test_context_manager(self):
        with Engine(dim=64, seed=42) as engine:
            vec = np.random.randn(64).astype(np.float32)
            compressed = engine.encode(vec)
            assert len(compressed) > 0

    def test_double_close(self):
        engine = Engine(dim=64, seed=42)
        engine.close()
        engine.close()  # should not raise

    def test_use_after_close(self):
        engine = Engine(dim=64, seed=42)
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.encode(np.zeros(64, dtype=np.float32))

    def test_invalid_dim_odd(self):
        with pytest.raises(ValueError):
            Engine(dim=7)

    def test_invalid_dim_zero(self):
        with pytest.raises(ValueError):
            Engine(dim=0)

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError):
            Engine(dim=-4)


class TestEncode:
    def test_roundtrip(self):
        with Engine(dim=64, seed=42) as engine:
            vec = np.random.randn(64).astype(np.float32)
            vec /= np.linalg.norm(vec)
            compressed = engine.encode(vec)
            decoded = engine.decode(compressed)
            assert decoded.shape == (64,)
            assert decoded.dtype == np.float32
            assert np.all(np.isfinite(decoded))

    def test_compression_ratio(self):
        with Engine(dim=1024, seed=42) as engine:
            vec = np.random.randn(1024).astype(np.float32)
            compressed = engine.encode(vec)
            ratio = (1024 * 4) / len(compressed)
            assert ratio > 5.0

    def test_dimension_mismatch(self):
        with Engine(dim=64, seed=42) as engine:
            wrong = np.random.randn(128).astype(np.float32)
            with pytest.raises(ValueError, match="shape"):
                engine.encode(wrong)

    def test_auto_converts_float64(self):
        with Engine(dim=64, seed=42) as engine:
            vec = np.random.randn(64)  # float64 by default
            compressed = engine.encode(vec)  # should auto-convert
            assert len(compressed) > 0

    def test_batch(self):
        with Engine(dim=256, seed=42) as engine:
            vectors = [np.random.randn(256).astype(np.float32) for _ in range(10)]
            compressed = [engine.encode(v) for v in vectors]
            decoded = [engine.decode(c) for c in compressed]
            assert all(d.shape == (256,) for d in decoded)
            assert all(np.all(np.isfinite(d)) for d in decoded)


class TestDot:
    def test_dot_product(self):
        with Engine(dim=128, seed=42) as engine:
            x = np.random.randn(128).astype(np.float32)
            q = np.random.randn(128).astype(np.float32)
            compressed = engine.encode(x)
            score = engine.dot(q, compressed)
            assert isinstance(score, float)
            assert np.isfinite(score)

    def test_dot_dimension_mismatch(self):
        with Engine(dim=64, seed=42) as engine:
            vec = np.random.randn(64).astype(np.float32)
            compressed = engine.encode(vec)
            wrong_query = np.random.randn(128).astype(np.float32)
            with pytest.raises(ValueError, match="shape"):
                engine.dot(wrong_query, compressed)

    def test_multiple_dims(self):
        for dim in [8, 16, 32, 64, 128]:
            with Engine(dim=dim, seed=42) as engine:
                x = np.random.randn(dim).astype(np.float32)
                q = np.random.randn(dim).astype(np.float32)
                compressed = engine.encode(x)
                score = engine.dot(q, compressed)
                assert np.isfinite(score), f"Non-finite score at dim={dim}"
