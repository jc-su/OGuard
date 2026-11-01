import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("string_reverse_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(offsets: np.ndarray, chars: np.ndarray) -> np.ndarray:
    out = np.zeros_like(chars)
    p = np.arange(chars.shape[0], dtype=np.int64)
    row = np.searchsorted(offsets, p, side="right") - 1
    out[offsets[row].astype(np.int64) + offsets[row + 1] - 1 - p] = chars
    return out


def reference_loop(offsets: np.ndarray, chars: np.ndarray) -> np.ndarray:
    out = np.zeros_like(chars)
    for r in range(offsets.shape[0] - 1):
        out[offsets[r] : offsets[r + 1]] = chars[offsets[r] : offsets[r + 1]][::-1]
    return out


def test_compile_ptx() -> None:
    mod = load()
    cuda.compile_ptx(mod.kernel, mod.TYPES)


def test_string_reverse() -> None:
    mod = load()
    rng = np.random.default_rng(5)
    for n in SIZES:
        lengths = rng.integers(0, 21, size=n)
        if n > 1:
            lengths[1] = 3000
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(lengths)
        total = int(offsets[n])
        chars = rng.integers(0, 256, size=total, dtype=np.uint8)
        expected = reference(offsets, chars)
        if n <= 1000:
            assert np.array_equal(expected, reference_loop(offsets, chars))
        d_offsets = cuda.to_device(offsets)
        d_chars = cuda.to_device(chars)
        d_out = cuda.to_device(np.zeros(total, dtype=np.uint8))
        mod.run(d_offsets, d_chars, d_out, n)
        cuda.synchronize()
        got = d_out.copy_to_host()
        assert np.array_equal(got, expected), n


if __name__ == "__main__":
    test_compile_ptx()
    test_string_reverse()
    print("string_reverse numba: PASS")
