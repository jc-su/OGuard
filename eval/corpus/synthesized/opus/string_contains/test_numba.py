"""GPU test of the Numba string_contains kernel against a Python reference.

Run from the model directory: python -P string_contains/test_numba.py
"""

import importlib.util
import pathlib
import warnings
from types import ModuleType

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Small test sizes launch few blocks; the low-occupancy warning is expected.
warnings.simplefilter("ignore", NumbaPerformanceWarning)

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
ALPHABET = np.frombuffer(b"abcd", dtype=np.uint8)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("string_contains_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_column(
    rng: np.random.Generator, n: int, max_short: int, long_fraction: float, max_long: int
) -> tuple[np.ndarray, np.ndarray]:
    lengths = rng.integers(0, max_short + 1, size=n)
    is_long = rng.random(n) < long_fraction
    lengths[is_long] = rng.integers(129, max_long + 1, size=int(is_long.sum()))
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    chars = ALPHABET[rng.integers(0, ALPHABET.size, size=int(offsets[-1]))]
    return offsets, chars


def reference(offsets: np.ndarray, chars: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    data = chars.tobytes()
    needle = pattern.tobytes()
    n = offsets.size - 1
    return np.array([needle in data[offsets[i] : offsets[i + 1]] for i in range(n)], dtype=np.int8)


def check(module: ModuleType, offsets: np.ndarray, chars: np.ndarray, pattern: np.ndarray) -> None:
    n = offsets.size - 1
    k = pattern.size
    d_offsets = cuda.to_device(offsets)
    d_chars = cuda.to_device(chars if chars.size > 0 else np.zeros(1, dtype=np.uint8))
    d_pattern = cuda.to_device(pattern if k > 0 else np.zeros(1, dtype=np.uint8))
    d_match = cuda.to_device(np.full(max(n, 1), -1, dtype=np.int8))
    module.run(d_offsets, d_chars, d_pattern, d_match, n, k)
    np.testing.assert_array_equal(d_match.copy_to_host()[:n], reference(offsets, chars, pattern))


def to_bytes(text: bytes) -> np.ndarray:
    return np.frombuffer(text, dtype=np.uint8).copy()


def test_string_contains() -> None:
    module = load_module()
    rng = np.random.default_rng(4)
    patterns = [to_bytes(p) for p in (b"", b"a", b"cab", b"abcda", b"dddddddd")]
    for n in SIZES:
        # Short strings only, then a mix with long strings (the warp path).
        for max_short, long_fraction in ((20, 0.0), (300, 0.02)):
            offsets, chars = make_column(rng, n, max_short, long_fraction, 3000)
            for pattern in patterns:
                check(module, offsets, chars, pattern)
    # Long strings with the pattern only at the start, only at the end, or
    # longer than the short-string limit.
    body = b"ab" * 700
    column = [b"xyz" + body, body + b"xyz", body + b"xy", b"xyz", b"xy", b"", body * 2]
    lengths = [len(s) for s in column]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    chars = to_bytes(b"".join(column))
    for pattern in (b"xyz", b"", b"ab" * 100 + b"x", b"b" + b"ab" * 699 + b"x"):
        check(module, offsets, chars, to_bytes(pattern))
    # A sliced column: offsets do not start at 0.
    offsets, chars = make_column(rng, 5000, 40, 0.05, 1000)
    check(module, offsets[100:].copy(), chars, to_bytes(b"cab"))


if __name__ == "__main__":
    test_string_contains()
    print("string_contains numba: passed")
