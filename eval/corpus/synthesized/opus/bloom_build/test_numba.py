"""Test the Numba bloom_build kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with words in WORDS, the smallest, a middle, and the largest
filter. Keys cover the whole int64 range, and a quarter of them repeat other
keys.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
WORDS = (1, 1024, 1 << 26)
MULTIPLIER_1 = np.uint64(0x9E3779B97F4A7C15)
MULTIPLIER_2 = np.uint64(0xC2B2AE3D27D4EB4F)


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("bloom_build_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_keys(n: int, rng: np.random.Generator) -> np.ndarray:
    keys = rng.integers(
        np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=n, dtype=np.int64, endpoint=True
    )
    repeated = rng.random(n) < 0.25
    keys[repeated] = keys[rng.integers(0, n, size=int(repeated.sum()))]
    return keys


def reference(keys: np.ndarray, words: int) -> np.ndarray:
    mask = np.uint64(32 * words - 1)
    unsigned = keys.view(np.uint64)
    bits = np.zeros(words, dtype=np.uint32)
    for multiplier in (MULTIPLIER_1, MULTIPLIER_2):
        position = ((unsigned * multiplier) >> np.uint64(32)) & mask
        word = (position >> np.uint64(5)).astype(np.int64)
        bit = np.left_shift(np.uint32(1), (position & np.uint64(31)).astype(np.uint32))
        np.bitwise_or.at(bits, word, bit)
    return bits


def check(n: int, words: int, rng: np.random.Generator) -> None:
    keys = make_keys(n, rng)
    bits = cuda.to_device(np.zeros(words, dtype=np.uint32))
    TASK.run(cuda.to_device(keys), bits, n, words)
    cuda.synchronize()
    got = bits.copy_to_host()
    expected = reference(keys, words)
    assert got.tobytes() == expected.tobytes(), f"bloom_build mismatch at n={n} words={words}"


def test_bloom_build() -> None:
    rng = np.random.default_rng(6)
    for n in SIZES:
        for words in WORDS:
            check(n, words, rng)


if __name__ == "__main__":
    test_bloom_build()
    print("bloom_build numba: passed for n in", SIZES, "and words in", WORDS)
