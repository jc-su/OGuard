"""Test the Numba bloom_probe kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with words in WORDS. The filter holds n random keys, set as
bloom_build sets them. Half of the probe keys come from the filter's keys and
half are random, so both hits and misses occur.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
WORDS = (1, 1024, 1 << 26)
MULTIPLIERS = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F))


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("bloom_probe_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def random_keys(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=n, dtype=np.int64, endpoint=True
    )


def positions(keys: np.ndarray, words: int, multiplier: np.uint64) -> np.ndarray:
    mask = np.uint64(32 * words - 1)
    return ((keys.view(np.uint64) * multiplier) >> np.uint64(32)) & mask


def build(keys: np.ndarray, words: int) -> np.ndarray:
    bits = np.zeros(words, dtype=np.uint32)
    for multiplier in MULTIPLIERS:
        p = positions(keys, words, multiplier)
        bit = np.left_shift(np.uint32(1), (p & np.uint64(31)).astype(np.uint32))
        np.bitwise_or.at(bits, (p >> np.uint64(5)).astype(np.int64), bit)
    return bits


def reference(keys: np.ndarray, bits: np.ndarray) -> np.ndarray:
    words = bits.shape[0]
    hit = np.ones(keys.shape[0], dtype=np.uint32)
    for multiplier in MULTIPLIERS:
        p = positions(keys, words, multiplier)
        word = bits[(p >> np.uint64(5)).astype(np.int64)]
        hit &= (word >> (p & np.uint64(31)).astype(np.uint32)) & np.uint32(1)
    return hit.astype(np.int8)


def check(n: int, words: int, rng: np.random.Generator) -> None:
    stored = random_keys(n, rng)
    bits = build(stored, words)
    keys = random_keys(n, rng)
    if n > 0:
        present = rng.random(n) < 0.5
        keys[present] = stored[rng.integers(0, n, size=int(present.sum()))]
    match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    TASK.run(cuda.to_device(keys), cuda.to_device(bits), match, n, words)
    cuda.synchronize()
    got = match.copy_to_host()
    expected = reference(keys, bits)
    assert got.tobytes() == expected.tobytes(), f"bloom_probe mismatch at n={n} words={words}"


def test_bloom_probe() -> None:
    rng = np.random.default_rng(7)
    for n in SIZES:
        for words in WORDS:
            check(n, words, rng)


if __name__ == "__main__":
    test_bloom_probe()
    print("bloom_probe numba: passed for n in", SIZES, "and words in", WORDS)
