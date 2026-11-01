"""Test the Numba string_equal kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
MAX_LENGTH = 16


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("string_equal_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def offsets_of(lengths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(lengths.shape[0] + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def random_chars(count: int, rng: np.random.Generator) -> np.ndarray:
    # A three-letter alphabet makes short random strings collide often.
    return rng.integers(ord("a"), ord("d"), size=count, dtype=np.uint8)


def make_columns(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build column a at random and column b from a, row by row.

    Row kinds of b: 0 is a copy of a, 1 changes one byte of a, 2 appends or
    removes the last byte of a, and 3 is a random string.
    """
    a_len = rng.integers(0, MAX_LENGTH + 1, size=n)
    a_offsets = offsets_of(a_len)
    a_chars = random_chars(int(a_offsets[-1]), rng)

    kind = rng.integers(0, 4, size=n)
    grow = (rng.integers(0, 2, size=n) == 1) | (a_len == 0)
    b_len = a_len.copy()
    resized = kind == 2
    b_len[resized] = np.where(grow, a_len + 1, a_len - 1)[resized]
    replaced = kind == 3
    b_len[replaced] = rng.integers(0, MAX_LENGTH + 1, size=int(replaced.sum()))
    b_offsets = offsets_of(b_len)
    b_chars = random_chars(int(b_offsets[-1]), rng)

    row = np.repeat(np.arange(n), b_len)
    pos = np.arange(b_chars.shape[0]) - b_offsets[row]
    copied = (kind[row] != 3) & (pos < a_len[row])
    b_chars[copied] = a_chars[a_offsets[row[copied]] + pos[copied]]

    changed = np.flatnonzero((kind == 1) & (a_len > 0))
    pos_changed = rng.integers(0, a_len[changed])
    b_chars[b_offsets[changed] + pos_changed] = a_chars[a_offsets[changed] + pos_changed] + 1
    return a_offsets, a_chars, b_offsets, b_chars


def reference(
    a_offsets: np.ndarray, a_chars: np.ndarray, b_offsets: np.ndarray, b_chars: np.ndarray
) -> np.ndarray:
    a_bytes = a_chars.tobytes()
    b_bytes = b_chars.tobytes()
    a_off = a_offsets.tolist()
    b_off = b_offsets.tolist()
    n = len(a_off) - 1
    return np.array(
        [a_bytes[a_off[i] : a_off[i + 1]] == b_bytes[b_off[i] : b_off[i + 1]] for i in range(n)],
        dtype=np.int8,
    )


def check(n: int, rng: np.random.Generator) -> None:
    a_offsets, a_chars, b_offsets, b_chars = make_columns(n, rng)
    match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    TASK.run(
        cuda.to_device(a_offsets),
        cuda.to_device(a_chars),
        cuda.to_device(b_offsets),
        cuda.to_device(b_chars),
        match,
        n,
    )
    cuda.synchronize()
    got = match.copy_to_host()
    expected = reference(a_offsets, a_chars, b_offsets, b_chars)
    assert got.tobytes() == expected.tobytes(), f"string_equal mismatch at n={n}"


def test_string_equal() -> None:
    rng = np.random.default_rng(3)
    for n in SIZES:
        check(n, rng)


if __name__ == "__main__":
    test_string_equal()
    print("string_equal numba: passed for n in", SIZES)
