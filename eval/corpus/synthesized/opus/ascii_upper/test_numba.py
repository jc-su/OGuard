"""Test the Numba ascii_upper kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with offsets[0] = 0 and with offsets[0] = 5, the second as for
a sliced Arrow column. out starts as the complement of the expected bytes, so
the test also fails when the kernel skips a byte of the column or writes a
byte before offsets[0].
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
BASES = (0, 5)
MAX_LENGTH = 16


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("ascii_upper_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_column(n: int, base: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lengths = rng.integers(0, MAX_LENGTH + 1, size=n)
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = base
    np.cumsum(lengths, out=offsets[1:])
    offsets[1:] += base
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(offsets: np.ndarray, chars: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Return out as the kernel must leave it when out held `before`."""
    begin = int(offsets[0])
    end = int(offsets[-1])
    column = chars[begin:end]
    lower = (column >= ord("a")) & (column <= ord("z"))
    expected = before.copy()
    expected[begin:end] = np.where(lower, column - 32, column)
    return expected


def check(n: int, base: int, rng: np.random.Generator) -> None:
    offsets, chars = make_column(n, base, rng)
    upper = reference(offsets, chars, chars)
    before = ~upper
    out = cuda.to_device(before)
    TASK.run(cuda.to_device(offsets), cuda.to_device(chars), out, n)
    cuda.synchronize()
    got = out.copy_to_host()
    expected = reference(offsets, chars, before)
    assert got.tobytes() == expected.tobytes(), f"ascii_upper mismatch at n={n} offsets[0]={base}"


def test_ascii_upper() -> None:
    rng = np.random.default_rng(4)
    for n in SIZES:
        for base in BASES:
            check(n, base, rng)


if __name__ == "__main__":
    test_ascii_upper()
    print("ascii_upper numba: passed for n in", SIZES, "and offsets[0] in", BASES)
