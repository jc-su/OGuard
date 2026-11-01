"""Test the Numba block_sort kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with keys over the whole int32 range and with keys from a
small range plus INT32_MIN and INT32_MAX, which gives many equal keys and
keys equal to the padding of a short last tile. out starts as the complement
of the expected keys, so the test also fails when the kernel skips a row.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
TILE = 1024
INT32 = np.iinfo(np.int32)


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("block_sort_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_keys(n: int, narrow: bool, rng: np.random.Generator) -> np.ndarray:
    if not narrow:
        return rng.integers(INT32.min, INT32.max, size=n, dtype=np.int32, endpoint=True)
    choices = np.array([INT32.min, -2, -1, 0, 1, 2, INT32.max], dtype=np.int32)
    return rng.choice(choices, size=n)


def reference(keys: np.ndarray) -> np.ndarray:
    out = keys.copy()
    for start in range(0, keys.shape[0], TILE):
        out[start : start + TILE].sort()
    return out


def check(n: int, narrow: bool, rng: np.random.Generator) -> None:
    keys = make_keys(n, narrow, rng)
    expected = reference(keys)
    out = cuda.to_device(~expected)
    TASK.run(cuda.to_device(keys), out, n)
    cuda.synchronize()
    got = out.copy_to_host()
    assert got.tobytes() == expected.tobytes(), f"block_sort mismatch at n={n} narrow={narrow}"


def test_block_sort() -> None:
    rng = np.random.default_rng(9)
    for n in SIZES:
        for narrow in (False, True):
            check(n, narrow, rng)


if __name__ == "__main__":
    test_block_sort()
    print("block_sort numba: passed for n in", SIZES)
