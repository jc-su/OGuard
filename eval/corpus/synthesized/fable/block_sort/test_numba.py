"""Test the Numba block_sort kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1024, 1025, 1000003)
TILE = 1024
INT32_MIN = -2147483648
INT32_MAX = 2147483647


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_block_sort_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_keys(n: int, rng: np.random.Generator) -> np.ndarray:
    """Full-range keys with 30% drawn from a small range for duplicates, and
    some INT32_MIN and INT32_MAX values."""
    keys = rng.integers(INT32_MIN, INT32_MAX + 1, size=n, dtype=np.int64)
    small = rng.random(n) < 0.3
    keys[small] = rng.integers(-5, 6, size=int(small.sum()))
    extreme = rng.random(n) < 0.01
    keys[extreme] = rng.choice(np.array([INT32_MIN, INT32_MAX]), size=int(extreme.sum()))
    return keys.astype(np.int32)


def reference(keys: np.ndarray) -> np.ndarray:
    n = keys.size
    tiles = (n + TILE - 1) // TILE
    padded = np.full(tiles * TILE, INT32_MAX, dtype=np.int32)
    padded[:n] = keys
    return np.sort(padded.reshape(tiles, TILE), axis=1).reshape(-1)[:n]


def check(task: ModuleType, n: int, rng: np.random.Generator) -> bool:
    keys = make_keys(n, rng)
    expected = reference(keys)

    d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
    task.run(cuda.to_device(keys), d_out, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    ok = out.tobytes() == expected.tobytes()
    print(f"n={n}: {'PASS' if ok else 'FAIL'} ({(n + TILE - 1) // TILE} tiles)")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(9)
    failed = [n for n in SIZES if not check(task, n, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
