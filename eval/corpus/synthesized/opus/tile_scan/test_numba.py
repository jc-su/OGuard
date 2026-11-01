"""Test the Numba tile_scan kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Values span the whole int32 range, so the sums wrap around, and the reference
wraps them the same way in int32 arithmetic. out and totals start as the
complement of the expected values, so the test also fails when the kernel
skips a row or a tile.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
TILE = 1024


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("tile_scan_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    tiles = (n + TILE - 1) // TILE
    padded = np.zeros(tiles * TILE, dtype=np.int32)
    padded[:n] = values
    rows = padded.reshape(tiles, TILE)
    inclusive = np.cumsum(rows, axis=1, dtype=np.int32)
    exclusive = inclusive - rows
    return exclusive.reshape(-1)[:n].copy(), inclusive[:, TILE - 1].copy()


def check(n: int, rng: np.random.Generator) -> None:
    values = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=n, dtype=np.int32, endpoint=True
    )
    expected_out, expected_totals = reference(values)
    out = cuda.to_device(~expected_out)
    totals = cuda.to_device(~expected_totals)
    TASK.run(cuda.to_device(values), out, totals, n)
    cuda.synchronize()
    got_out = out.copy_to_host()
    got_totals = totals.copy_to_host()
    assert got_out.tobytes() == expected_out.tobytes(), f"tile_scan out mismatch at n={n}"
    assert got_totals.tobytes() == expected_totals.tobytes(), f"tile_scan totals mismatch at n={n}"


def test_tile_scan() -> None:
    rng = np.random.default_rng(8)
    for n in SIZES:
        check(n, rng)


if __name__ == "__main__":
    test_tile_scan()
    print("tile_scan numba: passed for n in", SIZES)
