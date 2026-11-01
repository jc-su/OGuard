"""Test the Numba tile_scan kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1024, 1025, 1000003)
TILE = 1024


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_tile_scan_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    tiles = (n + TILE - 1) // TILE
    padded = np.zeros(tiles * TILE, dtype=np.int64)
    padded[:n] = values
    grid = padded.reshape(tiles, TILE)
    inclusive = np.cumsum(grid, axis=1)
    exclusive = (inclusive - grid).reshape(-1)[:n].astype(np.int32)
    totals = grid.sum(axis=1).astype(np.int32)
    return exclusive, totals


def check(task: ModuleType, n: int, rng: np.random.Generator) -> bool:
    values = rng.integers(-1000, 1001, size=n).astype(np.int32)
    expected_out, expected_totals = reference(values)
    tiles = (n + TILE - 1) // TILE

    d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
    d_totals = cuda.to_device(np.full(tiles, -1, dtype=np.int32))
    task.run(cuda.to_device(values), d_out, d_totals, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    totals = d_totals.copy_to_host()
    ok = out.tobytes() == expected_out.tobytes() and totals.tobytes() == expected_totals.tobytes()
    print(f"n={n}: {'PASS' if ok else 'FAIL'} ({tiles} tiles)")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(8)
    failed = [n for n in SIZES if not check(task, n, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
