"""Test the Numba lag kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
LAGS = (1, 3, 1024)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_lag_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_first(n: int, rng: np.random.Generator) -> np.ndarray:
    """Partition starts: mostly partitions of 1 to 40 rows, with a few of up to
    3000 rows so that k = 1024 has in-partition lags."""
    starts = [0]
    while starts[-1] < n:
        size = int(rng.integers(1, 41)) if rng.random() < 0.95 else int(rng.integers(1000, 3001))
        starts.append(starts[-1] + size)
    starts_arr = np.array(starts[:-1], dtype=np.int64)
    partition = np.searchsorted(starts_arr, np.arange(n), side="right") - 1
    return starts_arr[partition].astype(np.int32)


def check(task: ModuleType, n: int, k: int, rng: np.random.Generator) -> bool:
    values = rng.standard_normal(n)
    first = make_first(n, rng)
    j = np.arange(n) - k
    inside = j >= first
    expected = np.where(inside, values[np.where(inside, j, 0)], np.nan)

    d_out = cuda.to_device(np.full(n, -1.0))
    task.run(cuda.to_device(values), cuda.to_device(first), d_out, n, k)
    cuda.synchronize()
    out = d_out.copy_to_host()
    nan_ok = np.array_equal(np.isnan(out), np.isnan(expected))
    value_ok = np.array_equal(out[inside], expected[inside])
    ok = nan_ok and value_ok
    print(f"n={n} k={k}: {'PASS' if ok else 'FAIL'} ({int(inside.sum())} lagged rows)")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(5)
    failed = [(n, k) for n in SIZES for k in LAGS if not check(task, n, k, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
