"""Test the Numba gather_nullable kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
# (m, n): m output rows gathered from n rows of col.
CASES = (
    (0, 0),
    (0, 1000),
    (1, 0),
    (1, 1),
    (1000, 1),
    (1000, 1000),
    (1000003, 1000003),
    (1000003, 1000),
)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "heldout2_gather_nullable_numba", HERE / "numba.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task: ModuleType, m: int, n: int, rng: np.random.Generator) -> bool:
    col = rng.standard_normal(n)
    if n == 0:
        idx = np.full(m, -1, dtype=np.int32)
    else:
        idx = rng.integers(0, n, size=m).astype(np.int32)
        idx[rng.random(m) < 0.3] = -1
    expected = np.where(idx < 0, np.nan, col[np.maximum(idx, 0)] if n > 0 else np.nan)

    d_out = cuda.to_device(np.full(m, -1.0))
    task.run(cuda.to_device(col), cuda.to_device(idx), d_out, m, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    nan_ok = np.array_equal(np.isnan(out), np.isnan(expected))
    value_ok = np.array_equal(out[~np.isnan(expected)], expected[~np.isnan(expected)])
    ok = nan_ok and value_ok
    print(f"m={m} n={n}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(2)
    failed = [case for case in CASES if not check(task, *case, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
