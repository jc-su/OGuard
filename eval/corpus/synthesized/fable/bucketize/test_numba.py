"""Test the Numba bucketize kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
BOUND_COUNTS = (0, 1, 7, 1000)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_bucketize_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(n: int, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Integer-valued bounds with duplicates, and values that are a mix of
    exact bounds, values between bounds, and values outside the bounds."""
    bounds = np.sort(np.round(rng.standard_normal(b) * 30.0))
    values = rng.standard_normal(n) * 40.0
    on_bound = rng.random(n) < 0.3
    values[on_bound] = np.round(values[on_bound])
    return values, bounds


def check(task: ModuleType, n: int, b: int, rng: np.random.Generator) -> bool:
    values, bounds = make_inputs(n, b, rng)
    expected = np.searchsorted(bounds, values, side="right").astype(np.int32)

    d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
    task.run(cuda.to_device(values), cuda.to_device(bounds), d_out, n, b)
    cuda.synchronize()
    out = d_out.copy_to_host()
    ok = out.tobytes() == expected.tobytes()
    print(f"n={n} b={b}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(10)
    failed = [(n, b) for n in SIZES for b in BOUND_COUNTS if not check(task, n, b, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
