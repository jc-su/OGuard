"""Test the Numba coalesce kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_coalesce_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task: ModuleType, n: int, rng: np.random.Generator) -> bool:
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    valid_rows = rng.random(n) < 0.6
    valid = np.packbits(valid_rows, bitorder="little")
    assert valid.size == (n + 7) // 8
    expected = np.where(valid_rows, a, b)

    d_out = cuda.to_device(np.full(n, -1.0))
    task.run(cuda.to_device(a), cuda.to_device(valid), cuda.to_device(b), d_out, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    ok = out.tobytes() == expected.tobytes()
    print(f"n={n}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(1)
    failed = [n for n in SIZES if not check(task, n, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
