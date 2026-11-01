"""Test for the Numba min_max kernel against a NumPy reference, for
n in {0, 1, 1000, 1000003}.

Run as: CUDA_VISIBLE_DEVICES=1 python -P test_numba.py
"""

import importlib.util
import pathlib
import sys

import numpy as np
from numba import cuda

_HERE = pathlib.Path(__file__).resolve().parent


def _load_task_module():
    spec = importlib.util.spec_from_file_location(
        "min_max_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n, rng):
    values = rng.uniform(-1e6, 1e6, size=max(n, 1))
    expected = (values[:n].min(), values[:n].max()) if n > 0 else (np.inf, -np.inf)

    d_values = cuda.to_device(values)
    d_result = cuda.to_device(np.array([np.inf, -np.inf], dtype=np.float64))

    mod.run(d_values, d_result, n)
    cuda.synchronize()

    result = d_result.copy_to_host()
    ok = result[0] == expected[0] and result[1] == expected[1]
    print(f"n={n}: min={result[0]!r} (ref {expected[0]!r}) "
          f"max={result[1]!r} (ref {expected[1]!r}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(4)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, rng) for n in sizes)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
