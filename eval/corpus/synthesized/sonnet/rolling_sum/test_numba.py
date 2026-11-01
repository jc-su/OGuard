"""Test for the Numba rolling_sum kernel against a NumPy reference, for
n in {0, 1, 1000, 1000003} and a few window sizes (including window > n).

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
        "rolling_sum_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(values, window):
    n = values.size
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    return prefix[idx + 1] - prefix[lo]


def run_case(mod, n, window, rng):
    values = rng.uniform(-100.0, 100.0, size=max(n, 1))
    expected = reference(values[:n], window) if n > 0 else np.zeros(0)

    d_values = cuda.to_device(values)
    d_out = cuda.to_device(np.zeros(max(n, 1), dtype=np.float64))

    mod.run(d_values, d_out, n, window)
    cuda.synchronize()

    out = d_out.copy_to_host()[:n]
    ok = np.allclose(out, expected, rtol=1e-9, atol=1e-6)
    print(f"n={n} window={window}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(5)
    cases = [
        (0, 5),
        (1, 1),
        (1, 1024),
        (1000, 7),
        (1000, 1024),
        (1000003, 1),
        (1000003, 300),
        (1000003, 1024),
    ]
    ok = all(run_case(mod, n, w, rng) for n, w in cases)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
