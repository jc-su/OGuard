"""Test for the Numba merge_sorted kernel against a NumPy reference, for
total sizes m+p in {0, 1, 1000, 1000003}, split a few different ways,
including empty a or b and a low-cardinality case.

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
        "merge_sorted_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, m, p, value_range, rng):
    a = np.sort(rng.integers(-value_range, value_range + 1, size=max(m, 1)))[:m]
    b = np.sort(rng.integers(-value_range, value_range + 1, size=max(p, 1)))[:p]
    total = m + p
    expected = np.sort(np.concatenate([a, b])) if total > 0 else np.zeros(0, dtype=np.int64)

    d_a = cuda.to_device(a.astype(np.int64) if m > 0 else np.zeros(1, dtype=np.int64))
    d_b = cuda.to_device(b.astype(np.int64) if p > 0 else np.zeros(1, dtype=np.int64))
    d_out = cuda.to_device(np.zeros(max(total, 1), dtype=np.int64))

    mod.run(d_a, d_b, d_out, m, p)
    cuda.synchronize()

    out = d_out.copy_to_host()[:total]
    ok = np.array_equal(out, expected)
    print(f"m={m} p={p} range={value_range}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(8)
    cases = [
        (0, 0, 1000000),
        (1, 0, 1000000),
        (0, 1, 1000000),
        (400, 600, 1000000),
        (500000, 500003, 1000000),
        (500000, 500003, 5),  # low-cardinality: lots of ties
        (1000003, 0, 1000000),
        (0, 1000003, 1000000),
    ]
    ok = all(run_case(mod, m, p, r, rng) for m, p, r in cases)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
