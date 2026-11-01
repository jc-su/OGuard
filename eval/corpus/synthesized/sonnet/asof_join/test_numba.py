import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory must not shadow the numba package.
sys.path = [p for p in sys.path if p not in ("", HERE)]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


LIM = 2**61


def reference(left, right, n, r, tolerance):
    j = np.searchsorted(right, left, side="right").astype(np.int64) - 1
    if r == 0:
        return np.full(n, -1, dtype=np.int32)
    near = right[np.clip(j, 0, r - 1)]
    found = (j >= 0) & ((left - near) <= tolerance)
    return np.where(found, j, -1).astype(np.int32)


def make_case(rng, n, r, kind):
    if kind == "dense":
        right = np.sort(rng.integers(-LIM, LIM, r, dtype=np.int64, endpoint=True))
        left = rng.integers(-LIM, LIM, n, dtype=np.int64, endpoint=True)
        tolerance = 2**60
    elif kind == "near":
        right = np.sort(rng.integers(-1000, 1000, r, dtype=np.int64))
        left = rng.integers(-1100, 1100, n, dtype=np.int64)
        tolerance = 5
    elif kind == "zero_tolerance":
        right = np.sort(rng.integers(-100, 100, r, dtype=np.int64))
        left = rng.integers(-100, 100, n, dtype=np.int64)
        tolerance = 0
    elif kind == "max_tolerance":
        right = np.sort(rng.integers(-LIM, LIM, r, dtype=np.int64, endpoint=True))
        left = rng.integers(-LIM, LIM, n, dtype=np.int64, endpoint=True)
        tolerance = 2**62
    elif kind == "extremes":
        right = np.sort(np.concatenate([[-LIM, LIM], rng.integers(-LIM, LIM, max(r - 2, 0), dtype=np.int64)]))[:r]
        left = np.concatenate([[-LIM, LIM, 0], rng.integers(-LIM, LIM, max(n - 3, 0), dtype=np.int64)])[:n]
        tolerance = 2**62
    elif kind == "all_equal":
        right = np.full(r, 7, dtype=np.int64)
        left = np.full(n, 7, dtype=np.int64)
        tolerance = 0
    else:
        raise ValueError(kind)
    return left.astype(np.int64), right.astype(np.int64), tolerance


def main():
    rng = np.random.default_rng(4)
    failed = 0
    for n in (0, 1, 1000, 1000003):
        for r in (0, 1, 1000, 1000003):
            for kind in ("dense", "near", "zero_tolerance", "max_tolerance", "extremes", "all_equal"):
                left, right, tolerance = make_case(rng, n, r, kind)
                d_out = cuda.to_device(np.full(n, -5, dtype=np.int32))
                task.run(cuda.to_device(left), cuda.to_device(right), d_out, n, r, tolerance)
                cuda.synchronize()
                ok = np.array_equal(d_out.copy_to_host(), reference(left, right, n, r, tolerance))
                failed += not ok
                print(("PASS" if ok else "FAIL"), "asof_join numba", n, r, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
