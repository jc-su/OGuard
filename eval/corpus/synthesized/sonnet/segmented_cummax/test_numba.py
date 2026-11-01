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


I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


def reference(values, first, n):
    out = np.empty(n, dtype=np.int64)
    starts = np.flatnonzero(first == np.arange(n))
    ends = np.append(starts[1:], n)
    for a, b in zip(starts, ends):
        out[a:b] = np.maximum.accumulate(values[a:b])
    return out


def make_case(rng, n, kind):
    if kind == "single":
        first = np.zeros(n, dtype=np.int32)
        values = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
    elif kind == "single_increasing":
        first = np.zeros(n, dtype=np.int32)
        values = np.arange(n, dtype=np.int64) - n // 2
    elif kind == "single_decreasing":
        first = np.zeros(n, dtype=np.int32)
        values = (n - np.arange(n, dtype=np.int64))
    elif kind == "each_row":
        first = np.arange(n, dtype=np.int32)
        values = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
    elif kind == "short":
        starts = rng.random(n) < 0.1
        if n > 0:
            starts[0] = True
        first = np.maximum.accumulate(np.where(starts, np.arange(n), 0)).astype(np.int32)
        values = rng.integers(-100, 100, n, dtype=np.int64)
    elif kind == "long":
        starts = rng.random(n) < 0.0003
        if n > 0:
            starts[0] = True
        first = np.maximum.accumulate(np.where(starts, np.arange(n), 0)).astype(np.int32)
        values = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
    elif kind == "min_values":
        first = np.zeros(n, dtype=np.int32)
        values = np.full(n, I64_MIN, dtype=np.int64)
    else:
        raise ValueError(kind)
    return values, first


KINDS = ("single", "single_increasing", "single_decreasing", "each_row", "short", "long", "min_values")
SIZES = (0, 1, 2, 255, 2048, 2049, 1000, 1000003)


def main():
    rng = np.random.default_rng(12)
    failed = 0
    for n in SIZES:
        for kind in KINDS:
            values, first = make_case(rng, n, kind)
            d_out = cuda.to_device(np.full(n, 5, dtype=np.int64))
            task.run(cuda.to_device(values), cuda.to_device(first), d_out, n)
            cuda.synchronize()
            ok = np.array_equal(d_out.copy_to_host(), reference(values, first, n))
            failed += not ok
            print(("PASS" if ok else "FAIL"), "segmented_cummax numba", n, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
