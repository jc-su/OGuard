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
END_FILL = -7
VALUE_FILL = 123456789


def reference(values, n):
    run_ends = np.full(n, END_FILL, dtype=np.int32)
    run_values = np.full(n, VALUE_FILL, dtype=np.int64)
    if n == 0:
        return run_ends, run_values, 0
    starts = np.flatnonzero(np.concatenate([[True], values[1:] != values[:-1]]))
    r = len(starts)
    run_ends[:r] = np.append(starts[1:], n)
    run_values[:r] = values[starts]
    return run_ends, run_values, r


def make_case(rng, n, kind):
    if kind == "small_alphabet":
        return rng.integers(0, 3, n, dtype=np.int64)
    if kind == "distinct":
        return np.arange(n, dtype=np.int64)
    if kind == "all_equal":
        return np.full(n, I64_MIN, dtype=np.int64)
    if kind == "long_runs":
        lengths = rng.integers(1, 5000, n // 100 + 2)
        return np.repeat(rng.integers(I64_MIN, I64_MAX, len(lengths), dtype=np.int64, endpoint=True), lengths)[:n]
    if kind == "alternating":
        return np.where(np.arange(n) % 2 == 0, I64_MAX, I64_MIN).astype(np.int64)
    if kind == "random":
        return rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
    raise ValueError(kind)


KINDS = ("small_alphabet", "distinct", "all_equal", "long_runs", "alternating", "random")
SIZES = (0, 1, 2, 255, 2048, 2049, 1000, 1000003)


def main():
    rng = np.random.default_rng(11)
    failed = 0
    for n in SIZES:
        for kind in KINDS:
            values = make_case(rng, n, kind)
            d_ends = cuda.to_device(np.full(n, END_FILL, dtype=np.int32))
            d_vals = cuda.to_device(np.full(n, VALUE_FILL, dtype=np.int64))
            d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
            task.run(cuda.to_device(values), d_ends, d_vals, d_count, n)
            cuda.synchronize()
            want_ends, want_vals, want_count = reference(values, n)
            ok = (
                np.array_equal(d_ends.copy_to_host(), want_ends)
                and np.array_equal(d_vals.copy_to_host(), want_vals)
                and int(d_count.copy_to_host()[0]) == want_count
            )
            failed += not ok
            print(("PASS" if ok else "FAIL"), "run_end_encode numba", n, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
