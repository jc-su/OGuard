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


def reference(group, value, valid, n, g):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    grp = group[bits]
    counts = np.bincount(grp, minlength=g).astype(np.int64)
    sums = np.zeros(g, dtype=np.uint64)
    np.add.at(sums, grp, value[bits].view(np.uint64))
    return counts, sums.view(np.int64)


def make_case(rng, n, g, kind):
    group = rng.integers(0, g, n, dtype=np.int32)
    if kind == "small":
        value = rng.integers(-1000, 1000, n, dtype=np.int64)
        valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
    elif kind == "wrap":
        value = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
        valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
    elif kind == "max":
        value = np.full(n, I64_MAX, dtype=np.int64)
        valid = np.full((n + 7) // 8, 255, dtype=np.uint8)
    elif kind == "all_null":
        value = rng.integers(-1000, 1000, n, dtype=np.int64)
        valid = np.zeros((n + 7) // 8, dtype=np.uint8)
    elif kind == "skew":
        group = np.where(rng.random(n) < 0.9, 0, group).astype(np.int32)
        value = rng.integers(-1000, 1000, n, dtype=np.int64)
        valid = np.full((n + 7) // 8, 255, dtype=np.uint8)
    else:
        raise ValueError(kind)
    return group, value, valid


def main():
    rng = np.random.default_rng(3)
    failed = 0
    for n in (0, 1, 1000, 1000003):
        for g in (1, 7, 256, 1024):
            for kind in ("small", "wrap", "max", "all_null", "skew"):
                group, value, valid = make_case(rng, n, g, kind)
                d_counts = cuda.to_device(np.zeros(g, dtype=np.int64))
                d_sums = cuda.to_device(np.zeros(g, dtype=np.int64))
                task.run(
                    cuda.to_device(group), cuda.to_device(value), cuda.to_device(valid),
                    d_counts, d_sums, n, g,
                )
                cuda.synchronize()
                want_counts, want_sums = reference(group, value, valid, n, g)
                ok = np.array_equal(d_counts.copy_to_host(), want_counts) and np.array_equal(
                    d_sums.copy_to_host(), want_sums
                )
                failed += not ok
                print(("PASS" if ok else "FAIL"), "group_sum_valid numba", n, g, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
