"""GPU test of the Numba merge_sorted kernel against a NumPy reference."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory shadows the numba package while the script's
# directory is on sys.path, so drop it before importing numba.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
RANGES = (50, 2**62)  # many duplicates; nearly distinct


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shapes(n):
    return ((n, n), (n, n // 3), (n // 3, n), (n, 0), (0, n))


def check(task, m, p, value_range, rng):
    a = np.sort(rng.integers(-value_range, value_range, size=m, dtype=np.int64))
    b = np.sort(rng.integers(-value_range, value_range, size=p, dtype=np.int64))
    expected = np.sort(np.concatenate([a, b]), kind="stable")

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.to_device(np.full(m + p, np.iinfo(np.int64).min, dtype=np.int64))
    task.run(d_a, d_b, d_out, m, p)
    cuda.synchronize()
    out = d_out.copy_to_host()
    assert np.array_equal(out, expected), f"m={m} p={p} range={value_range}: merge differs at {np.flatnonzero(out != expected)[:5]}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(9)
    for n in SIZES:
        for m, p in shapes(n):
            for value_range in RANGES:
                check(task, m, p, value_range, rng)
        print(f"merge_sorted numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
