"""GPU test of the Numba min_max kernel against a NumPy reference."""

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
SHIFTS = (0.0, 1e7, -1e7)  # mixed signs, all positive, all negative


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, n, shift, rng):
    values = rng.standard_normal(n) * 1e6 + shift
    if n == 0:
        expected = np.array([np.inf, -np.inf])
    else:
        expected = np.array([values.min(), values.max()])

    d_values = cuda.to_device(values)
    d_result = cuda.to_device(np.array([np.inf, -np.inf]))
    task.run(d_values, d_result, n)
    cuda.synchronize()
    result = d_result.copy_to_host()
    assert np.array_equal(result, expected), f"n={n} shift={shift}: got {result}, expected {expected}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(5)
    for n in SIZES:
        for shift in SHIFTS:
            check(task, n, shift, rng)
        print(f"min_max numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
