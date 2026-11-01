"""GPU test of the Numba valid_rows kernel against a NumPy reference."""

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
DENSITIES = (0.0, 0.5, 0.97, 1.0)


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, n, density, rng):
    nbytes = (n + 7) // 8
    bits = rng.random(nbytes * 8) < density
    valid = np.packbits(bits, bitorder="little").astype(np.uint8)
    assert valid.shape[0] == nbytes
    expected = np.flatnonzero(bits[:n]).astype(np.int32)

    d_valid = cuda.to_device(valid)
    d_rows = cuda.to_device(np.full(n, -1, dtype=np.int32))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    task.run(d_valid, d_rows, d_count, n)
    cuda.synchronize()
    count = int(d_count.copy_to_host()[0])
    rows = d_rows.copy_to_host()

    assert count == expected.shape[0], f"n={n} density={density}: count {count} != {expected.shape[0]}"
    got = np.sort(rows[:count])
    assert np.array_equal(got, expected), f"n={n} density={density}: row ids differ"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(1)
    for n in SIZES:
        for density in DENSITIES:
            check(task, n, density, rng)
        print(f"valid_rows numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
