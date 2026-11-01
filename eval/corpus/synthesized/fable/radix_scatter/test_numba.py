"""GPU test of the Numba radix_scatter kernel against a NumPy reference."""

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
SHIFTS = (0, 8, 16, 24)
DISTRIBUTIONS = ("uniform", "skewed", "sorted")


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_keys(n, distribution, rng):
    if distribution == "uniform":
        return rng.integers(0, 2**32, size=n, dtype=np.uint32)
    if distribution == "skewed":
        # Few distinct values: most bytes fall into one or two bins.
        return rng.choice(np.array([0, 1, 0x01020304, 0xFFFFFFFF], dtype=np.uint32), size=n)
    return np.sort(rng.integers(0, 2**32, size=n, dtype=np.uint32))


def check(task, n, shift, distribution, rng):
    keys = make_keys(n, distribution, rng)
    byte = ((keys >> np.uint32(shift)) & np.uint32(255)).astype(np.int64)
    hist = np.bincount(byte, minlength=256)
    cursors_in = (np.cumsum(hist) - hist).astype(np.int32)

    d_keys = cuda.to_device(keys)
    d_cursors = cuda.to_device(cursors_in.copy())
    d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
    task.run(d_keys, d_cursors, d_out, n, shift)
    cuda.synchronize()
    out = d_out.copy_to_host()
    cursors_out = d_cursors.copy_to_host()

    tag = f"n={n} shift={shift} {distribution}"
    assert np.array_equal(np.sort(out), np.arange(n, dtype=np.int32)), f"{tag}: out is not a permutation"
    assert np.all(np.diff(byte[out]) >= 0), f"{tag}: out is not grouped by byte in bin order"
    assert np.array_equal(cursors_out, cursors_in + hist), f"{tag}: cursors did not advance to the bin ends"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(7)
    for n in SIZES:
        for distribution in DISTRIBUTIONS:
            for shift in SHIFTS:
                check(task, n, shift, distribution, rng)
        print(f"radix_scatter numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
