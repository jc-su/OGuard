"""GPU test of the Numba run_end_decode kernel against a NumPy reference."""

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
SHAPES = ("single", "unit", "short", "long")


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_runs(n, shape, rng):
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if shape == "single":
        return np.array([n], dtype=np.int64)
    if shape == "unit":
        return np.ones(n, dtype=np.int64)
    high = 8 if shape == "short" else 3000
    lengths = rng.integers(1, high, size=n)
    ends = np.cumsum(lengths)
    lengths = lengths[: np.searchsorted(ends, n, side="left") + 1]
    lengths[-1] = n - (lengths[:-1].sum())
    assert lengths.sum() == n and (lengths > 0).all()
    return lengths


def check(task, n, shape, rng):
    lengths = make_runs(n, shape, rng)
    r = lengths.shape[0]
    run_ends = np.cumsum(lengths).astype(np.int32)
    run_values = rng.standard_normal(r) * 1e3
    expected = np.repeat(run_values, lengths)

    d_values = cuda.to_device(run_values)
    d_ends = cuda.to_device(run_ends)
    d_out = cuda.to_device(np.full(n, np.nan))
    task.run(d_values, d_ends, d_out, r, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    assert np.array_equal(out, expected), f"n={n} shape={shape} r={r}: output differs"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(3)
    for n in SIZES:
        for shape in SHAPES:
            check(task, n, shape, rng)
        print(f"run_end_decode numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
