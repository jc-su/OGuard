"""GPU test for gather: Numba kernel against a NumPy reference, plus PTX export."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory shadows the numba package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

SIZES = (0, 1, 1000, 1000003)


def load_task():
    spec = importlib.util.spec_from_file_location("gather_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(launch, reset, reps=20):
    """Average kernel time in microseconds; reset runs before every timed launch."""
    start, end = cuda.event(), cuda.event()
    total = 0.0
    for _ in range(reps):
        reset()
        start.record()
        launch()
        end.record()
        end.synchronize()
        total += cuda.event_elapsed_time(start, end)
    return total / reps * 1e3


def check(task, n, rng):
    m = n
    col = rng.uniform(0.0, 1.0, n).astype(np.float64)
    sel = rng.integers(0, max(n, 1), m).astype(np.int32)
    expected = col[sel]

    d_col = cuda.to_device(col)
    d_sel = cuda.to_device(sel)
    d_out = cuda.device_array(max(m, 1), dtype=np.float64)
    task.run(d_col, d_sel, d_out, m, n)
    cuda.synchronize()

    out = d_out.copy_to_host()[:m]
    assert np.array_equal(out, expected), f"n={n}: out mismatch"

    if n == max(SIZES):
        us = bench(lambda: task.run(d_col, d_sel, d_out, m, n), lambda: None)
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    ptx, _ = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("PASS gather")


if __name__ == "__main__":
    main()
