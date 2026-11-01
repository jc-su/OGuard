"""GPU test for filter_select: Numba kernel against a NumPy reference, plus PTX export."""

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
    spec = importlib.util.spec_from_file_location("filter_select_numba", os.path.join(HERE, "numba.py"))
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
    quantity = rng.uniform(0.0, 100.0, n).astype(np.float64)
    threshold = 50.0
    expected = np.nonzero(quantity < threshold)[0].astype(np.int32)

    d_quantity = cuda.to_device(quantity)
    d_sel = cuda.device_array(max(n, 1), dtype=np.int32)
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    task.run(d_quantity, threshold, d_sel, d_count, n)
    cuda.synchronize()

    count = int(d_count.copy_to_host()[0])
    assert count == expected.size, f"n={n}: count {count} != {expected.size}"
    sel = np.sort(d_sel.copy_to_host()[:count])
    assert np.array_equal(sel, expected), f"n={n}: sel mismatch"

    if n == max(SIZES):
        us = bench(
            lambda: task.run(d_quantity, threshold, d_sel, d_count, n),
            lambda: d_count.copy_to_device(np.zeros(1, dtype=np.int32)),
        )
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
    print("PASS filter_select")


if __name__ == "__main__":
    main()
