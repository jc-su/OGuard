"""GPU test of the Numba rolling_sum kernel against a NumPy reference."""

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
WINDOWS = (1, 2, 7, 256, 1000, 1024)


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(values, window):
    # Prefix sums in 80-bit extended precision, then differences.
    cs = np.cumsum(values, dtype=np.longdouble)
    ref = cs.copy()
    if window < values.shape[0]:
        ref[window:] -= cs[:-window]
    return ref.astype(np.float64)


def check(task, n, window, d_values, values):
    expected = reference(values, window)
    d_out = cuda.to_device(np.full(n, np.nan))
    task.run(d_values, d_out, n, window)
    cuda.synchronize()
    out = d_out.copy_to_host()
    ok = np.allclose(out, expected, rtol=1e-12, atol=1e-9)
    assert ok, f"n={n} window={window}: max abs diff {np.max(np.abs(out - expected))}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(6)
    for n in SIZES:
        values = rng.uniform(-1.0, 1.0, size=n)
        d_values = cuda.to_device(values)
        for window in WINDOWS:
            check(task, n, window, d_values, values)
        print(f"rolling_sum numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
