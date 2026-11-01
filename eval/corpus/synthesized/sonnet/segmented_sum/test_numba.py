"""Test and compile segmented_sum's numba.py to PTX.

Runs the kernel on the GPU for n (total values, split across g
segments of varied, including zero, length) in {0, 1, 1000, 1000003}
and checks the result against a NumPy reference. Also writes
numba.ptx.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "segmented_sum_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_offsets(n, g, rng):
    if n == 0:
        sizes = np.zeros(g, dtype=np.int64)
    else:
        sizes = rng.multinomial(n, np.full(g, 1.0 / g))
    offsets = np.zeros(g + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes)
    return offsets


def check(task, n, rng):
    g = max(1, min(n, 5000)) if n > 0 else 5
    offsets = make_offsets(n, g, rng)
    values = rng.uniform(-5.0, 5.0, size=max(n, 1)).astype(np.float64)

    ref = np.zeros(g, dtype=np.float64)
    for j in range(g):
        ref[j] = values[offsets[j] : offsets[j + 1]].sum() if n > 0 else 0.0

    d_offsets = cuda.to_device(offsets)
    d_values = cuda.to_device(values)
    d_sums = cuda.device_array(g, dtype=np.float64)

    task.run(d_offsets, d_values, d_sums, g)
    cuda.synchronize()

    got = d_sums.copy_to_host()
    assert np.allclose(got, ref, atol=1e-8), f"n={n}: mismatch"
    print(f"n={n}: PASS (g={g})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)

    ptx, resty = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("wrote numba.ptx")


if __name__ == "__main__":
    main()
