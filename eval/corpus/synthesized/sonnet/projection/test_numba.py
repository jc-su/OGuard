"""Test and compile projection's numba.py to PTX.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the result against a NumPy reference. Also writes numba.ptx.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "projection_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task, n, rng):
    price = rng.uniform(0.0, 100.0, size=max(n, 1)).astype(np.float64)
    discount = rng.uniform(0.0, 1.0, size=max(n, 1)).astype(np.float64)
    ref = price[:n] * (1.0 - discount[:n])

    d_price = cuda.to_device(price)
    d_discount = cuda.to_device(discount)
    d_out = cuda.device_array(max(n, 1), dtype=np.float64)

    task.run(d_price, d_discount, d_out, n)
    cuda.synchronize()

    got = d_out.copy_to_host()[:n]
    assert np.allclose(got, ref), f"n={n}: mismatch"
    print(f"n={n}: PASS")


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
