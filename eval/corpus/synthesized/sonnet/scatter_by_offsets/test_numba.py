"""Test and compile scatter_by_offsets's numba.py to PTX.

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
        "scatter_by_offsets_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task, n, rng):
    flags = (
        (rng.uniform(size=n) < 0.3).astype(np.int8)
        if n > 0
        else np.zeros(0, dtype=np.int8)
    )
    offsets = np.zeros(max(n, 1), dtype=np.int32)
    if n > 0:
        offsets[:n] = np.concatenate(
            ([0], np.cumsum(flags[:-1], dtype=np.int32))
        ).astype(np.int32)
    total = int(flags.sum()) if n > 0 else 0

    ref = np.full(max(total, 1), -1, dtype=np.int32)
    for i in range(n):
        if flags[i]:
            ref[offsets[i]] = i

    d_flags = cuda.to_device(flags if n > 0 else np.zeros(1, dtype=np.int8))
    d_offsets = cuda.to_device(offsets)
    d_out = cuda.device_array(max(total, 1), dtype=np.int32)

    task.run(d_flags, d_offsets, d_out, n, total)
    cuda.synchronize()

    got = d_out.copy_to_host()[:total]
    assert np.array_equal(got, ref[:total]), f"n={n}: mismatch"
    print(f"n={n}: PASS (total={total})")


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
