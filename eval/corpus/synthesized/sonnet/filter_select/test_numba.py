"""Test and compile filter_select's numba.py to PTX.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks the
result against a NumPy reference. Also writes numba.ptx.

The kernel file is named numba.py per the task spec, which collides
with the real numba package name, so it is loaded by file path with a
distinct module name. The task directory is never added to sys.path.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "filter_select_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(quantity, threshold):
    ids = np.nonzero(quantity < threshold)[0].astype(np.int32)
    return ids, np.int32(ids.shape[0])


def check(task, n, rng):
    quantity = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    threshold = 0.0

    ref_ids, ref_count = reference(quantity, threshold)

    d_quantity = cuda.to_device(quantity)
    d_sel = cuda.device_array(max(n, 1), dtype=np.int32)
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    task.run(d_quantity, d_sel, d_count, threshold, n)
    cuda.synchronize()

    got_count = d_count.copy_to_host()[0]
    assert got_count == ref_count, f"n={n}: count {got_count} != {ref_count}"

    if got_count > 0:
        got_ids = np.sort(d_sel.copy_to_host()[:got_count])
        assert np.array_equal(got_ids, np.sort(ref_ids)), f"n={n}: ids mismatch"

    print(f"n={n}: PASS (count={got_count})")


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
