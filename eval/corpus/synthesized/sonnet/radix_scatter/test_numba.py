"""Test for the Numba radix_scatter kernel against a NumPy reference, for
n in {0, 1, 1000, 1000003} and a few shifts.

Run as: CUDA_VISIBLE_DEVICES=1 python -P test_numba.py
"""

import importlib.util
import pathlib
import sys

import numpy as np
from numba import cuda

_HERE = pathlib.Path(__file__).resolve().parent


def _load_task_module():
    spec = importlib.util.spec_from_file_location(
        "radix_scatter_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n, shift, rng):
    keys = rng.integers(0, 2**32, size=max(n, 1), dtype=np.uint64).astype(np.uint32)
    byte = ((keys[:n].astype(np.uint32) >> np.uint32(shift)) & np.uint32(255))
    counts = np.bincount(byte, minlength=256) if n > 0 else np.zeros(256, dtype=np.int64)
    bucket_start = np.zeros(256, dtype=np.int64)
    bucket_start[1:] = np.cumsum(counts)[:-1]
    cursors = bucket_start.astype(np.int32).copy()

    d_keys = cuda.to_device(keys)
    d_cursors = cuda.to_device(cursors)
    d_out = cuda.to_device(np.zeros(max(n, 1), dtype=np.int32))

    mod.run(d_keys, d_cursors, d_out, n, shift)
    cuda.synchronize()

    out = d_out.copy_to_host()[:n]
    final_cursors = d_cursors.copy_to_host()

    ok = True
    expect_end = bucket_start + counts
    if not np.array_equal(final_cursors, expect_end):
        ok = False
    if ok:
        for b in range(256):
            got = np.sort(out[bucket_start[b]:expect_end[b]])
            expect = np.sort(np.flatnonzero(byte == b))
            if not np.array_equal(got, expect):
                ok = False
                break
    print(f"n={n} shift={shift}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(6)
    cases = [
        (0, 0),
        (1, 8),
        (1000, 0),
        (1000, 16),
        (1000003, 0),
        (1000003, 24),
    ]
    ok = all(run_case(mod, n, s, rng) for n, s in cases)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
