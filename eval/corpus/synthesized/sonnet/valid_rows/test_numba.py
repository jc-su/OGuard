"""Test for the Numba valid_rows kernel against a NumPy reference, for
n in {0, 1, 1000, 1000003}.

Run as: CUDA_VISIBLE_DEVICES=1 python -P test_numba.py
(-P, or running from outside this directory, keeps this directory's own
numba.py from shadowing the real `numba` package.)
"""

import importlib.util
import pathlib
import sys

import numpy as np
from numba import cuda

_HERE = pathlib.Path(__file__).resolve().parent


def _load_task_module():
    spec = importlib.util.spec_from_file_location(
        "valid_rows_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n, rng):
    nbytes = max((n + 7) // 8, 1)
    valid = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    if n % 8 != 0 and n > 0:
        # Clear the padding bits above row n so they can't be counted.
        used_bits = n % 8
        pad_mask = np.uint8((1 << used_bits) - 1)
        valid[-1] &= pad_mask

    bits = np.unpackbits(valid, bitorder="little")[:n]
    expected = np.flatnonzero(bits).astype(np.int32)

    d_valid = cuda.to_device(valid)
    d_rows = cuda.to_device(np.zeros(max(n, 1), dtype=np.int32))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    mod.run(d_valid, d_rows, d_count, n)
    cuda.synchronize()

    count = d_count.copy_to_host()[0]
    rows = np.sort(d_rows.copy_to_host()[:count])

    ok = count == expected.size and np.array_equal(rows, expected)
    print(f"n={n}: count={count} expected={expected.size} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(0)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, rng) for n in sizes)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
