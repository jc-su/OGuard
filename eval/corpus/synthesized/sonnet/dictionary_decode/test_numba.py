"""Test for the Numba dictionary_decode kernel against a NumPy reference,
for n in {0, 1, 1000, 1000003}.

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
        "dictionary_decode_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n, rng):
    d = 37
    dictionary = rng.uniform(-1e6, 1e6, size=d)
    codes = rng.integers(0, d, size=max(n, 1), dtype=np.int32)
    expected = dictionary[codes[:n]] if n > 0 else np.zeros(0)

    d_codes = cuda.to_device(codes)
    d_dict = cuda.to_device(dictionary)
    d_out = cuda.to_device(np.zeros(max(n, 1), dtype=np.float64))

    mod.run(d_codes, d_dict, d_out, n, d)
    cuda.synchronize()

    out = d_out.copy_to_host()[:n]
    ok = np.array_equal(out, expected)
    print(f"n={n}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(1)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, rng) for n in sizes)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
