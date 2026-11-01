"""GPU test of the Numba dictionary_decode kernel against a NumPy reference."""

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
DICT_SIZES = (1, 7, 4096)


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, n, d, rng):
    codes = rng.integers(0, d, size=n, dtype=np.int32)
    dictionary = rng.standard_normal(d) * 1e3
    expected = dictionary[codes]

    d_codes = cuda.to_device(codes)
    d_dict = cuda.to_device(dictionary)
    d_out = cuda.to_device(np.full(n, np.nan))
    task.run(d_codes, d_dict, d_out, n, d)
    cuda.synchronize()
    out = d_out.copy_to_host()
    assert np.array_equal(out, expected), f"n={n} d={d}: output differs"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(2)
    for n in SIZES:
        for d in DICT_SIZES:
            check(task, n, d, rng)
        print(f"dictionary_decode numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
