"""Test for the Numba run_end_decode kernel against a NumPy reference, for
n in {0, 1, 1000, 1000003}.

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
        "run_end_decode_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_runs(n, rng):
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    max_runs = min(n, 200)
    r = rng.integers(1, max_runs + 1)
    if n > 1 and r > 1:
        cuts = rng.choice(np.arange(1, n), size=min(r - 1, n - 1), replace=False)
        cuts = np.sort(cuts)
    else:
        cuts = np.zeros(0, dtype=np.int64)
    run_ends = np.concatenate([cuts, [n]]).astype(np.int32)
    run_values = rng.uniform(-1e6, 1e6, size=run_ends.size)
    return run_ends, run_values


def reference(run_ends, run_values, n):
    out = np.empty(n, dtype=np.float64)
    prev = 0
    for j in range(run_ends.size):
        out[prev:run_ends[j]] = run_values[j]
        prev = run_ends[j]
    return out


def run_case(mod, n, rng):
    run_ends, run_values = make_runs(n, rng)
    r = run_ends.size
    expected = reference(run_ends, run_values, n) if n > 0 else np.zeros(0)

    d_values = cuda.to_device(run_values if r > 0 else np.zeros(1))
    d_ends = cuda.to_device(run_ends if r > 0 else np.zeros(1, dtype=np.int32))
    d_out = cuda.to_device(np.zeros(max(n, 1), dtype=np.float64))

    mod.run(d_values, d_ends, d_out, r, n)
    cuda.synchronize()

    out = d_out.copy_to_host()[:n]
    ok = np.array_equal(out, expected)
    print(f"n={n} r={r}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(2)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, rng) for n in sizes)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
