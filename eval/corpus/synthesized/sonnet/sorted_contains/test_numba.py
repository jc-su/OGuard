"""Test for the Numba sorted_contains kernel against a NumPy reference,
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
        "sorted_contains_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n, s, rng):
    universe = max(2 * (s + n) + 10, 1000)
    members = np.sort(rng.choice(universe, size=s, replace=False)).astype(np.int64) if s > 0 else np.zeros(0, dtype=np.int64)
    keys = rng.integers(0, universe, size=max(n, 1)).astype(np.int64)
    member_set = set(members.tolist())
    expected = np.array([1 if k in member_set else 0 for k in keys[:n]], dtype=np.int8)

    d_keys = cuda.to_device(keys)
    d_members = cuda.to_device(members if s > 0 else np.zeros(1, dtype=np.int64))
    d_match = cuda.to_device(np.zeros(max(n, 1), dtype=np.int8))

    mod.run(d_keys, d_members, d_match, n, s)
    cuda.synchronize()

    match = d_match.copy_to_host()[:n]
    ok = np.array_equal(match, expected)
    hits = int(expected.sum()) if n > 0 else 0
    print(f"n={n} s={s}: hits={hits} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(9)
    cases = [
        (0, 0),
        (0, 500),
        (1, 0),
        (1, 1),
        (1000, 300),
        (1000003, 0),
        (1000003, 200000),
    ]
    ok = all(run_case(mod, n, s, rng) for n, s in cases)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
