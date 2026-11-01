"""GPU test of the Numba sorted_contains kernel against a NumPy reference."""

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
MEMBER_COUNTS = (0, 1, 100, 100000)


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, n, s_target, rng):
    members = np.unique(rng.integers(-(2**40), 2**40, size=s_target, dtype=np.int64))
    s = members.shape[0]
    keys = rng.integers(-(2**40), 2**40, size=n, dtype=np.int64)
    if s > 0 and n > 0:
        hit = rng.random(n) < 0.5
        keys[hit] = members[rng.integers(0, s, size=int(hit.sum()))]
    expected = np.isin(keys, members).astype(np.int8)

    d_keys = cuda.to_device(keys)
    d_members = cuda.to_device(members)
    d_match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    task.run(d_keys, d_members, d_match, n, s)
    cuda.synchronize()
    match = d_match.copy_to_host()
    assert np.array_equal(match, expected), f"n={n} s={s}: match differs at {np.flatnonzero(match != expected)[:5]}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(10)
    for n in SIZES:
        for s in MEMBER_COUNTS:
            check(task, n, s, rng)
        print(f"sorted_contains numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
