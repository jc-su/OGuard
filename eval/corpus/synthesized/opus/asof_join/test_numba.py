"""Test the Numba kernel of asof_join against a NumPy reference."""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py and triton.py in this directory would shadow the packages.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]

SIZES = (0, 1, 1000, 1000003)
rng = np.random.default_rng(20261003)


def load(name):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{os.path.basename(HERE)}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(got, want):
    for name, expected in want.items():
        actual = got[name]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            bad = np.flatnonzero(actual != expected) if actual.shape == expected.shape else []
            where = f" first mismatch at {bad[0]}: got {actual[bad[0]]} want {expected[bad[0]]}" if len(bad) else ""
            return f"{name} differs{where}"
    return None

ARGS = ("left", "right", "out", "n", "r", "tolerance")


def make_cases(n):
    for r in (0, 1, n // 2 + 1, 2 * n + 5):
        for tolerance, spread in ((0, 1000), (3, 1000), (2**62, 2**61), (10**6, 2**40)):
            right = np.sort(rng.integers(-spread, spread, r, dtype=np.int64, endpoint=True))
            left = rng.integers(-spread, spread, n, dtype=np.int64, endpoint=True)
            left[::97] = 2**61
            left[1::97] = -(2**61)
            out = np.full(n, 12345, dtype=np.int32)
            case = dict(left=left, right=right, out=out, n=n, r=r, tolerance=tolerance)
            yield f"r={r} tolerance={tolerance}", case


def check(case, got):
    left, right = case["left"], case["right"]
    j = np.searchsorted(right, left, side="right") - 1
    found = right[np.maximum(j, 0)] if len(right) else np.zeros_like(left)
    hit = (j >= 0) & (left - found <= case["tolerance"])
    return compare(got, {"out": np.where(hit, j, -1).astype(np.int32)})


def execute(case):
    from numba import cuda

    module = load("numba")
    dev = {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    cuda.synchronize()
    return {k: dev[k].copy_to_host() for k, v in case.items() if isinstance(v, np.ndarray)}


def main():
    failures = 0
    for n in SIZES:
        for label, case in make_cases(n):
            error = check(case, execute(case))
            if error:
                failures += 1
                print(f"FAIL n={n} {label}: {error}")
        print(f"n={n} done")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
