"""Test the Numba kernel of between_bitmap against a NumPy reference."""

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

ARGS = ("values", "valid", "out", "n", "lo", "hi")
I64_MIN, I64_MAX = -(2**63), 2**63 - 1


def make_cases(n):
    values = rng.integers(-1000, 1001, n, dtype=np.int64)
    extreme = rng.random(n) < 0.01
    values[extreme] = np.where(rng.random(int(extreme.sum())) < 0.5, I64_MIN, I64_MAX)
    # Random bytes: the padding bits of the last byte are set at random as well.
    valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
    for lo, hi in ((-300, 450), (I64_MIN, I64_MAX), (5, -5), (I64_MIN, -1000), (7, 7)):
        out = np.full((n + 7) // 8, 0xA5, dtype=np.uint8)
        yield f"lo={lo} hi={hi}", dict(values=values, valid=valid, out=out, n=n, lo=lo, hi=hi)


def check(case, got):
    n = case["n"]
    is_valid = np.unpackbits(case["valid"], count=n, bitorder="little").astype(bool)
    bits = is_valid & (case["values"] >= case["lo"]) & (case["values"] <= case["hi"])
    return compare(got, {"out": np.packbits(bits, bitorder="little")})


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
