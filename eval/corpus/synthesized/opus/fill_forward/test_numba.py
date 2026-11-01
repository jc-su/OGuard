"""Test the Numba kernel of fill_forward against a NumPy reference."""

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

ARGS = ("values", "valid", "out", "n")


def make_cases(n):
    for mode, prob in (("half valid", 0.5), ("all null", 0.0), ("sparse", 1 / 300000), ("all valid", 1.0)):
        bits = rng.random(n) < prob
        valid = np.packbits(bits, bitorder="little")
        if n % 8:
            valid[-1] |= np.uint8((0xFF << (n % 8)) & 0xFF)  # padding bits set
        values = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
        yield mode, dict(values=values, valid=valid, out=np.full(n, 99, dtype=np.int64), n=n)


def check(case, got):
    n = case["n"]
    bits = np.unpackbits(case["valid"], count=n, bitorder="little").astype(bool)
    last = np.where(bits, np.arange(n), -1)
    np.maximum.accumulate(last, out=last)
    want = np.where(last >= 0, case["values"][np.maximum(last, 0)], 0) if n else np.zeros(0, dtype=np.int64)
    return compare(got, {"out": want.astype(np.int64)})


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
