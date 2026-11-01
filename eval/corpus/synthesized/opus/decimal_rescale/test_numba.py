"""Test the Numba kernel of decimal_rescale against a NumPy reference."""

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

ARGS = ("values", "out", "overflow", "n", "shift")
SPECIALS = np.array(
    [-(2**63), 2**63 - 1, 0, 1, -1, 5, -5, 15, -15, 25, -25, 922337203685477580, -922337203685477580,
     922337203685477581, -922337203685477581, 5 * 10**17, -5 * 10**17, 449999999999999999],
    dtype=np.int64,
)


def make_values(n):
    kind = rng.integers(0, 4, n)
    full = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
    scaled = full >> rng.integers(0, 64, n)
    special = SPECIALS[rng.integers(0, len(SPECIALS), n)]
    ties = rng.integers(-10**6, 10**6, n, endpoint=True) * 5
    return np.select([kind == 0, kind == 1, kind == 2], [full, scaled, special], ties).astype(np.int64)


def make_cases(n):
    values = make_values(n)
    for shift in range(-18, 19):
        out = np.full(n, 777, dtype=np.int64)
        overflow = np.full(n, 7, dtype=np.int8)
        yield f"shift={shift}", dict(values=values, out=out, overflow=overflow, n=n, shift=shift)


def reference(values, shift):
    """Exact reference on Python integers."""
    v = values.astype(object)
    if shift >= 0:
        prod = v * 10**shift
        fits = ((prod >= -(2**63)) & (prod < 2**63)).astype(bool)
        return np.where(fits, prod, 0).astype(np.int64), (~fits).astype(np.int8)
    d = 10**-shift
    a = np.abs(v)
    q = a // d
    rem = a - q * d
    q = q + (2 * rem >= d).astype(bool).astype(object)
    return np.where((v < 0).astype(bool), -q, q).astype(np.int64), np.zeros(len(values), dtype=np.int8)


def check(case, got):
    out, flag = reference(case["values"], case["shift"])
    return compare(got, {"out": out, "overflow": flag})


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
