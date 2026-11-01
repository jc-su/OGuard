"""Test the Numba kernel of timestamp_floor against a NumPy reference."""

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

ARGS = ("ts", "out", "n", "unit")


def make_cases(n):
    for unit in (1, 7, 10**6, 3600 * 10**6, 86400 * 10**6, 2**40):
        kind = rng.integers(0, 4, n)
        wide = rng.integers(-(2**62), 2**62, n)
        multiple = rng.integers(-1000, 1001, n) * unit
        near = rng.integers(-(10**7), 10**7, n, endpoint=True)
        ts = np.select([kind == 0, kind == 1, kind == 2], [wide, multiple, multiple - 1], near).astype(np.int64)
        if n > 2:
            ts[0], ts[1] = -(2**62), 2**62
        yield f"unit={unit}", dict(ts=ts, out=np.full(n, 99, dtype=np.int64), n=n, unit=unit)


def check(case, got):
    # NumPy integer floor division rounds toward negative infinity.
    return compare(got, {"out": (case["ts"] // case["unit"]) * case["unit"]})


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
