"""Test the Numba kernel of range_window_sum against a NumPy reference."""

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

ARGS = ("ts", "values", "out", "n", "range")


def make_ts(n, mode):
    if mode == "small gaps":
        ts = rng.integers(-2 * n, 2 * n, n, endpoint=True)
    elif mode == "all equal":
        ts = np.full(n, -3)
    elif mode == "wide":
        ts = rng.integers(-(2**62), 2**62, n)
        if n > 2:
            ts[0], ts[1] = -(2**62), 2**62
    else:  # long runs of peers
        ts = rng.integers(0, 7, n) * 1000
    return np.sort(ts).astype(np.int64)


def make_cases(n):
    for mode in ("small gaps", "all equal", "wide", "peers"):
        for width in (0, 5, 1000, 2**62) + ((2**58,) if mode == "wide" else ()):
            values = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
            case = dict(ts=make_ts(n, mode), values=values, out=np.full(n, 99, dtype=np.int64), n=n, range=width)
            yield f"{mode} range={width}", case


def check(case, got):
    ts, values = case["ts"], case["values"]
    prefix = np.zeros(len(ts) + 1, dtype=np.uint64)
    np.cumsum(values.view(np.uint64), out=prefix[1:])
    lo = np.searchsorted(ts, ts - case["range"], side="left")
    hi = np.searchsorted(ts, ts, side="right")
    return compare(got, {"out": (prefix[hi] - prefix[lo]).view(np.int64)})


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
