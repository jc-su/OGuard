"""Test the Triton kernel of segmented_cummax against a NumPy reference."""

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

ARGS = ("values", "first", "out", "n")
I64_MIN = -(2**63)


def make_first(n, mode):
    if mode == "every row":
        start = np.ones(n, dtype=bool)
    elif mode == "one partition":
        start = np.zeros(n, dtype=bool)
    else:
        start = rng.random(n) < (1 / 8 if mode == "short" else 1 / 300000)
    if n:
        start[0] = True
    first = np.where(start, np.arange(n), 0)
    np.maximum.accumulate(first, out=first)
    return first.astype(np.int32)


def make_cases(n):
    for mode in ("short", "one partition", "every row", "long"):
        values = rng.integers(I64_MIN, 2**63 - 1, n, dtype=np.int64, endpoint=True)
        small = rng.random(n) < 0.5
        values[small] = rng.integers(-500, 500, int(small.sum()))
        values[rng.random(n) < 1 / 64] = I64_MIN
        if mode == "one partition" and n > 10:
            # Decreasing values after an early maximum: the carry must reach every block.
            values[3] = 2**63 - 1
        yield mode, dict(values=values, first=make_first(n, mode), out=np.full(n, 99, dtype=np.int64), n=n)


def check(case, got):
    values, first, n = case["values"], case["first"], case["n"]
    want = np.empty(n, dtype=np.int64)
    starts = np.flatnonzero(first == np.arange(n))
    bounds = np.append(starts, n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        np.maximum.accumulate(values[a:b], out=want[a:b])
    return compare(got, {"out": want})


def execute(case):
    import torch

    module = load("triton")
    dev = {k: torch.from_numpy(v.copy()).to("cuda") if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    torch.cuda.synchronize()
    return {k: dev[k].cpu().numpy() for k, v in case.items() if isinstance(v, np.ndarray)}


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
