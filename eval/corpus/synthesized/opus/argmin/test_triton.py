"""Test the Triton kernel of argmin against a NumPy reference."""

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

ARGS = ("values", "row", "n")


def make_cases(n):
    full = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
    if n > 10:
        full[n // 2] = full[n - 3] = -(2**63)
    yield "full range", dict(values=full, row=np.full(1, -1, dtype=np.int32), n=n)
    yield "ties", dict(values=rng.integers(-25, 25, n, dtype=np.int64), row=np.full(1, -1, dtype=np.int32), n=n)
    yield "equal", dict(values=np.full(n, 7, dtype=np.int64), row=np.full(1, -1, dtype=np.int32), n=n)
    yield "decreasing", dict(values=np.arange(n, 0, -1, dtype=np.int64), row=np.full(1, -1, dtype=np.int32), n=n)
    yield "max", dict(values=np.full(n, 2**63 - 1, dtype=np.int64), row=np.full(1, -1, dtype=np.int32), n=n)


def check(case, got):
    want = int(np.argmin(case["values"])) if case["n"] else -1
    return compare(got, {"row": np.array([want], dtype=np.int32)})


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
