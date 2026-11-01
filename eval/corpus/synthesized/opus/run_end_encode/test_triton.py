"""Test the Triton kernel of run_end_encode against a NumPy reference."""

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

ARGS = ("values", "run_ends", "run_values", "count", "n")


def runs_of(n, change_prob, alphabet):
    change = rng.random(n) < change_prob
    return np.where(change, rng.integers(0, alphabet, n), -1).astype(np.int64), change


def make_values(n, mode):
    if mode == "one run":
        return np.full(n, 5, dtype=np.int64)
    if mode == "distinct":
        return rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
    prob, alphabet = (0.25, 3) if mode == "short runs" else (1e-5, 2**62)
    fresh, change = runs_of(n, prob, alphabet)
    # Forward fill the drawn values over the rows that keep the previous value.
    idx = np.where(change, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = fresh[idx]
    filled[idx == 0] = fresh[0] if n and change[0] else 0
    return filled


def make_cases(n):
    for mode in ("short runs", "one run", "distinct", "long runs"):
        case = dict(
            values=make_values(n, mode),
            run_ends=np.full(n, -5, dtype=np.int32),
            run_values=np.full(n, -6, dtype=np.int64),
            count=np.zeros(1, dtype=np.int32),
            n=n,
        )
        yield mode, case


def check(case, got):
    values, n = case["values"], case["n"]
    ends = np.flatnonzero(np.append(values[1:] != values[:-1], True)) if n else np.zeros(0, dtype=np.int64)
    r = len(ends)
    run_ends = case["run_ends"].copy()
    run_values = case["run_values"].copy()
    run_ends[:r] = ends + 1
    run_values[:r] = values[ends]
    return compare(got, {"count": np.array([r], dtype=np.int32), "run_ends": run_ends, "run_values": run_values})


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
