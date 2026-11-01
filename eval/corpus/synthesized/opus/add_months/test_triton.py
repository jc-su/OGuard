"""Test the Triton kernel of add_months against a NumPy reference."""

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

ARGS = ("days", "months", "out", "n")
FIRST_DAY = int(np.datetime64("0001-01-01", "D").astype(np.int64))
LAST_DAY = int(np.datetime64("9999-12-31", "D").astype(np.int64))
FIRST_MONTH = int(np.datetime64("0001-01", "M").astype(np.int64))
LAST_MONTH = int(np.datetime64("9999-12", "M").astype(np.int64))


def make_cases(n):
    for max_months in (120000, 25):
        days = rng.integers(FIRST_DAY, LAST_DAY, n, endpoint=True)
        # A third of the rows are among the last three days of a month.
        ends = rng.random(n) < 1 / 3
        month = rng.integers(FIRST_MONTH, LAST_MONTH, n, endpoint=True)
        month_end = (month + 1).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) - 1
        days = np.where(ends, month_end - rng.integers(0, 3, n), days)
        cur = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
        lo = np.maximum(FIRST_MONTH - cur, -max_months)
        hi = np.minimum(LAST_MONTH - cur, max_months)
        months = lo + (rng.random(n) * (hi - lo + 1)).astype(np.int64)
        months = np.minimum(months, hi)
        out = np.full(n, 0x7FFFFFFF, dtype=np.int32)
        case = dict(days=days.astype(np.int32), months=months.astype(np.int32), out=out, n=n)
        yield f"max_months={max_months}", case


def check(case, got):
    days = case["days"].astype("datetime64[D]")
    month = days.astype("datetime64[M]")
    day_of_month = (days - month.astype("datetime64[D]")).astype(np.int64)
    target = month + case["months"].astype(np.int64)
    start = target.astype("datetime64[D]").astype(np.int64)
    length = (target + 1).astype("datetime64[D]").astype(np.int64) - start
    return compare(got, {"out": (start + np.minimum(day_of_month, length - 1)).astype(np.int32)})


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
