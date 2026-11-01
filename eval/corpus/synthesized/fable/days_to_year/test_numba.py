"""GPU test of the Numba days_to_year kernel against a NumPy datetime64 reference."""

import datetime
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory shadows the numba package while the script's
# directory is on sys.path, so drop it before importing numba.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
EPOCH = datetime.date(1970, 1, 1)
FIRST_DAY = (datetime.date(1, 1, 1) - EPOCH).days
LAST_DAY = (datetime.date(9999, 12, 31) - EPOCH).days
BOUNDARY_YEARS = (1, 2, 4, 100, 101, 400, 401, 1600, 1900, 1969, 1970, 2000, 2024, 2100, 2400, 9999)


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boundary_days():
    out = []
    for y in BOUNDARY_YEARS:
        for md in ((1, 1), (2, 28), (3, 1), (12, 31)):
            out.append((datetime.date(y, *md) - EPOCH).days)
    return np.array(out, dtype=np.int32)


def make_days(n, rng):
    days = rng.integers(FIRST_DAY, LAST_DAY + 1, size=n, dtype=np.int32)
    b = boundary_days()
    k = min(n, b.shape[0])
    days[:k] = b[:k]
    return days


def reference(days):
    dates = np.datetime64("1970-01-01", "D") + days.astype("timedelta64[D]")
    return (dates.astype("datetime64[Y]").astype(np.int64) + 1970).astype(np.int32)


def check(task, n, rng):
    days = make_days(n, rng)
    expected = reference(days)
    d_days = cuda.to_device(days)
    d_year = cuda.to_device(np.zeros(n, dtype=np.int32))
    task.run(d_days, d_year, n)
    cuda.synchronize()
    year = d_year.copy_to_host()
    bad = np.flatnonzero(year != expected)
    assert bad.shape[0] == 0, f"n={n}: {bad.shape[0]} mismatches, first day {days[bad[0]]}: got {year[bad[0]]}, expected {expected[bad[0]]}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(8)
    for n in SIZES:
        check(task, n, rng)
        print(f"days_to_year numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
