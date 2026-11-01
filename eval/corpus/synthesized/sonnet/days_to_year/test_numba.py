"""Test for the Numba days_to_year kernel against Python's `datetime`
module (an independent reference implementation of the proleptic
Gregorian calendar), for n in {0, 1, 1000, 1000003}.

Run as: CUDA_VISIBLE_DEVICES=1 python -P test_numba.py
"""

import datetime
import importlib.util
import pathlib
import sys

import numpy as np
from numba import cuda

_HERE = pathlib.Path(__file__).resolve().parent
_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()
_MIN_DAY = datetime.date(1, 1, 1).toordinal() - _EPOCH_ORDINAL
_MAX_DAY = datetime.date(9999, 12, 31).toordinal() - _EPOCH_ORDINAL


def _load_task_module():
    spec = importlib.util.spec_from_file_location(
        "days_to_year_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_year(day):
    return datetime.date.fromordinal(int(day) + _EPOCH_ORDINAL).year


def run_case(mod, n, rng):
    days = rng.integers(_MIN_DAY, _MAX_DAY + 1, size=max(n, 1), dtype=np.int64).astype(np.int32)
    expected = np.array([reference_year(d) for d in days[:n]], dtype=np.int32)

    d_days = cuda.to_device(days)
    d_year = cuda.to_device(np.zeros(max(n, 1), dtype=np.int32))

    mod.run(d_days, d_year, n)
    cuda.synchronize()

    year = d_year.copy_to_host()[:n]
    ok = np.array_equal(year, expected)
    print(f"n={n}: -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_fixed_points(mod):
    known_days = np.array([0, -1, 1, _MIN_DAY, _MAX_DAY], dtype=np.int32)
    expected = np.array([1970, 1969, 1970, 1, 9999], dtype=np.int32)
    d_days = cuda.to_device(known_days)
    d_year = cuda.to_device(np.zeros(5, dtype=np.int32))
    mod.run(d_days, d_year, 5)
    cuda.synchronize()
    got = d_year.copy_to_host()
    ok = np.array_equal(got, expected)
    print(f"fixed points {got.tolist()} vs {expected.tolist()}: -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    print(f"valid day range: [{_MIN_DAY}, {_MAX_DAY}]")
    rng = np.random.default_rng(7)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, rng) for n in sizes)
    ok &= run_fixed_points(mod)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
