import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DAY_MIN = int(np.datetime64("0001-01-01", "D").astype(np.int64))
DAY_MAX = int(np.datetime64("9999-12-31", "D").astype(np.int64))


def reference(days, months):
    d = days.astype("datetime64[D]")
    mo = d.astype("datetime64[M]")
    dom = (d - mo).astype(np.int64)
    target = mo + months.astype("timedelta64[M]")
    t0 = target.astype("datetime64[D]")
    length = ((target + np.timedelta64(1, "M")).astype("datetime64[D]") - t0).astype(np.int64)
    return (t0 + np.minimum(dom, length - 1).astype("timedelta64[D]")).astype(np.int64).astype(np.int32)


def make(n, rng):
    days = rng.integers(DAY_MIN, DAY_MAX + 1, size=n, dtype=np.int64)
    if n > 0:
        k = n // 3
        days[:k] = rng.integers(np.datetime64("1996-01-01", "D").astype(np.int64),
                                np.datetime64("2032-12-31", "D").astype(np.int64), size=k)
        specials = [np.datetime64(s, "D").astype(np.int64)
                    for s in ("2024-02-29", "2023-01-31", "0001-01-01", "9999-12-31", "2000-02-29", "1900-03-31")]
        days[: min(n, len(specials))] = specials[:n]
    months = rng.integers(-120000, 120001, size=n, dtype=np.int64)
    mo = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)  # months since 1970-01
    lo_m = np.datetime64("0001-01", "M").astype(np.int64)
    hi_m = np.datetime64("9999-12", "M").astype(np.int64)
    months = np.clip(mo + months, lo_m, hi_m) - mo
    return days.astype(np.int32), months.astype(np.int32)


def check(n, seed):
    rng = np.random.default_rng(seed)
    days, months = make(n, rng)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int32))
    mod.run(cuda.to_device(days), cuda.to_device(months), d_out, n)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(days, months)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("add_months numba: ok")
