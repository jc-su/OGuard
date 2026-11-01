import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def make_values(n, shift, rng):
    bits = rng.integers(0, 1 << 63, size=n, dtype=np.uint64) ^ (
        rng.integers(0, 2, size=n, dtype=np.uint64) << np.uint64(63)
    )
    sh = rng.integers(0, 64, size=n).astype(np.uint64)
    v = (bits >> sh).view(np.int64)
    k = abs(shift)
    p = 10**k
    specials = [0, 1, -1, INT64_MIN, INT64_MAX, INT64_MIN + 1, INT64_MAX - 1]
    lim = INT64_MAX // p
    for d in (-1, 0, 1):
        specials += [lim + d, -lim + d, -lim - 1 + d]
    for t in (1, 3, 5, 7, 15, 25, 35):
        h = t * p // 2 if p % 2 == 0 else t * p
        for d in (-1, 0, 1):
            specials += [h + d, -h + d]
    specials = [s for s in specials if INT64_MIN <= s <= INT64_MAX]
    cnt = min(len(specials), n)
    v[:cnt] = np.array(specials[:cnt], dtype=np.int64)
    return v


def reference(values, shift):
    vo = values.astype(object)
    if shift >= 0:
        prod = vo * (10**shift)
        ovf = (prod > INT64_MAX) | (prod < INT64_MIN)
        out = np.where(ovf, 0, prod).astype(np.int64)
        return out, ovf.astype(np.int8)
    d = 10 ** (-shift)
    sign = np.where(vo < 0, -1, 1)
    mag = np.abs(vo)
    q = (2 * mag + d) // (2 * d)
    return (sign * q).astype(np.int64), np.zeros(len(values), dtype=np.int8)


def check(n, shift, rng):
    values = make_values(n, shift, rng)
    exp_out, exp_ovf = reference(values, shift)
    d_values = cuda.to_device(values)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    d_ovf = cuda.to_device(np.full(n, 5, dtype=np.int8))
    mod.run(d_values, d_out, d_ovf, n, shift)
    cuda.synchronize()
    ok = np.array_equal(d_out.copy_to_host(), exp_out) and np.array_equal(d_ovf.copy_to_host(), exp_ovf)
    print(f"n={n} shift={shift}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    ok = True
    for n in (0, 1, 1000):
        for shift in range(-18, 19):
            ok &= check(n, shift, rng)
    for shift in (-18, -9, -1, 0, 1, 2, 9, 18):
        ok &= check(1000003, shift, rng)
    sys.exit(0 if ok else 1)
