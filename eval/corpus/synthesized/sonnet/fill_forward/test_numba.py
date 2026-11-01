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

MODES = ("half", "mostly_null", "all_null", "all_valid", "first_only", "last_only", "middle_only")


def make_case(n, mode, rng):
    values = rng.integers(-(1 << 63), 1 << 63, size=n, dtype=np.int64)
    if mode == "half":
        valid = rng.random(n) < 0.5
    elif mode == "mostly_null":
        valid = rng.random(n) < 0.001
    elif mode == "all_null":
        valid = np.zeros(n, dtype=bool)
    elif mode == "all_valid":
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.zeros(n, dtype=bool)
        if n:
            valid[{"first_only": 0, "last_only": n - 1, "middle_only": n // 2}[mode]] = True
    # the padding bits of the last byte are 1, they must be ignored
    padded = np.concatenate([valid, np.ones(-n % 8, dtype=bool)])
    bitmap = np.packbits(padded, bitorder="little")
    return values, valid, bitmap


def reference(values, valid):
    n = len(values)
    last = np.maximum.accumulate(np.where(valid, np.arange(n), -1)) if n else np.zeros(0, dtype=np.int64)
    return np.where(last >= 0, values[np.maximum(last, 0)], 0).astype(np.int64)


def check(n, mode, rng):
    values, valid, bitmap = make_case(n, mode, rng)
    expected = reference(values, valid)
    d_out = cuda.to_device(np.full(n, 77, dtype=np.int64))
    mod.run(cuda.to_device(values), cuda.to_device(bitmap), d_out, n)
    cuda.synchronize()
    ok = np.array_equal(d_out.copy_to_host(), expected)
    print(f"n={n} mode={mode}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(18)
    ok = True
    for n in (0, 1, 1000, 2048, 2049, 1000003):
        for mode in MODES:
            ok &= check(n, mode, rng)
    sys.exit(0 if ok else 1)
