import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

LIMIT = 2**62
UNITS = (1, 2, 7, 1000, 3600000000, 86400000000, 999999937, 2**40, 2**40 - 1, 1000003)


def make_ts(rng, n, unit):
    ts = rng.integers(-LIMIT, LIMIT + 1, n, dtype=np.int64)
    if n >= 8:
        ts[0] = -LIMIT
        ts[1] = LIMIT
        ts[2] = 0
        ts[3] = -1
        k = rng.integers(-LIMIT // unit + 1, LIMIT // unit, 4, dtype=np.int64)
        ts[4] = k[0] * unit
        ts[5] = k[1] * unit - 1
        ts[6] = k[2] * unit + 1
        ts[7] = k[3] * unit + unit - 1
    return ts


def cases():
    rng = np.random.default_rng(4)
    for n in (0, 1, 1000, 1000003):
        for unit in UNITS:
            yield f"n={n} unit={unit}", make_ts(rng, n, unit), unit
    for unit in (1, 7, 3600000000):
        yield f"pre-1970 unit={unit}", rng.integers(-86400000000 * 365 * 60, 0, 1000, dtype=np.int64), unit


def reference(ts, unit):
    return np.array([(int(t) // unit) * unit for t in ts], dtype=np.int64)


failed = 0
for name, ts, unit in cases():
    n = len(ts)
    expected = reference(ts, unit)
    d_out = cuda.to_device(np.full(n, 0x5A5A5A5A5A5A5A5A, dtype=np.int64))
    task.run(cuda.to_device(ts), d_out, n, unit)
    ok = np.array_equal(d_out.copy_to_host(), expected)
    print(("PASS" if ok else "FAIL"), name)
    failed += not ok
sys.exit(1 if failed else 0)
