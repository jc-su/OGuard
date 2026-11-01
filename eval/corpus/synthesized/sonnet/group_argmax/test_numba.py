import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The task directory holds numba.py, which would shadow the numba package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

I64 = np.iinfo(np.int64)


def reference(group, value, g):
    n = len(group)
    best = np.full(g, -1, dtype=np.int32)
    if n == 0:
        return best
    maxv = np.full(g, I64.min, dtype=np.int64)
    np.maximum.at(maxv, group, value)
    cand = np.flatnonzero(value == maxv[group])
    groups, first = np.unique(group[cand], return_index=True)
    best[groups] = cand[first].astype(np.int32)
    return best


def cases(n, rng):
    for g in (1, 7, 1000, 1 << 20):
        group = rng.integers(0, g, size=n, dtype=np.int32)
        yield g, "random", group, rng.integers(I64.min, I64.max, size=n, dtype=np.int64, endpoint=True)
        yield g, "ties", group, rng.integers(-2, 3, size=n, dtype=np.int64)
        yield g, "equal", group, np.full(n, I64.min, dtype=np.int64)
        yield g, "ascending", group, np.arange(n, dtype=np.int64)
        yield g, "descending", group, np.arange(n, 0, -1, dtype=np.int64)
        yield g, "sparse", np.zeros(n, dtype=np.int32), rng.integers(-5, 6, size=n, dtype=np.int64)


def main():
    rng = np.random.default_rng(1)
    failed = False
    for n in (0, 1, 1000, 1000003):
        for g, name, group, value in cases(n, rng):
            expect = reference(group, value, g)
            d_group = cuda.to_device(group)
            d_value = cuda.to_device(value)
            d_best = cuda.to_device(np.full(g, -1, dtype=np.int32))
            task.run(d_group, d_value, d_best, n, g)
            cuda.synchronize()
            ok = np.array_equal(d_best.copy_to_host(), expect)
            print(f"group_argmax numba n={n} g={g} {name}: {'PASS' if ok else 'FAIL'}")
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
