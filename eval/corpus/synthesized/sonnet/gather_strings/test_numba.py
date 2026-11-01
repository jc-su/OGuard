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


def make_column(rng, n, max_len, empty_frac):
    lens = rng.integers(0, max_len + 1, n)
    lens[rng.random(n) < empty_frac] = 0
    return lens


def column(lens, rng):
    offsets = np.zeros(len(lens) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lens)
    chars = rng.integers(0, 256, int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(offsets, chars, idx):
    lens = (offsets[idx.astype(np.int64) + 1] - offsets[idx]).astype(np.int64)
    out_offsets = np.zeros(len(idx) + 1, dtype=np.int32)
    out_offsets[1:] = np.cumsum(lens)
    total = int(out_offsets[-1])
    row = np.repeat(np.arange(len(idx)), lens)
    within = np.arange(total) - out_offsets[row].astype(np.int64)
    src = offsets[idx][row].astype(np.int64) + within
    return out_offsets, chars[src]


def cases():
    rng = np.random.default_rng(1)
    for n in (0, 1, 1000, 1000003):
        offsets, chars = column(make_column(rng, n, 24, 0.2), rng)
        for m in (0, 1, 1000, n):
            if n == 0 and m > 0:
                continue
            idx = rng.integers(0, n, m).astype(np.int32) if n else np.zeros(0, np.int32)
            yield f"n={n} m={m}", offsets, chars, idx
    perm = rng.permutation(1000003).astype(np.int32)
    offsets, chars = column(make_column(rng, 1000003, 24, 0.2), rng)
    yield "permutation", offsets, chars, perm
    lens = make_column(rng, 1000, 20, 0.3)
    lens[17] = 500000
    offsets, chars = column(lens, rng)
    idx = rng.integers(0, 1000, 1000).astype(np.int32)
    idx[[3, 400, 999]] = 17
    yield "long string", offsets, chars, idx
    offsets, chars = column(np.zeros(1000, dtype=np.int64), rng)
    yield "all empty", offsets, chars, rng.integers(0, 1000, 1000).astype(np.int32)


failed = 0
for name, offsets, chars, idx in cases():
    n, m = len(offsets) - 1, len(idx)
    out_offsets, expected = reference(offsets, chars, idx)
    d_out = cuda.to_device(np.full(len(expected), 0xAA, dtype=np.uint8))
    task.run(
        cuda.to_device(offsets),
        cuda.to_device(chars),
        cuda.to_device(idx),
        cuda.to_device(out_offsets),
        d_out,
        n,
        m,
    )
    ok = np.array_equal(d_out.copy_to_host(), expected)
    print(("PASS" if ok else "FAIL"), name)
    failed += not ok
sys.exit(1 if failed else 0)
