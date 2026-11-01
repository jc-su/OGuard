import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def random_lens(rng, n, max_len, empty_frac):
    lens = rng.integers(0, max_len + 1, n)
    lens[rng.random(n) < empty_frac] = 0
    return lens


def column(lens, rng):
    offsets = np.zeros(len(lens) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lens)
    chars = rng.integers(0, 256, int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(a_offsets, a_chars, b_offsets, b_chars):
    n = len(a_offsets) - 1
    out_offsets = a_offsets + b_offsets
    la = (a_offsets[1:] - a_offsets[:-1]).astype(np.int64)
    lb = (b_offsets[1:] - b_offsets[:-1]).astype(np.int64)
    total = int(out_offsets[-1])
    out = np.zeros(total, dtype=np.uint8)
    row = np.repeat(np.arange(n), la + lb)
    within = np.arange(total) - out_offsets[row].astype(np.int64)
    from_a = within < la[row]
    out[from_a] = a_chars[a_offsets[row][from_a].astype(np.int64) + within[from_a]]
    from_b = ~from_a
    out[from_b] = b_chars[b_offsets[row][from_b].astype(np.int64) + within[from_b] - la[row][from_b]]
    return out_offsets, out


def cases():
    rng = np.random.default_rng(2)
    for n in (0, 1, 1000, 1000003):
        a = column(random_lens(rng, n, 24, 0.2), rng)
        b = column(random_lens(rng, n, 24, 0.2), rng)
        yield f"n={n}", a, b
    la = random_lens(rng, 1000, 20, 0.3)
    lb = random_lens(rng, 1000, 20, 0.3)
    la[17] = 500000
    lb[300] = 300000
    la[300] = 0
    lb[17] = 0
    yield "long strings", column(la, rng), column(lb, rng)
    zeros = np.zeros(1000, dtype=np.int64)
    yield "all empty", column(zeros, rng), column(zeros, rng)
    yield "a empty", column(zeros, rng), column(random_lens(rng, 1000, 40, 0.1), rng)
    yield "b empty", column(random_lens(rng, 1000, 40, 0.1), rng), column(zeros, rng)


def dev(a):
    return torch.from_numpy(a).cuda()


failed = 0
for name, (a_off, a_chars), (b_off, b_chars) in cases():
    n = len(a_off) - 1
    out_offsets, expected = reference(a_off, a_chars, b_off, b_chars)
    d_out = torch.full((len(expected),), 0xAA, dtype=torch.uint8, device="cuda")
    task.run(dev(a_off), dev(a_chars), dev(b_off), dev(b_chars), dev(out_offsets), d_out, n)
    torch.cuda.synchronize()
    ok = np.array_equal(d_out.cpu().numpy(), expected)
    print(("PASS" if ok else "FAIL"), name)
    failed += not ok
sys.exit(1 if failed else 0)
