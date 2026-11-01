import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory must not shadow the triton package.
sys.path = [p for p in sys.path if p not in ("", HERE)]

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


def reference(values, valid, n, lo, hi):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    keep = bits & (values >= lo) & (values <= hi)
    return np.packbits(keep, bitorder="little")


def make_case(rng, n, kind):
    nbytes = (n + 7) // 8
    if kind == "small":
        values = rng.integers(-50, 50, n, dtype=np.int64)
        lo, hi = -10, 20
    elif kind == "wide":
        values = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
        lo, hi = -(2**62), 2**62
    elif kind == "full":
        values = rng.integers(I64_MIN, I64_MAX, n, dtype=np.int64, endpoint=True)
        values[: min(n, 2)] = [I64_MIN, I64_MAX][: min(n, 2)]
        lo, hi = I64_MIN, I64_MAX
    elif kind == "empty_range":
        values = rng.integers(-50, 50, n, dtype=np.int64)
        lo, hi = 10, -10
    elif kind == "point":
        values = rng.integers(-3, 3, n, dtype=np.int64)
        lo, hi = 1, 1
    else:
        raise ValueError(kind)
    # Padding bits of the last validity byte are random, to check the output mask.
    valid = rng.integers(0, 256, nbytes, dtype=np.uint8)
    return values, valid, lo, hi


def main():
    rng = np.random.default_rng(1)
    failed = 0
    for n in (0, 1, 7, 8, 9, 1000, 1000003):
        for kind in ("small", "wide", "full", "empty_range", "point"):
            values, valid, lo, hi = make_case(rng, n, kind)
            nbytes = (n + 7) // 8
            t_values = torch.from_numpy(values).cuda()
            t_valid = torch.from_numpy(valid).cuda()
            t_out = torch.full((nbytes,), 0xA5, dtype=torch.uint8, device="cuda")
            task.run(t_values, t_valid, t_out, n, lo, hi)
            torch.cuda.synchronize()
            got = t_out.cpu().numpy()
            want = reference(values, valid, n, lo, hi)
            ok = np.array_equal(got, want)
            failed += not ok
            print(("PASS" if ok else "FAIL"), "between_bitmap triton", n, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
