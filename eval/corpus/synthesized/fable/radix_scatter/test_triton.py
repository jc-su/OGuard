import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the task's triton.py must not shadow the triton package
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]


def load_task():
    spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, keys_np: np.ndarray, shift: int, label: str) -> None:
    n = len(keys_np)
    digits = ((keys_np >> np.uint32(shift)) & np.uint32(255)).astype(np.int64)
    counts = np.bincount(digits, minlength=256)
    cursors_in = (np.cumsum(counts) - counts).astype(np.int32)
    keys = torch.from_numpy(keys_np.view(np.int32)).cuda().view(torch.uint32)
    cursors = torch.from_numpy(cursors_in.copy()).cuda()
    out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    task.run(keys, cursors, out, n, shift)
    torch.cuda.synchronize()
    got = out.cpu().numpy().astype(np.int64)
    # a permutation of the row ids, grouped by digit in cursor order
    assert np.array_equal(np.sort(got), np.arange(n)), (n, shift, label)
    expected_digit = np.repeat(np.arange(256), counts)
    assert np.array_equal(digits[got], expected_digit), (n, shift, label)
    assert np.array_equal(cursors.cpu().numpy(), cursors_in + counts.astype(np.int32)), (n, shift, label)
    print(f"PASS radix_scatter n={n} shift={shift} {label}")


def main() -> None:
    rng = np.random.default_rng(7)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        keys_np = rng.integers(0, 2**32, size=n, dtype=np.uint32)
        for shift in (0, 8, 16, 24):
            check(task, keys_np, shift, "uniform")
        check(task, np.full(n, 0xDEADBEEF, dtype=np.uint32), 8, "all_equal")
        check(task, rng.integers(0, 3, size=n, dtype=np.uint32) << np.uint32(16), 16, "three_digits")
        check(task, np.full(n, 0xFFFFFFFF, dtype=np.uint32), 24, "digit_255")


if __name__ == "__main__":
    main()
