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


def check(task, n: int, valid_np: np.ndarray, label: str) -> None:
    valid = torch.from_numpy(valid_np).cuda()
    rows = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    task.run(valid, rows, count, n)
    torch.cuda.synchronize()
    ref = np.flatnonzero(np.unpackbits(valid_np, bitorder="little")[:n])
    got_count = int(count.item())
    assert got_count == len(ref), (n, label, got_count, len(ref))
    got = np.sort(rows[:got_count].cpu().numpy())
    assert np.array_equal(got, ref), (n, label)
    print(f"PASS valid_rows n={n} {label} count={got_count}")


def main() -> None:
    rng = np.random.default_rng(1)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        nbytes = (n + 7) // 8
        check(task, n, rng.integers(0, 256, size=nbytes, dtype=np.uint8), "random")
        check(task, n, np.full(nbytes, 255, dtype=np.uint8), "all_valid")
        check(task, n, np.zeros(nbytes, dtype=np.uint8), "all_null")
        if nbytes > 0:
            sparse = np.zeros(nbytes, dtype=np.uint8)
            sparse[rng.integers(0, nbytes, size=max(nbytes // 64, 1))] = 1 << 5
            check(task, n, sparse, "sparse")


if __name__ == "__main__":
    main()
