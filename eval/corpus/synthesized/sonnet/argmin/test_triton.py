import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The task directory holds triton.py, which would shadow the triton package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

I64 = np.iinfo(np.int64)


def cases(n, rng):
    yield "random", rng.integers(I64.min, I64.max, size=n, dtype=np.int64, endpoint=True)
    yield "ties", rng.integers(-3, 4, size=n, dtype=np.int64)
    yield "equal", np.full(n, 7, dtype=np.int64)
    yield "max", np.full(n, I64.max, dtype=np.int64)
    yield "min", np.full(n, I64.min, dtype=np.int64)
    yield "descending", np.arange(n, 0, -1, dtype=np.int64)
    yield "ascending", np.arange(n, dtype=np.int64)
    last = rng.integers(0, 1000, size=n, dtype=np.int64) + 1
    if n:
        last[-1] = 0
    yield "min_last", last
    two = np.full(n, I64.max, dtype=np.int64)
    if n:
        two[n // 2] = I64.min
        two[-1] = I64.min
    yield "two_min", two


def main():
    rng = np.random.default_rng(1)
    failed = False
    for n in (0, 1, 1000, 1000003):
        for name, values in cases(n, rng):
            expect = int(np.argmin(values)) if n else -1
            t_values = torch.from_numpy(values).cuda()
            t_row = torch.full((1,), -1, dtype=torch.int32, device="cuda")
            task.run(t_values, t_row, n)
            torch.cuda.synchronize()
            got = int(t_row.cpu()[0])
            ok = got == expect
            print(f"argmin triton n={n} {name}: {'PASS' if ok else f'FAIL got {got} expect {expect}'}")
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
