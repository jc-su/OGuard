"""GPU test for filter_select: Triton kernel against a NumPy reference."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory shadows the triton package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "filter_select_triton", os.path.join(HERE, "triton.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(launch, reset, reps=20):
    """Average kernel time in microseconds; reset runs before every timed launch."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(reps):
        reset()
        start.record()
        launch()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    return total / reps * 1e3


def check(task, n, rng):
    quantity = rng.uniform(0.0, 100.0, n).astype(np.float64)
    threshold = 50.0
    expected = np.nonzero(quantity < threshold)[0].astype(np.int32)

    d_quantity = torch.from_numpy(quantity).to(DEV)
    d_sel = torch.empty(max(n, 1), dtype=torch.int32, device=DEV)
    d_count = torch.zeros(1, dtype=torch.int32, device=DEV)
    task.run(d_quantity, threshold, d_sel, d_count, n)
    torch.cuda.synchronize()

    count = int(d_count.item())
    assert count == expected.size, f"n={n}: count {count} != {expected.size}"
    sel = np.sort(d_sel[:count].cpu().numpy())
    assert np.array_equal(sel, expected), f"n={n}: sel mismatch"

    if n == max(SIZES):
        us = bench(
            lambda: task.run(d_quantity, threshold, d_sel, d_count, n),
            lambda: d_count.zero_(),
        )
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS filter_select")


if __name__ == "__main__":
    main()
