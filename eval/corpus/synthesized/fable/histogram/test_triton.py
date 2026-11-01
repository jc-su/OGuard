"""GPU test for histogram: Triton kernel against a NumPy reference."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory shadows the triton package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
SHIFTS = (0, 8, 16, 24)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "histogram_triton", os.path.join(HERE, "triton.py")
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


def check(task, n, shift, rng):
    keys = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    expected = np.bincount((keys >> np.uint32(shift)) & np.uint32(255), minlength=256).astype(
        np.int32
    )

    d_keys = torch.from_numpy(keys).to(DEV)
    d_counts = torch.zeros(256, dtype=torch.int32, device=DEV)
    task.run(d_keys, d_counts, shift, n)
    torch.cuda.synchronize()

    counts = d_counts.cpu().numpy()
    assert np.array_equal(counts, expected), f"n={n} shift={shift}: counts mismatch"

    if n == max(SIZES) and shift == 8:
        us = bench(lambda: task.run(d_keys, d_counts, shift, n), lambda: d_counts.zero_())
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        for shift in SHIFTS:
            check(task, n, shift, rng)
        print(f"  n={n}: ok")
    print("PASS histogram")


if __name__ == "__main__":
    main()
