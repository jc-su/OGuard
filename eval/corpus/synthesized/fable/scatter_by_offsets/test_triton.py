"""GPU test for scatter_by_offsets: Triton kernel against a NumPy reference."""

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
        "scatter_by_offsets_triton", os.path.join(HERE, "triton.py")
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
    flags = rng.integers(0, 2, n).astype(np.int8)
    offsets = (np.cumsum(flags, dtype=np.int64) - flags).astype(np.int32)
    total = int(flags.sum())
    expected = np.nonzero(flags)[0].astype(np.int32)

    d_flags = torch.from_numpy(flags).to(DEV)
    d_offsets = torch.from_numpy(offsets).to(DEV)
    d_out = torch.full((max(total, 1),), -1, dtype=torch.int32, device=DEV)
    task.run(d_flags, d_offsets, d_out, n, total)
    torch.cuda.synchronize()

    out = d_out[:total].cpu().numpy()
    assert np.array_equal(out, expected), f"n={n}: out mismatch"

    if n == max(SIZES):
        us = bench(lambda: task.run(d_flags, d_offsets, d_out, n, total), lambda: None)
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS scatter_by_offsets")


if __name__ == "__main__":
    main()
