"""GPU test for segmented_sum: Triton kernel against a NumPy reference.

n is the number of values. Two average segment lengths per size exercise
both kernel paths: about 8 values per segment (tile of segments) and about
128 (wide value tiles). Segment boundaries are random cut points, so segments
vary in length and some are empty. n=0 is also run with three empty segments.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory shadows the triton package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
AVG_LENGTHS = (8, 128)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "segmented_sum_triton", os.path.join(HERE, "triton.py")
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


def reference(offsets, values):
    g = offsets.size - 1
    sums = np.zeros(g, dtype=np.float64)
    nonempty = (offsets[1:] - offsets[:-1]) > 0
    if nonempty.any():
        # Consecutive non-empty starts bound each other, and the last one ends at n.
        sums[nonempty] = np.add.reduceat(values, offsets[:-1][nonempty])
    return sums


def check(task, n, g, rng, timed):
    cuts = np.sort(rng.integers(0, n + 1, max(g - 1, 0)))
    offsets = (
        np.concatenate([[0], cuts, [n]]).astype(np.int32) if g > 0 else np.zeros(1, dtype=np.int32)
    )
    values = rng.uniform(0.0, 1.0, n).astype(np.float64)
    expected = reference(offsets, values)

    d_offsets = torch.from_numpy(offsets).to(DEV)
    d_values = torch.from_numpy(values if n else np.zeros(1, dtype=np.float64)).to(DEV)
    d_sums = torch.full((max(g, 1),), float("nan"), dtype=torch.float64, device=DEV)
    task.run(d_offsets, d_values, d_sums, g)
    torch.cuda.synchronize()

    sums = d_sums[:g].cpu().numpy()
    assert np.allclose(sums, expected, rtol=1e-9, atol=1e-9), f"n={n} g={g}: sums mismatch"

    if timed:
        us = bench(lambda: task.run(d_offsets, d_values, d_sums, g), lambda: None)
        print(f"  n={n} g={g}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        if n == 0:
            check(task, 0, 3, rng, False)
        for avg in AVG_LENGTHS:
            g = 0 if n == 0 else max(n // avg, 1)
            check(task, n, g, rng, n == max(SIZES))
        print(f"  n={n}: ok")
    print("PASS segmented_sum")


if __name__ == "__main__":
    main()
