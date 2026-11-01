"""GPU test for string_length: Triton kernel against a NumPy reference.

Two average lengths per size exercise both kernel paths: about 12 code points
per string (tile of strings) and about 96 (wide byte tiles). Code points are
1 to 4 bytes wide, so byte length and code point count differ.
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
AVG_CODE_POINTS = (12, 96)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "string_length_triton", os.path.join(HERE, "triton.py")
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


def make_strings(rng, n, avg_cp):
    """Arrow string column with random UTF-8 content. Returns offsets, chars, code point counts."""
    cp_counts = rng.integers(0, 2 * avg_cp + 1, n).astype(np.int64)
    total_cp = int(cp_counts.sum())
    w = rng.integers(0, 20, total_cp, dtype=np.int8)
    widths = (
        1 + (w >= 14).astype(np.int8) + (w >= 17).astype(np.int8) + (w >= 19).astype(np.int8)
    ).astype(np.int64)
    cp_starts = np.cumsum(widths) - widths
    total_bytes = int(widths.sum())
    chars = rng.integers(0x80, 0xC0, total_bytes, dtype=np.uint8)  # continuation bytes
    lead = np.where(widths == 1, rng.integers(0x41, 0x5B, total_cp), 0)
    lead = np.where(widths == 2, 0xC3, lead)
    lead = np.where(widths == 3, 0xE2, lead)
    lead = np.where(widths == 4, 0xF0, lead)
    chars[cp_starts] = lead.astype(np.uint8)
    cum_bytes = np.concatenate([[0], np.cumsum(widths)])
    cp_offsets = np.concatenate([[0], np.cumsum(cp_counts)])
    offsets = cum_bytes[cp_offsets].astype(np.int32)
    return offsets, chars, cp_counts.astype(np.int32)


def check(task, n, avg_cp, rng):
    offsets, chars, expected = make_strings(rng, n, avg_cp)
    # Independent reference: count non-continuation bytes per string.
    starts = ((chars & 0xC0) != 0x80).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(starts)])
    assert np.array_equal(cum[offsets[1:]] - cum[offsets[:-1]], expected)

    d_offsets = torch.from_numpy(offsets).to(DEV)
    d_chars = torch.from_numpy(chars if chars.size else np.zeros(1, dtype=np.uint8)).to(DEV)
    d_lengths = torch.empty(max(n, 1), dtype=torch.int32, device=DEV)
    task.run(d_offsets, d_chars, d_lengths, n)
    torch.cuda.synchronize()

    lengths = d_lengths[:n].cpu().numpy()
    assert np.array_equal(lengths, expected), f"n={n} avg={avg_cp}: lengths mismatch"

    if n == max(SIZES):
        us = bench(lambda: task.run(d_offsets, d_chars, d_lengths, n), lambda: None)
        print(f"  n={n} bytes={chars.size}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        for avg_cp in AVG_CODE_POINTS:
            check(task, n, avg_cp, rng)
        print(f"  n={n}: ok")
    print("PASS string_length")


if __name__ == "__main__":
    main()
