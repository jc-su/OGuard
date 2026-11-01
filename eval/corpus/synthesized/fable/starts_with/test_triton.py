"""GPU test for starts_with: Triton kernel against a NumPy reference.

Strings are random ASCII of length 0 to 20; about 30% are forced to begin
with the pattern. Every size is checked with k=3 ("abc"), k=0 (every string
matches) and k=25 (longer than any string: nothing matches).
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
PATTERN = np.frombuffer(b"abc", dtype=np.uint8)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "starts_with_triton", os.path.join(HERE, "triton.py")
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


def make_strings(rng, n):
    lengths = rng.integers(0, 21, n).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(ord("a"), ord("z") + 1, int(lengths.sum()), dtype=np.uint8)
    force = (rng.random(n) < 0.3) & (lengths >= 3)
    for j in range(3):
        chars[offsets[:-1][force] + j] = PATTERN[j]
    return offsets, chars


def reference(offsets, chars, prefix):
    n = offsets.size - 1
    k = prefix.size
    lengths = offsets[1:] - offsets[:-1]
    ok = lengths >= k
    last = max(chars.size - 1, 0)
    for j in range(k):
        at = np.minimum(offsets[:-1].astype(np.int64) + j, last)
        ok &= (chars[at] == prefix[j]) if chars.size else np.zeros(n, dtype=bool)
    return ok.astype(np.int8)


def check(task, n, rng):
    offsets, chars = make_strings(rng, n)
    d_offsets = torch.from_numpy(offsets).to(DEV)
    d_chars = torch.from_numpy(chars if chars.size else np.zeros(1, dtype=np.uint8)).to(DEV)
    d_match = torch.empty(max(n, 1), dtype=torch.int8, device=DEV)

    for prefix in (
        PATTERN,
        np.zeros(0, dtype=np.uint8),
        np.frombuffer(b"abc" + b"x" * 22, dtype=np.uint8),
    ):
        k = prefix.size
        expected = reference(offsets, chars, prefix)
        d_prefix = torch.from_numpy((prefix if k else np.zeros(1, dtype=np.uint8)).copy()).to(DEV)
        d_match.fill_(0x7F)
        task.run(d_offsets, d_chars, d_prefix, d_match, n, k)
        torch.cuda.synchronize()
        match = d_match[:n].cpu().numpy()
        assert np.array_equal(match, expected), f"n={n} k={k}: match mismatch"

        if n == max(SIZES) and k == 3:
            us = bench(
                lambda p=d_prefix, kk=k: task.run(d_offsets, d_chars, p, d_match, n, kk),
                lambda: None,
            )
            print(f"  n={n}: {us:.1f} us per launch (matches={int(expected.sum())})")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS starts_with")


if __name__ == "__main__":
    main()
