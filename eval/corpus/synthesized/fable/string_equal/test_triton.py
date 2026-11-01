import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("string_equal_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def offsets_of(lengths):
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)


def make_columns(n, rng):
    """Column a is random. Half the rows of b copy a; a third of those get one byte flipped."""
    la = rng.integers(0, 25, n)
    ao = offsets_of(la)
    a_chars = rng.integers(0, 256, ao[-1], dtype=np.uint8)
    copy = rng.random(n) < 0.5
    lb = np.where(copy, la, rng.integers(0, 25, n))
    bo = offsets_of(lb)
    b_chars = rng.integers(0, 256, bo[-1], dtype=np.uint8)
    row_b = np.repeat(np.arange(n), lb)
    pos_b = np.arange(bo[-1]) - np.repeat(bo[:-1], lb)
    sel = copy[row_b]
    b_chars[sel] = a_chars[np.repeat(ao[:-1], lb)[sel] + pos_b[sel]]
    flip = np.flatnonzero(copy & (la > 0) & (rng.random(n) < 0.33))
    at = bo[flip] + rng.integers(0, 1 << 30, flip.shape[0]) % lb[flip]
    b_chars[at] ^= 1
    return ao, a_chars, bo, b_chars


def reference(ao, a_chars, bo, b_chars, n):
    la = ao[1:] - ao[:-1]
    lb = bo[1:] - bo[:-1]
    same_len = la == lb
    row_a = np.repeat(np.arange(n), la)
    pos_a = np.arange(ao[-1]) - np.repeat(ao[:-1], la)
    sel = same_len[row_a]
    b_at = np.repeat(bo[:-1], la)[sel] + pos_a[sel]
    differs = a_chars[sel] != b_chars[b_at]
    bad = np.bincount(row_a[sel], weights=differs, minlength=n) > 0
    return (same_len & ~bad).astype(np.int8)


def check(n, rng):
    ao, a_chars, bo, b_chars = make_columns(n, rng)
    ref = reference(ao, a_chars, bo, b_chars, n)
    dev = "cuda"
    match = torch.empty(n, dtype=torch.int8, device=dev)
    mod.run(
        torch.from_numpy(ao).to(dev),
        torch.from_numpy(a_chars).to(dev),
        torch.from_numpy(bo).to(dev),
        torch.from_numpy(b_chars).to(dev),
        match,
        n,
    )
    got = match.cpu().numpy()
    assert np.array_equal(got, ref), f"string_equal mismatch at n={n}"
    if n >= 1000:
        assert 0 < ref.sum() < n, "test data should mix equal and unequal rows"


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    for n in (0, 1, 1000, 1000003):
        check(n, rng)
        print(f"string_equal triton n={n} ok")
