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


def make_column(rng: np.random.Generator, n: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = rng.integers(0, max_len + 1, size=n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(97, 100, size=int(offsets[-1]), dtype=np.uint8)  # bytes a, b, c
    return offsets, chars


def reference(offsets: np.ndarray, chars: np.ndarray, pattern: bytes) -> np.ndarray:
    data = chars.tobytes()
    n = len(offsets) - 1
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        out[i] = data.find(pattern, int(offsets[i]), int(offsets[i + 1])) != -1
    return out


def check(task, offsets_np: np.ndarray, chars_np: np.ndarray, pattern: bytes, label: str) -> None:
    n = len(offsets_np) - 1
    k = len(pattern)
    offsets = torch.from_numpy(offsets_np).cuda()
    chars = torch.from_numpy(chars_np).cuda()
    pat = torch.from_numpy(np.frombuffer(pattern, dtype=np.uint8).copy()).cuda()
    match = torch.full((n,), 7, dtype=torch.int8, device="cuda")
    task.run(offsets, chars, pat, match, n, k)
    torch.cuda.synchronize()
    got = match.cpu().numpy()
    ref = reference(offsets_np, chars_np, pattern)
    bad = np.flatnonzero(got != ref)
    assert len(bad) == 0, (n, label, pattern, bad[:5], got[bad[:5]], ref[bad[:5]])
    print(f"PASS string_contains n={n} {label} pattern={pattern!r} matches={int(ref.sum())}")


def main() -> None:
    rng = np.random.default_rng(4)
    task = load_task()
    patterns = [b"", b"a", b"ab", b"abcabcabca", b"z", b"abcabcabcabcabcabcabc"]
    for n in (0, 1, 1000, 1000003):
        offsets_np, chars_np = make_column(rng, n, 24)
        for pattern in patterns:
            check(task, offsets_np, chars_np, pattern, "len<=24")
    # long strings and long patterns in a small column
    offsets_np, chars_np = make_column(rng, 1000, 400)
    for pattern in (b"abc" * 5, b"c" * 9, b"ab" * 20):
        check(task, offsets_np, chars_np, pattern, "len<=400")
    # every string empty
    offsets_np = np.zeros(1001, dtype=np.int32)
    chars_np = np.zeros(0, dtype=np.uint8)
    for pattern in (b"", b"a"):
        check(task, offsets_np, chars_np, pattern, "all_empty")


if __name__ == "__main__":
    main()
