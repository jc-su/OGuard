"""GPU test of the Triton string_prefix kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
WIDTHS = (1, 3, 8, 33, 64)


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("string_prefix_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_column(rng: np.random.Generator, n: int, base: int) -> tuple[np.ndarray, np.ndarray]:
    """Return offsets and chars of n strings whose first byte is at chars[base]."""
    lengths = rng.integers(0, 100, n, dtype=np.int64)
    lengths[rng.random(n) < 0.1] = 0
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = base
    np.cumsum(lengths, out=offsets[1:])
    offsets[1:] += base
    chars = rng.integers(0, 256, int(offsets[n]), dtype=np.uint8)
    return offsets.astype(np.int32), chars


def reference(offsets: np.ndarray, chars: np.ndarray, n: int, k: int) -> np.ndarray:
    expected = np.zeros((n, k), dtype=np.uint8)
    begin = offsets[:-1].astype(np.int64)
    lengths = offsets[1:].astype(np.int64) - begin
    for col in range(k):
        has = lengths > col
        expected[has, col] = chars[begin[has] + col]
    return expected.reshape(-1)


def check(task: ModuleType, rng: np.random.Generator, n: int, k: int, base: int) -> None:
    offsets, chars = make_column(rng, n, base)
    expected = reference(offsets, chars, n, k)

    d_offsets = torch.from_numpy(offsets).cuda()
    d_chars = torch.from_numpy(chars).cuda()
    # Fill with a nonzero byte so that the test sees the zero padding written.
    d_out = torch.full((n * k,), 0xAB, dtype=torch.uint8, device="cuda")
    task.run(d_offsets, d_chars, d_out, n, k)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()
    assert np.array_equal(got, expected), f"n={n} k={k} base={base}: output differs"


def test_string_prefix() -> None:
    task = load_task()
    rng = np.random.default_rng(2)
    for n in SIZES:
        for k in WIDTHS:
            for base in (0, 5):
                check(task, rng, n, k, base)
                print(f"string_prefix n={n} k={k} offsets[0]={base}: ok")
    offsets, chars = make_column(rng, 4, 0)
    for k in (0, 65):
        try:
            task.run(
                torch.from_numpy(offsets).cuda(),
                torch.from_numpy(chars).cuda(),
                torch.empty(4 * max(k, 1), dtype=torch.uint8, device="cuda"),
                4,
                k,
            )
        except ValueError:
            print(f"string_prefix k={k} rejected: ok")
        else:
            raise AssertionError(f"k={k} was not rejected")


if __name__ == "__main__":
    test_string_prefix()
    print("PASS")
