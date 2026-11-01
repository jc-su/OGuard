"""Test the Numba string_equal kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_string_equal_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_columns(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two string columns: 40% identical rows, 30% same length with one byte
    changed, 30% with a different length."""
    a_len = rng.integers(0, 17, size=n)
    a_off = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(a_len, out=a_off[1:])
    a_chars = rng.integers(97, 101, size=int(a_off[-1])).astype(np.uint8)

    kind = rng.random(n)
    other_len = rng.integers(0, 17, size=n)
    b_len = np.where(kind < 0.7, a_len, other_len)
    b_off = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(b_len, out=b_off[1:])
    total_b = int(b_off[-1])
    b_chars = rng.integers(97, 101, size=total_b).astype(np.uint8)
    row_of_b = np.repeat(np.arange(n), b_len)
    same_len = a_len == b_len
    copy = same_len[row_of_b]
    j = np.arange(total_b) - b_off[row_of_b]
    b_chars[copy] = a_chars[a_off[row_of_b[copy]] + j[copy]]
    changed = np.nonzero((kind >= 0.4) & (kind < 0.7) & (a_len > 0))[0]
    pos = b_off[changed] + (rng.random(changed.size) * a_len[changed]).astype(np.int64)
    b_chars[pos] ^= 1
    return a_off, a_chars, b_off, b_chars


def reference(
    a_off: np.ndarray, a_chars: np.ndarray, b_off: np.ndarray, b_chars: np.ndarray
) -> np.ndarray:
    n = a_off.size - 1
    a_len = np.diff(a_off)
    same_len = a_len == np.diff(b_off)
    row_of = np.repeat(np.arange(n), a_len)
    k = np.nonzero(same_len[row_of])[0]
    r = row_of[k]
    pos_b = b_off[r] + (k - a_off[r])
    bad = a_chars[k] != b_chars[pos_b]
    mismatches = np.bincount(r[bad], minlength=n)
    return (same_len & (mismatches == 0)).astype(np.int8)


def check(task: ModuleType, n: int, rng: np.random.Generator) -> bool:
    a_off, a_chars, b_off, b_chars = make_columns(n, rng)
    expected = reference(a_off, a_chars, b_off, b_chars)
    d_match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    task.run(
        cuda.to_device(a_off),
        cuda.to_device(a_chars),
        cuda.to_device(b_off),
        cuda.to_device(b_chars),
        d_match,
        n,
    )
    cuda.synchronize()
    match = d_match.copy_to_host()
    ok = match.tobytes() == expected.tobytes()
    print(f"n={n}: {'PASS' if ok else 'FAIL'} (matches {int(expected.sum())} of {n})")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(3)
    failed = [n for n in SIZES if not check(task, n, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
