"""Test the Numba ascii_upper kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_ascii_upper_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_column(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rows of 0 to 16 bytes drawn from all 256 byte values, with a bias to
    ASCII letters."""
    lengths = rng.integers(0, 17, size=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    chars = rng.integers(0, 256, size=total).astype(np.uint8)
    letters = rng.random(total) < 0.5
    chars[letters] = rng.integers(65, 123, size=int(letters.sum())).astype(np.uint8)
    return offsets, chars


def check(task: ModuleType, n: int, rng: np.random.Generator) -> bool:
    offsets, chars = make_column(n, rng)
    lower = (chars >= 97) & (chars <= 122)
    expected = np.where(lower, chars - np.uint8(32), chars).astype(np.uint8)

    d_out = cuda.to_device(np.zeros(chars.size, dtype=np.uint8))
    task.run(cuda.to_device(offsets), cuda.to_device(chars), d_out, n)
    cuda.synchronize()
    out = d_out.copy_to_host()
    ok = out.tobytes() == expected.tobytes()
    print(f"n={n}: {'PASS' if ok else 'FAIL'} ({chars.size} bytes)")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(4)
    failed = [n for n in SIZES if not check(task, n, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
