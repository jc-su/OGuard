"""Test the transpose Numba kernel on the GPU against a NumPy reference.

The shapes cover rows * cols in {0, 1, 1000, 1000003} and shapes that are not
multiples of the 32 x 32 tile.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SHAPES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 5),
    (5, 0),
    (1, 1),
    (1, 1000),
    (1000, 1),
    (25, 40),
    (40, 25),
    (1, 1000003),
    (1000003, 1),
    (33, 31),
    (999, 1001),
    (1024, 1024),
)


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("transpose_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def check(rows: int, cols: int, rng: np.random.Generator) -> None:
    source = rng.integers(-(2**31), 2**31 - 1, size=rows * cols, dtype=np.int32, endpoint=True)
    d_out = cuda.to_device(np.full(rows * cols, -1, dtype=np.int32))
    TASK.run(cuda.to_device(source), d_out, rows, cols)
    expected = source.reshape(rows, cols).T.reshape(rows * cols)
    assert np.array_equal(d_out.copy_to_host(), expected), (rows, cols)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_shapes() -> None:
    rng = np.random.default_rng(6)
    for rows, cols in SHAPES:
        check(rows, cols, rng)


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_shapes()
    print("transpose numba: all tests passed")
