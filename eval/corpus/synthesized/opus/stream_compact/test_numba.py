"""Test the stream_compact Numba kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
DENSITIES: tuple[float, ...] = (0.0, 0.01, 0.5, 1.0)
SENTINEL = np.int64(0x5A5A5A5A5A5A5A5A)


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("stream_compact_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def check(n: int, density: float, rng: np.random.Generator) -> None:
    values = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64, endpoint=True)
    nonzero = rng.integers(-128, 127, size=n, dtype=np.int8, endpoint=True)
    nonzero[nonzero == 0] = 1
    keep = np.where(rng.random(n) < density, nonzero, np.int8(0)).astype(np.int8)
    d_out = cuda.to_device(np.full(n, SENTINEL, dtype=np.int64))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    TASK.run(cuda.to_device(values), cuda.to_device(keep), d_out, d_count, n)
    count = int(d_count.copy_to_host()[0])
    out = d_out.copy_to_host()
    expected = values[keep != 0]
    assert count == expected.size, (n, density, count, expected.size)
    assert np.array_equal(np.sort(out[:count]), np.sort(expected)), (n, density)
    assert np.all(out[count:] == SENTINEL), (n, density)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        for density in DENSITIES:
            check(n, density, rng)


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    print("stream_compact numba: all tests passed")
