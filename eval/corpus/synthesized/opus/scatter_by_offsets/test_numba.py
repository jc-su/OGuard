"""GPU test of the Numba scatter_by_offsets kernel against a NumPy reference.

Run from the parent directory: `python -P scatter_by_offsets/test_numba.py`.
The -P flag keeps this directory off sys.path, so its numba.py does not
shadow Numba.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

SIZES = (0, 1, 1000, 1000003)
DENSITIES = (0.0, 0.001, 0.5, 1.0)
GUARD = 64
UNSET = -7


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("numba.py")
    spec = importlib.util.spec_from_file_location("scatter_by_offsets_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scatter_and_check(
    module: ModuleType,
    flags: npt.NDArray[np.int8],
    launch: tuple[CUDADispatcher, int, int] | None,
) -> None:
    """Scatter with module.run, or with the given kernel, grid, and block size."""
    n = flags.size
    offsets = (np.cumsum(flags, dtype=np.int32) - flags).astype(np.int32)
    total = int(flags.sum(dtype=np.int64))
    d_out = cuda.to_device(np.full(total + GUARD, UNSET, dtype=np.int32))
    args = (cuda.to_device(flags), cuda.to_device(offsets), d_out, n, total)
    if launch is None:
        module.run(*args)
    else:
        kernel, grid, block = launch
        kernel[grid, block](*args)
    cuda.synchronize()
    out = d_out.copy_to_host()
    want = np.flatnonzero(flags).astype(np.int32)
    assert np.array_equal(out[:total], want), "scattered row ids differ"
    assert np.all(out[total:] == UNSET), "write past total"


def test_scatter_by_offsets() -> None:
    module = load_module()
    kernel = cuda.jit(module.kernel)
    rng = np.random.default_rng(8)
    for n in SIZES:
        for density in DENSITIES:
            flags = (rng.random(n) < density).astype(np.int8)
            scatter_and_check(module, flags, None)
            scatter_and_check(module, flags, (kernel, 3, 100))
        print(f"scatter_by_offsets n={n}: ok")


if __name__ == "__main__":
    test_scatter_by_offsets()
    print("PASS")
