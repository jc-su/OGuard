"""GPU test of the Triton scatter_by_offsets kernel against a NumPy reference.

Run from the parent directory: `python -P scatter_by_offsets/test_triton.py`.
The -P flag keeps this directory off sys.path, so its triton.py and numba.py
do not shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import torch

SIZES = (0, 1, 1000, 1000003)
DENSITIES = (0.0, 0.001, 0.5, 1.0)
GUARD = 64
UNSET = -7


def load_module() -> ModuleType:
    path = Path(__file__).resolve().with_name("triton.py")
    spec = importlib.util.spec_from_file_location("scatter_by_offsets_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scatter_and_check(module: ModuleType, flags: npt.NDArray[np.int8]) -> None:
    n = flags.size
    offsets = (np.cumsum(flags, dtype=np.int32) - flags).astype(np.int32)
    total = int(flags.sum(dtype=np.int64))
    # out is the first total elements of a larger buffer, so a write past
    # total lands in the guard elements.
    d_buffer = torch.full((total + GUARD,), UNSET, dtype=torch.int32, device="cuda")
    module.run(
        torch.from_numpy(flags).cuda(),
        torch.from_numpy(offsets).cuda(),
        d_buffer[:total],
        n,
        total,
    )
    torch.cuda.synchronize()
    out = d_buffer.cpu().numpy()
    want = np.flatnonzero(flags).astype(np.int32)
    assert np.array_equal(out[:total], want), "scattered row ids differ"
    assert np.all(out[total:] == UNSET), "write past total"


def test_scatter_by_offsets() -> None:
    module = load_module()
    rng = np.random.default_rng(8)
    for n in SIZES:
        for density in DENSITIES:
            flags = (rng.random(n) < density).astype(np.int8)
            scatter_and_check(module, flags)
        print(f"scatter_by_offsets n={n}: ok")


if __name__ == "__main__":
    test_scatter_by_offsets()
    print("PASS")
