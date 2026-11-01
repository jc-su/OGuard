"""Test the gather Triton kernel on the GPU against a NumPy reference.

Run from the parent directory: python -P gather/test_triton.py. The -P flag
keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("gather_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, m: int, rng: np.random.Generator) -> None:
    col = rng.standard_normal(n)
    sel = rng.integers(0, n, m, dtype=np.int32) if m > 0 else np.empty(0, np.int32)
    d_col = torch.from_numpy(col).cuda()
    d_sel = torch.from_numpy(sel).cuda()
    d_out = torch.full((m,), np.nan, dtype=torch.float64, device="cuda")

    IMPL.run(d_col, d_sel, d_out, m, n)
    torch.cuda.synchronize()

    np.testing.assert_array_equal(d_out.cpu().numpy(), col[sel])


def test_gather() -> None:
    rng = np.random.default_rng(2)
    for n in SIZES:
        # sel can hold row ids only when col has rows.
        ms = sorted({n, n // 2, 2 * n}) if n > 0 else [0]
        for m in ms:
            check(n, m, rng)


if __name__ == "__main__":
    test_gather()
    print(f"gather triton: pass for n in {SIZES}")
