"""Test the filter_select Triton kernel on the GPU against a NumPy reference.

Run from the parent directory: python -P filter_select/test_triton.py. The -P
flag keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
THRESHOLDS = (-1.0, 24.0, 51.0, 0.30000000000000004)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("filter_select_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, threshold: float, rng: np.random.Generator) -> None:
    quantity = rng.uniform(0.0, 50.0, n)
    quantity[rng.random(n) < 0.1] = threshold  # Rows equal to the threshold.
    # Rows next to the threshold in float64 but equal to it in float32.
    quantity[rng.random(n) < 0.05] = np.nextafter(threshold, -np.inf)
    d_quantity = torch.from_numpy(quantity).cuda()
    d_sel = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    # run() must reset count, so start from a nonzero value.
    d_count = torch.tensor([12345], dtype=torch.int32, device="cuda")

    IMPL.run(d_quantity, threshold, d_sel, d_count, n)
    torch.cuda.synchronize()

    expected = np.flatnonzero(quantity < threshold).astype(np.int32)
    count = int(d_count.item())
    sel = d_sel.cpu().numpy()
    assert count == expected.size, f"n={n}: count {count} != {expected.size}"
    np.testing.assert_array_equal(np.sort(sel[:count]), expected)
    assert np.all(sel[count:] == -1), f"n={n}: write past count"


def test_filter_select() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        for threshold in THRESHOLDS:
            check(n, threshold, rng)


if __name__ == "__main__":
    test_filter_select()
    print(f"filter_select triton: pass for n in {SIZES}")
