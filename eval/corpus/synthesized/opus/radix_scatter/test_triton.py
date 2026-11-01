"""GPU test of the Triton radix_scatter kernel against a NumPy reference.

Run from the model directory with `python -P radix_scatter/test_triton.py`, or
with `python -m pytest --import-mode=importlib radix_scatter/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
SHIFTS = (0, 8, 13, 24)


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "radix_scatter_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = _load_task()


def test_declared_signature_compiles() -> None:
    signature = dict(task.TYPES) | {name: "constexpr" for name in task.CONSTANTS}
    source = ASTSource(fn=task.kernel, signature=signature, constexprs=task.CONSTANTS)
    triton.compile(source, options={"num_warps": task.NUM_WARPS})


def _keys(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    match kind:
        case "uniform":
            return rng.integers(0, 2**32, n, dtype=np.uint32)
        case "constant":
            return np.full(n, 0xDEADBEEF, dtype=np.uint32)
        case "few":
            return rng.choice(np.array([0, 0xFFFFFFFF, 0x01020304, 0x80808080], dtype=np.uint32), n)
        case _:
            raise ValueError(kind)


def test_radix_scatter() -> None:
    rng = np.random.default_rng(12)
    for n in SIZES:
        for kind in ("uniform", "constant", "few"):
            keys = _keys(rng, n, kind)
            for shift in SHIFTS:
                digit = ((keys >> np.uint32(shift)) & np.uint32(255)).astype(np.int64)
                count = np.bincount(digit, minlength=256)
                start = (np.cumsum(count) - count).astype(np.int32)
                cursors = torch.from_numpy(start.copy()).cuda()
                out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
                task.run(torch.from_numpy(keys).cuda(), cursors, out, n, shift)
                torch.cuda.synchronize()
                rows = out.cpu().numpy()
                # Every row id exactly once, grouped by digit in cursor order.
                np.testing.assert_array_equal(np.sort(rows), np.arange(n, dtype=np.int32))
                np.testing.assert_array_equal(digit[rows], np.sort(digit))
                np.testing.assert_array_equal(cursors.cpu().numpy(), start + count)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_radix_scatter()
    print("radix_scatter triton: passed")
