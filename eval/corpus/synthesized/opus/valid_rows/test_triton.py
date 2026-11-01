"""GPU test of the Triton valid_rows kernel against a NumPy reference.

Run from the model directory with `python -P valid_rows/test_triton.py`, or with
`python -m pytest --import-mode=importlib valid_rows/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "valid_rows_triton", Path(__file__).with_name("triton.py")
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


def _check(bits: np.ndarray, n: int) -> None:
    packed = np.packbits(bits, bitorder="little")
    assert packed.size == (n + 7) // 8
    valid = torch.from_numpy(packed).cuda()
    rows = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    task.run(valid, rows, count, n)
    torch.cuda.synchronize()
    expected = np.flatnonzero(bits[:n]).astype(np.int32)
    got_count = int(count.item())
    assert got_count == expected.size, (n, got_count, expected.size)
    got = rows.cpu().numpy()
    np.testing.assert_array_equal(np.sort(got[:got_count]), expected)
    assert np.all(got[got_count:] == -1)


def test_valid_rows() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        padded = (n + 7) // 8 * 8
        for density in (0.0, 0.03, 0.5, 1.0):
            bits = (rng.random(padded) < density).astype(np.uint8)
            _check(bits, n)
            # Arrow leaves the padding bits of the last byte unspecified.
            bits[n:] = 1
            _check(bits, n)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_valid_rows()
    print("valid_rows triton: passed")
