"""GPU test of the Triton merge_sorted kernel against a NumPy reference.

Run from the model directory with `python -P merge_sorted/test_triton.py`, or
with `python -m pytest --import-mode=importlib merge_sorted/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "merge_sorted_triton", Path(__file__).with_name("triton.py")
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


def _sorted(rng: np.random.Generator, size: int, spread: int) -> np.ndarray:
    values = rng.integers(-spread, spread, size)
    if size >= 2:
        values[0], values[-1] = INT64_MIN, INT64_MAX
    return np.sort(values)


def _check(a: np.ndarray, b: np.ndarray) -> None:
    m, p = a.size, b.size
    out = torch.full((m + p,), 7, dtype=torch.int64, device="cuda")
    task.run(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(), out, m, p)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), np.sort(np.concatenate([a, b])))


def test_merge_sorted() -> None:
    rng = np.random.default_rng(9)
    for m in SIZES:
        for p in SIZES:
            # A narrow value range gives long runs of equal keys across a and b.
            for spread in (3, 10**15):
                _check(_sorted(rng, m, spread), _sorted(rng, p, spread))


def test_disjoint_ranges() -> None:
    low = np.arange(0, 1000003, dtype=np.int64)
    high = np.arange(1000003, 2000006, dtype=np.int64)
    _check(low, high)
    _check(high, low)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_merge_sorted()
    test_disjoint_ranges()
    print("merge_sorted triton: passed")
