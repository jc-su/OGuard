"""GPU test of the Triton run_end_decode kernel against a NumPy reference.

Run from the model directory with `python -P run_end_decode/test_triton.py`, or
with `python -m pytest --import-mode=importlib run_end_decode/test_triton.py`.
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
        "run_end_decode_triton", Path(__file__).with_name("triton.py")
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


def _run_ends(rng: np.random.Generator, n: int, layout: str) -> np.ndarray:
    """Strictly increasing run ends with a last value of n."""
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    match layout:
        case "single":
            ends = np.array([n])
        case "unit":
            ends = np.arange(1, n + 1)
        case "short":
            ends = np.cumsum(rng.integers(1, 8, n))
        case "mixed":
            # Mostly short runs with a few runs of thousands of rows.
            lengths = np.where(
                rng.random(n) < 0.01, rng.integers(1000, 5000, n), rng.integers(1, 4, n)
            )
            ends = np.cumsum(lengths)
        case _:
            raise ValueError(layout)
    ends = ends[ends < n]
    return np.append(ends, n).astype(np.int32)


def test_run_end_decode() -> None:
    rng = np.random.default_rng(3)
    for n in SIZES:
        for layout in ("single", "unit", "short", "mixed"):
            run_ends = _run_ends(rng, n, layout)
            r = run_ends.size
            run_values = rng.standard_normal(r)
            out = torch.full((n,), np.nan, dtype=torch.float64, device="cuda")
            task.run(
                torch.from_numpy(run_values).cuda(), torch.from_numpy(run_ends).cuda(), out, r, n
            )
            torch.cuda.synchronize()
            expected = np.repeat(run_values, np.diff(run_ends, prepend=0))
            np.testing.assert_array_equal(out.cpu().numpy(), expected)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_run_end_decode()
    print("run_end_decode triton: passed")
