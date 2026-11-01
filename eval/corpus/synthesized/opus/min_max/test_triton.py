"""GPU test of the Triton min_max kernel against a NumPy reference.

Run from the model directory with `python -P min_max/test_triton.py`, or with
`python -m pytest --import-mode=importlib min_max/test_triton.py`.
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
        "min_max_triton", Path(__file__).with_name("triton.py")
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


def _values(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    match kind:
        case "normal":
            return rng.standard_normal(n) * 1e6
        case "negative":
            return -rng.random(n) - 1.0
        case "positive":
            return rng.random(n) + 1e-300
        case "constant":
            return np.full(n, 3.25)
        case _:
            raise ValueError(kind)


def test_min_max() -> None:
    rng = np.random.default_rng(5)
    for n in SIZES:
        for kind in ("normal", "negative", "positive", "constant"):
            values = _values(rng, n, kind)
            result = torch.tensor([np.inf, -np.inf], dtype=torch.float64, device="cuda")
            task.run(torch.from_numpy(values).cuda(), result, n)
            torch.cuda.synchronize()
            expected = [values.min(), values.max()] if n else [np.inf, -np.inf]
            np.testing.assert_array_equal(result.cpu().numpy(), expected)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_min_max()
    print("min_max triton: passed")
