"""GPU test of the Triton dictionary_decode kernel against a NumPy reference.

Run from the model directory with `python -P dictionary_decode/test_triton.py`,
or with `python -m pytest --import-mode=importlib dictionary_decode/test_triton.py`.
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
        "dictionary_decode_triton", Path(__file__).with_name("triton.py")
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


def test_dictionary_decode() -> None:
    rng = np.random.default_rng(2)
    for n in SIZES:
        for d in (1, 257, 100_000):
            codes = rng.integers(0, d, n, dtype=np.int32)
            dictionary = rng.standard_normal(d)
            out = torch.full((n,), np.nan, dtype=torch.float64, device="cuda")
            task.run(torch.from_numpy(codes).cuda(), torch.from_numpy(dictionary).cuda(), out, n, d)
            torch.cuda.synchronize()
            np.testing.assert_array_equal(out.cpu().numpy(), dictionary[codes])


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_dictionary_decode()
    print("dictionary_decode triton: passed")
