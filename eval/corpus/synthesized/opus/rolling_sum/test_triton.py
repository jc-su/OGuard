"""GPU test of the Triton rolling_sum kernel against a NumPy reference.

test_rolling_sum uses dyadic values (multiples of 2**-10 below 2**20 in
magnitude), for which every window sum and every prefix sum is exact in float64,
so the kernel must match the reference bit for bit. test_rounding_error uses
values of mixed magnitude and sign and checks sampled outputs against the
correctly rounded sum (math.fsum): the error must stay within the classical
bound for summing the window in any order, (window - 1) * eps * sum(|x|).

Run from the model directory with `python -P rolling_sum/test_triton.py`, or
with `python -m pytest --import-mode=importlib rolling_sum/test_triton.py`.
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
WINDOWS = (1, 2, 7, 1000, 1023, 1024)
EPS = np.finfo(np.float64).eps


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rolling_sum_triton", Path(__file__).with_name("triton.py")
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


def _run(values: np.ndarray, window: int) -> np.ndarray:
    n = values.size
    out = torch.full((n,), np.nan, dtype=torch.float64, device="cuda")
    task.run(torch.from_numpy(values).cuda(), out, n, window)
    torch.cuda.synchronize()
    return out.cpu().numpy()


def test_rolling_sum() -> None:
    rng = np.random.default_rng(6)
    for n in SIZES:
        values = rng.integers(-(2**30), 2**30, n) * 2.0**-10
        prefix = np.concatenate([[0.0], np.cumsum(values)])
        upper = np.arange(1, n + 1)
        for window in WINDOWS:
            expected = prefix[upper] - prefix[np.maximum(upper - window, 0)]
            np.testing.assert_array_equal(_run(values, window), expected)


def test_rounding_error() -> None:
    rng = np.random.default_rng(7)
    for n in (1000, 1000003):
        values = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)
        rows = np.arange(n) if n <= 1000 else rng.integers(0, n, 3000)
        for window in WINDOWS:
            got = _run(values, window)
            for i in rows:
                terms = values[max(0, i - window + 1) : i + 1]
                exact = math.fsum(terms)
                bound = (window - 1) * EPS * math.fsum(np.abs(terms)) + EPS * abs(exact) / 2
                assert abs(got[i] - exact) <= bound, (n, window, i, got[i], exact)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_rolling_sum()
    test_rounding_error()
    print("rolling_sum triton: passed")
