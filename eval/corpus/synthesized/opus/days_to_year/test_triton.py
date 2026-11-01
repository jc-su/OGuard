"""GPU test of the Triton days_to_year kernel against a NumPy reference.

Run from the model directory with `python -P days_to_year/test_triton.py`, or
with `python -m pytest --import-mode=importlib days_to_year/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
FIRST_DAY = int(np.datetime64("0001-01-01", "D").astype(np.int64))
LAST_DAY = int(np.datetime64("9999-12-31", "D").astype(np.int64))


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "days_to_year_triton", Path(__file__).with_name("triton.py")
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


def _expected(days: np.ndarray) -> np.ndarray:
    years = days.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    return years.astype(np.int32)


def _check(days: np.ndarray) -> None:
    n = days.size
    year = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    task.run(torch.from_numpy(days).cuda(), year, n)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(year.cpu().numpy(), _expected(days))


def test_days_to_year() -> None:
    rng = np.random.default_rng(8)
    for n in SIZES:
        _check(rng.integers(FIRST_DAY, LAST_DAY + 1, n, dtype=np.int32))


def test_every_day_of_the_calendar() -> None:
    # 3.65 million days: every year boundary and leap day from 0001 to 9999.
    _check(np.arange(FIRST_DAY, LAST_DAY + 1, dtype=np.int32))


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_days_to_year()
    test_every_day_of_the_calendar()
    print("days_to_year triton: passed")
