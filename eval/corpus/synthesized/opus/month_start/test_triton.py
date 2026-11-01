"""GPU test of the Triton month_start kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
LOW = -719162  # 0001-01-01
HIGH = 2932896  # 9999-12-31
# Range ends, the epoch, and days around leap days and century years.
EDGES = np.array(
    [
        LOW,
        HIGH,
        0,
        -1,
        31,
        58,
        59,
        60,
        np.datetime64("2000-02-29", "D").astype(np.int64),
        np.datetime64("2000-03-01", "D").astype(np.int64),
        np.datetime64("1900-02-28", "D").astype(np.int64),
        np.datetime64("1900-03-01", "D").astype(np.int64),
        np.datetime64("1600-02-29", "D").astype(np.int64),
        np.datetime64("0004-02-29", "D").astype(np.int64),
        np.datetime64("1969-12-31", "D").astype(np.int64),
        np.datetime64("9999-12-01", "D").astype(np.int64),
    ],
    dtype=np.int32,
)


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("month_start_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(days: np.ndarray) -> np.ndarray:
    # NumPy datetime64 uses the proleptic Gregorian calendar.
    months = days.astype("datetime64[D]").astype("datetime64[M]")
    return months.astype("datetime64[D]").astype(np.int64).astype(np.int32)


def check(task: ModuleType, days: np.ndarray, label: str) -> None:
    n = days.size
    expected = reference(days)

    d_days = torch.from_numpy(days).cuda()
    d_out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    task.run(d_days, d_out, n)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()
    assert np.array_equal(got, expected), f"{label}: output differs"


def test_month_start() -> None:
    task = load_task()
    rng = np.random.default_rng(8)
    for n in SIZES:
        days = rng.integers(LOW, HIGH, n, dtype=np.int32, endpoint=True)
        days[: min(n, EDGES.size)] = EDGES[: min(n, EDGES.size)]
        check(task, days, f"n={n}")
        print(f"month_start n={n}: ok")
    every_day = np.arange(LOW, HIGH + 1, dtype=np.int32)
    check(task, every_day, "every day in range")
    print(f"month_start every day ({every_day.size} days): ok")


if __name__ == "__main__":
    test_month_start()
    print("PASS")
