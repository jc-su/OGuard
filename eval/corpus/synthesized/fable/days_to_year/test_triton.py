import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the task's triton.py must not shadow the triton package
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

DAY_MIN = int((np.datetime64("0001-01-01") - np.datetime64("1970-01-01")).astype(np.int64))
DAY_MAX = int((np.datetime64("9999-12-31") - np.datetime64("1970-01-01")).astype(np.int64))


def load_task():
    spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(days_np: np.ndarray) -> np.ndarray:
    years = days_np.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    return years.astype(np.int32)


def check(task, days_np: np.ndarray, label: str) -> None:
    n = len(days_np)
    days = torch.from_numpy(days_np).cuda()
    year = torch.zeros(n, dtype=torch.int32, device="cuda")
    task.run(days, year, n)
    torch.cuda.synchronize()
    got = year.cpu().numpy()
    ref = reference(days_np)
    bad = np.flatnonzero(got != ref)
    assert len(bad) == 0, (n, label, days_np[bad[:5]], got[bad[:5]], ref[bad[:5]])
    print(f"PASS days_to_year n={n} {label}")


def year_boundaries() -> np.ndarray:
    """Dec 31 and Jan 1 around every year change, plus Feb 28/29 and Mar 1 of every year."""
    years = np.arange(1, 10000)
    jan1 = np.array([f"{y:04d}-01-01" for y in years], dtype="datetime64[D]")
    mar1 = np.array([f"{y:04d}-03-01" for y in years], dtype="datetime64[D]")
    epoch = np.datetime64("1970-01-01")
    d_jan1 = (jan1 - epoch).astype(np.int64)
    d_mar1 = (mar1 - epoch).astype(np.int64)
    allv = np.concatenate([d_jan1, d_jan1 - 1, d_mar1, d_mar1 - 1, d_mar1 - 2])
    allv = allv[(allv >= DAY_MIN) & (allv <= DAY_MAX)]
    return allv.astype(np.int32)


def main() -> None:
    rng = np.random.default_rng(8)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        check(task, rng.integers(DAY_MIN, DAY_MAX + 1, size=n, dtype=np.int32), "uniform")
        check(task, rng.integers(-30000, 30000, size=n, dtype=np.int32), "near_epoch")
    bounds = year_boundaries()
    check(task, bounds, "year_boundaries")
    check(task, np.array([DAY_MIN, DAY_MAX, 0, -1], dtype=np.int32), "limits")


if __name__ == "__main__":
    main()
