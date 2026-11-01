import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The task directory holds triton.py, which would shadow the triton package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

MIN_DAY = int(np.datetime64("0001-01-01", "D").astype(np.int64))
MAX_DAY = int(np.datetime64("9999-12-31", "D").astype(np.int64))
MIN_MONTH = int(np.datetime64("0001-01", "M").astype(np.int64))
MAX_MONTH = int(np.datetime64("9999-12", "M").astype(np.int64))


def reference(days, months):
    d = days.astype(np.int64).astype("datetime64[D]")
    cur = d.astype("datetime64[M]")
    dom = (d - cur.astype("datetime64[D]")).astype(np.int64)
    tgt = cur + months.astype(np.int64).astype("timedelta64[M]")
    first = tgt.astype("datetime64[D]")
    dim = ((tgt + 1).astype("datetime64[D]") - first).astype(np.int64)
    return (first.astype(np.int64) + np.minimum(dom, dim - 1)).astype(np.int32)


def make_input(n, rng):
    days = rng.integers(MIN_DAY, MAX_DAY + 1, size=n, dtype=np.int64)
    # Half of the rows are the last day of their month.
    last = (days.astype("datetime64[D]").astype("datetime64[M]") + 1).astype("datetime64[D]")
    last = last.astype(np.int64) - 1
    pick = rng.random(n) < 0.5
    days = np.where(pick, last, days)
    if n >= 4:
        days[:4] = [MIN_DAY, MAX_DAY, MIN_DAY, MAX_DAY]
    cur = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    lo = np.maximum(MIN_MONTH, cur - 120000)
    hi = np.minimum(MAX_MONTH, cur + 120000)
    tgt = lo + (rng.random(n) * (hi - lo + 1)).astype(np.int64)
    months = tgt - cur
    if n >= 4:
        months[:2] = 0
        months[2] = min(120000, MAX_MONTH - cur[2])
        months[3] = max(-120000, MIN_MONTH - cur[3])
    return days.astype(np.int32), months.astype(np.int32)


def main():
    rng = np.random.default_rng(1)
    failed = False
    for n in (0, 1, 1000, 1000003):
        days, months = make_input(n, rng)
        expect = reference(days, months)
        t_days = torch.from_numpy(days).cuda()
        t_months = torch.from_numpy(months).cuda()
        t_out = torch.zeros(n, dtype=torch.int32, device="cuda")
        task.run(t_days, t_months, t_out, n)
        torch.cuda.synchronize()
        ok = np.array_equal(t_out.cpu().numpy(), expect)
        print(f"add_months triton n={n}: {'PASS' if ok else 'FAIL'}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
