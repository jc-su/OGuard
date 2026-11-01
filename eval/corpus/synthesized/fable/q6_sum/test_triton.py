"""GPU test for q6_sum: Triton kernel against a NumPy reference."""

import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory shadows the triton package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location("q6_sum_triton", os.path.join(HERE, "triton.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(launch, reset, reps=20):
    """Average kernel time in microseconds; reset runs before every timed launch."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(reps):
        reset()
        start.record()
        launch()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    return total / reps * 1e3


def check(task, n, rng):
    date_lo, date_hi = 2000, 2365
    disc_lo, disc_hi, qty_max = 0.05, 0.07, 24.0
    shipdate = rng.integers(0, 2557, n).astype(np.int32)
    discount = (rng.integers(0, 11, n) / 100.0).astype(np.float64)
    quantity = rng.integers(1, 51, n).astype(np.float64)
    price = rng.uniform(1.0, 100000.0, n).astype(np.float64)
    mask = (
        (shipdate >= date_lo)
        & (shipdate < date_hi)
        & (discount >= disc_lo)
        & (discount <= disc_hi)
        & (quantity < qty_max)
    )
    expected = float(np.sum(price[mask] * discount[mask]))

    d_shipdate = torch.from_numpy(shipdate).to(DEV)
    d_discount = torch.from_numpy(discount).to(DEV)
    d_quantity = torch.from_numpy(quantity).to(DEV)
    d_price = torch.from_numpy(price).to(DEV)
    d_result = torch.zeros(1, dtype=torch.float64, device=DEV)
    args = (
        d_shipdate,
        d_discount,
        d_quantity,
        d_price,
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_max,
        d_result,
        n,
    )
    task.run(*args)
    torch.cuda.synchronize()

    result = float(d_result.item())
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9), (
        f"n={n}: {result} != {expected}"
    )

    if n == max(SIZES):
        us = bench(lambda: task.run(*args), lambda: d_result.zero_())
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS q6_sum")


if __name__ == "__main__":
    main()
