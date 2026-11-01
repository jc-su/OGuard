"""Test q6_sum's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the result against a NumPy reference.
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "q6_sum_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATE_LO, DATE_HI = 9000, 9100
DISC_LO, DISC_HI = 0.02, 0.09
QTY_MAX = 24.0


def reference(shipdate, discount, quantity, price):
    mask = (
        (shipdate >= DATE_LO)
        & (shipdate < DATE_HI)
        & (discount >= DISC_LO)
        & (discount <= DISC_HI)
        & (quantity < QTY_MAX)
    )
    return float(np.sum(price[mask] * discount[mask]))


def check(task, n, rng):
    m = max(n, 1)
    shipdate = rng.integers(8900, 9200, size=m).astype(np.int32)
    discount = rng.uniform(0.0, 0.12, size=m).astype(np.float64)
    quantity = rng.uniform(0.0, 50.0, size=m).astype(np.float64)
    price = rng.uniform(1.0, 1000.0, size=m).astype(np.float64)

    ref = reference(shipdate[:n], discount[:n], quantity[:n], price[:n])

    d_shipdate = torch.from_numpy(shipdate).cuda()
    d_discount = torch.from_numpy(discount).cuda()
    d_quantity = torch.from_numpy(quantity).cuda()
    d_price = torch.from_numpy(price).cuda()
    d_result = torch.zeros(1, dtype=torch.float64, device="cuda")

    task.run(
        d_shipdate,
        d_discount,
        d_quantity,
        d_price,
        d_result,
        DATE_LO,
        DATE_HI,
        DISC_LO,
        DISC_HI,
        QTY_MAX,
        n,
    )
    torch.cuda.synchronize()

    got = d_result.cpu().numpy()[0]
    assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), f"n={n}: got {got} != {ref}"
    print(f"n={n}: PASS (result={got})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
