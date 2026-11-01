import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("min_max_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n: int, seed: int):
    rng = np.random.default_rng(seed)
    values_np = rng.standard_normal(n) * 1000.0 if n > 0 else np.zeros(0, dtype=np.float64)

    values = torch.from_numpy(values_np).cuda()
    result = torch.tensor([float("inf"), float("-inf")], dtype=torch.float64).cuda()

    mod.run(values, result, n)
    torch.cuda.synchronize()

    got = result.cpu().numpy()
    if n == 0:
        assert got[0] == float("inf") and got[1] == float("-inf"), f"n=0: result changed: {got}"
    else:
        assert got[0] == values_np.min(), f"n={n}: min mismatch {got[0]} vs {values_np.min()}"
        assert got[1] == values_np.max(), f"n={n}: max mismatch {got[1]} vs {values_np.max()}"


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 3)
        print(f"min_max n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
