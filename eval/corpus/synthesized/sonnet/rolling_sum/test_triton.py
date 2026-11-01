import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("rolling_sum_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(values)])
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = csum[i + 1] - csum[lo]
    return out


def run_case(mod, n: int, window: int, seed: int):
    rng = np.random.default_rng(seed)
    values_np = rng.standard_normal(n) if n > 0 else np.zeros(0, dtype=np.float64)
    expected = ref_rolling_sum(values_np, window)

    values = torch.from_numpy(values_np).cuda()
    out = torch.zeros(max(n, 1), dtype=torch.float64).cuda()

    mod.run(values, out, n, window)
    torch.cuda.synchronize()

    got = out[:n].cpu().numpy()
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9), f"n={n} window={window}: mismatch, max diff {np.max(np.abs(got-expected)) if n else 0}"


def main():
    mod = _load_mod()
    cases = [(0, 1), (1, 1), (1, 5), (1000, 1), (1000, 7), (1000, 1024),
             (1000003, 1), (1000003, 256), (1000003, 1024)]
    for n, window in cases:
        run_case(mod, n, window, seed=n + window + 11)
        print(f"rolling_sum n={n} window={window}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
