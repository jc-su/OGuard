import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_end_decode_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_case(n: int, seed: int):
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32), 0
    # random increasing run ends ending at n
    max_r = min(n, 733)
    r = int(rng.integers(1, max_r + 1))
    if r == 1:
        cuts = np.array([], dtype=np.int64)
    else:
        cuts = rng.choice(np.arange(1, n), size=r - 1, replace=False)
        cuts.sort()
    run_ends = np.concatenate([cuts, [n]]).astype(np.int32)
    run_values = rng.standard_normal(r)
    return run_values, run_ends, r


def ref_decode(run_values, run_ends, n):
    out = np.zeros(n, dtype=np.float64)
    prev = 0
    for j in range(len(run_ends)):
        out[prev:run_ends[j]] = run_values[j]
        prev = run_ends[j]
    return out


def run_case(mod, n: int, seed: int):
    run_values_np, run_ends_np, r = make_case(n, seed)
    expected = ref_decode(run_values_np, run_ends_np, n)

    run_values = torch.from_numpy(run_values_np).cuda() if r > 0 else torch.zeros(1, dtype=torch.float64).cuda()
    run_ends = torch.from_numpy(run_ends_np).cuda() if r > 0 else torch.zeros(1, dtype=torch.int32).cuda()
    out = torch.zeros(max(n, 1), dtype=torch.float64).cuda()

    mod.run(run_values, run_ends, out, r, n)
    torch.cuda.synchronize()

    got = out[:n].cpu().numpy()
    assert np.array_equal(got, expected), f"n={n}: mismatch"


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 7)
        print(f"run_end_decode n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
