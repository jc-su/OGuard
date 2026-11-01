import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("sorted_contains_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n: int, seed: int):
    rng = np.random.default_rng(seed)
    s = max(1, min(n, 733)) if n > 0 else 5
    members_np = np.sort(rng.choice(np.arange(-2_000_000, 2_000_000), size=s, replace=False)).astype(np.int64)
    if n > 0:
        # mix of members and non-members
        from_members = rng.choice(members_np, size=n // 2)
        others = rng.integers(-2_000_000, 2_000_000, size=n - n // 2).astype(np.int64)
        keys_np = np.concatenate([from_members, others])
        rng.shuffle(keys_np)
    else:
        keys_np = np.zeros(0, dtype=np.int64)

    members_set = set(members_np.tolist())
    expected = np.array([1 if k in members_set else 0 for k in keys_np], dtype=np.int8)

    keys = torch.from_numpy(keys_np).cuda()
    members = torch.from_numpy(members_np).cuda()
    match = torch.zeros(max(n, 1), dtype=torch.int8).cuda()

    mod.run(keys, members, match, n, s)
    torch.cuda.synchronize()

    got = match[:n].cpu().numpy()
    assert np.array_equal(got, expected), f"n={n}: mismatch"


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 29)
        print(f"sorted_contains n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
