import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("merge_sorted_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # stable merge, a before equal b
    out = np.empty(len(a) + len(b), dtype=np.int64)
    ia = ib = k = 0
    while ia < len(a) and ib < len(b):
        if a[ia] <= b[ib]:
            out[k] = a[ia]
            ia += 1
        else:
            out[k] = b[ib]
            ib += 1
        k += 1
    if ia < len(a):
        out[k:] = a[ia:]
    if ib < len(b):
        out[k:] = b[ib:]
    return out


def run_case(mod, m: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    lo, hi = -5_000_000, 5_000_000
    a_np = np.sort(rng.integers(lo, hi, size=m)).astype(np.int64) if m > 0 else np.zeros(0, dtype=np.int64)
    b_np = np.sort(rng.integers(lo, hi, size=p)).astype(np.int64) if p > 0 else np.zeros(0, dtype=np.int64)
    # force some exact duplicates across a and b to exercise tie-breaking
    if m > 0 and p > 0:
        k = min(5, m, p)
        b_np[:k] = a_np[:k]
        b_np.sort()

    expected = ref_merge(a_np, b_np)

    a = torch.from_numpy(a_np).cuda() if m > 0 else torch.zeros(1, dtype=torch.int64).cuda()
    b = torch.from_numpy(b_np).cuda() if p > 0 else torch.zeros(1, dtype=torch.int64).cuda()
    out = torch.zeros(max(m + p, 1), dtype=torch.int64).cuda()

    mod.run(a, b, out, m, p)
    torch.cuda.synchronize()

    got = out[: m + p].cpu().numpy()
    assert np.array_equal(got, expected), f"m={m} p={p}: mismatch"


def main():
    mod = _load_mod()
    # n in the task's test grid is m+p; split each total across two arrays.
    cases = [(0, 0), (0, 1), (1, 0), (1, 0), (0, 1000),
             (400, 600), (1000, 3), (3, 1000),
             (500000, 500003), (1000003, 0), (0, 1000003)]
    for m, p in cases:
        run_case(mod, m, p, seed=m + p + 41)
        print(f"merge_sorted m={m} p={p}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
