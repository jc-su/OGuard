import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("radix_scatter_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n: int, shift: int, seed: int):
    rng = np.random.default_rng(seed)
    keys_np = rng.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32) if n > 0 else np.zeros(0, dtype=np.uint32)

    bytes_of = ((keys_np.astype(np.uint64) >> shift) & 255).astype(np.int64)
    hist = np.zeros(256, dtype=np.int64)
    for b in bytes_of:
        hist[b] += 1
    exclusive = np.zeros(256, dtype=np.int32)
    exclusive[1:] = np.cumsum(hist)[:-1].astype(np.int32)

    keys = torch.from_numpy(keys_np.view(np.uint32)).cuda()
    cursors = torch.from_numpy(exclusive.copy()).cuda()
    out = torch.full((max(n, 1),), -1, dtype=torch.int32).cuda()

    mod.run(keys, cursors, out, n, shift)
    torch.cuda.synchronize()

    got_out = out[:n].cpu().numpy()
    got_cursors = cursors.cpu().numpy()

    if n > 0:
        # every row id 0..n-1 appears exactly once
        assert np.array_equal(np.sort(got_out), np.arange(n, dtype=np.int32)), f"n={n} shift={shift}: out is not a permutation of row ids"
        # rows are grouped by byte in contiguous ranges matching the histogram
        expected_end = exclusive.astype(np.int64) + hist
        for b in range(256):
            lo, hi = int(exclusive[b]), int(expected_end[b])
            if hi > lo:
                ids = got_out[lo:hi]
                assert np.all(bytes_of[ids] == b), f"n={n} shift={shift} byte={b}: wrong bucket"
        assert np.array_equal(got_cursors, expected_end.astype(np.int32)), f"n={n} shift={shift}: cursors not advanced correctly"


def main():
    mod = _load_mod()
    cases = [(0, 0), (1, 0), (1, 24), (1000, 0), (1000, 8), (1000, 24),
             (1000003, 0), (1000003, 16), (1000003, 24)]
    for n, shift in cases:
        run_case(mod, n, shift, seed=n + shift + 19)
        print(f"radix_scatter n={n} shift={shift}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
