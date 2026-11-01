import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("block_sort_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TILE = 1024


def reference(keys, n):
    tiles = (n + TILE - 1) // TILE
    padded = np.full(tiles * TILE, np.iinfo(np.int32).max, dtype=np.int32)
    padded[:n] = keys
    return np.sort(padded.reshape(tiles, TILE), axis=1).reshape(-1)[:n]


def check(n, rng):
    info = np.iinfo(np.int32)
    keys = rng.integers(-100, 101, n, dtype=np.int32)
    wide = rng.random(n) < 0.2
    keys[wide] = rng.integers(info.min, info.max, wide.sum(), dtype=np.int32, endpoint=True)
    if n > 2:
        keys[0] = info.max
        keys[1] = info.min
    ref = reference(keys, n)
    dev = "cuda"
    out = torch.empty(n, dtype=torch.int32, device=dev)
    mod.run(torch.from_numpy(keys).to(dev), out, n)
    assert np.array_equal(out.cpu().numpy(), ref), f"block_sort mismatch at n={n}"


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    for n in (0, 1, 1000, 1000003):
        check(n, rng)
        print(f"block_sort triton n={n} ok")
