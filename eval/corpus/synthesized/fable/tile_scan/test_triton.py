import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("tile_scan_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TILE = 1024


def reference(values, n):
    tiles = (n + TILE - 1) // TILE
    padded = np.zeros(tiles * TILE, dtype=np.int32)
    padded[:n] = values
    grid = padded.reshape(tiles, TILE)
    exclusive = np.cumsum(grid, axis=1, dtype=np.int32) - grid
    return exclusive.reshape(-1)[:n], grid.sum(axis=1, dtype=np.int32)


def check(n, rng):
    values = rng.integers(-1000, 1001, n, dtype=np.int32)
    ref_out, ref_totals = reference(values, n)
    dev = "cuda"
    out = torch.empty(n, dtype=torch.int32, device=dev)
    totals = torch.empty((n + TILE - 1) // TILE, dtype=torch.int32, device=dev)
    mod.run(torch.from_numpy(values).to(dev), out, totals, n)
    assert np.array_equal(out.cpu().numpy(), ref_out), f"tile_scan out mismatch at n={n}"
    assert np.array_equal(totals.cpu().numpy(), ref_totals), f"tile_scan totals mismatch at n={n}"


if __name__ == "__main__":
    rng = np.random.default_rng(8)
    for n in (0, 1, 1000, 1000003):
        check(n, rng)
        print(f"tile_scan triton n={n} ok")
