import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("coalesce_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(a, valid, b, n):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    return np.where(bits, a, b)


def check(n, rng):
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
    ref = reference(a, valid, b, n)
    dev = "cuda"
    out = torch.empty(n, dtype=torch.float64, device=dev)
    mod.run(
        torch.from_numpy(a).to(dev),
        torch.from_numpy(valid).to(dev),
        torch.from_numpy(b).to(dev),
        out,
        n,
    )
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"coalesce mismatch at n={n}"


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for n in (0, 1, 1000, 1000003):
        check(n, rng)
        print(f"coalesce triton n={n} ok")
