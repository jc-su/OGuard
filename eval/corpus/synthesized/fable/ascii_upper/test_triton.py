import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("ascii_upper_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(chars):
    lower = (chars >= 97) & (chars <= 122)
    return np.where(lower, chars - 32, chars).astype(np.uint8)


def check(n, rng):
    lengths = rng.integers(0, 17, n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, offsets[-1], dtype=np.uint8)
    ref = reference(chars)
    dev = "cuda"
    out = torch.empty(int(offsets[-1]), dtype=torch.uint8, device=dev)
    mod.run(torch.from_numpy(offsets).to(dev), torch.from_numpy(chars).to(dev), out, n)
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"ascii_upper mismatch at n={n}"


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    for n in (0, 1, 1000, 1000003):
        check(n, rng)
        print(f"ascii_upper triton n={n} ok")
