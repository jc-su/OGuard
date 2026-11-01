import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_column(n, mode, rng):
    if mode == "short":
        lens = rng.integers(0, 17, size=n)
    elif mode == "mixed":
        lens = rng.integers(0, 12, size=n)
        long_rows = rng.integers(0, max(n, 1), size=max(n // 5000, 1))
        lens[long_rows[long_rows < n]] = 3000
    elif mode == "empty":
        lens = np.zeros(n, dtype=np.int64)
    elif mode == "one_long":
        lens = np.zeros(n, dtype=np.int64)
        if n:
            lens[n // 2] = 200000
    else:
        raise ValueError(mode)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    assert offsets[n] <= 2**31 - 1
    chars = rng.integers(0, 256, size=int(offsets[n]), dtype=np.uint8)
    return offsets.astype(np.int32), chars


def reference(offsets, chars, n):
    lens = np.diff(offsets.astype(np.int64))
    h = np.full(n, 2166136261, dtype=np.uint32)
    order = np.argsort(-lens, kind="stable")
    sl = lens[order]
    hs = h[order].copy()
    start = offsets[:-1].astype(np.int64)[order]
    prime = np.uint32(16777619)
    maxlen = int(sl[0]) if n else 0
    # number of strings longer than k, for k = 0 .. maxlen - 1
    longer = np.searchsorted(-sl, -np.arange(maxlen), side="left")
    for k in range(maxlen):
        cnt = int(longer[k])
        c = chars[start[:cnt] + k].astype(np.uint32)
        hs[:cnt] = (hs[:cnt] ^ c) * prime
    h[order] = hs
    return h


def check(n, mode, rng):
    offsets, chars = make_column(n, mode, rng)
    expected = reference(offsets, chars, n)
    d_out = torch.full((n,), -559038737, dtype=torch.int32, device="cuda").view(torch.uint32)
    d_offsets = torch.from_numpy(offsets).cuda()
    d_chars = torch.from_numpy(chars).cuda()
    mod.run(d_offsets, d_chars, d_out, n)
    torch.cuda.synchronize()
    ok = np.array_equal(d_out.view(torch.int32).cpu().numpy().view(np.uint32), expected)
    print(f"n={n} mode={mode}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    ok = True
    for n in (0, 1, 1000, 1000003):
        for mode in ("short", "mixed", "empty", "one_long"):
            ok &= check(n, mode, rng)
    sys.exit(0 if ok else 1)
