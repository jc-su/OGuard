import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
SPECIAL = [b"", b"+", b"-", b"0", b"-0", b"+0", b"00000000000000000000000000012", b"-9223372036854775808",
           b"9223372036854775807", b"9223372036854775808", b"-9223372036854775809", b"18446744073709551616",
           b"99999999999999999999", b" 12", b"12 ", b"1-2", b"+-1", b"--1", b"1e3", b"0x10", b"12.0", b"\x00",
           b"-", b"+9223372036854775807", b"-00000009223372036854775808", b"1" * 40, b"abc", b"\xff5"]
PATTERN = re.compile(rb"[+-]?[0-9]+")


def make(n, seed):
    rng = np.random.default_rng(seed)
    kinds = rng.integers(0, 6, size=n)
    picks = rng.integers(0, len(SPECIAL), size=n)
    wide = rng.integers(INT64_MIN, INT64_MAX, size=n, dtype=np.int64)
    zeros = rng.integers(0, 5, size=n)
    small = rng.integers(0, 1000000, size=n)
    mid = rng.integers(0, 1 << 40, size=n)
    big = rng.integers(0, 1 << 62, size=n)
    junk = rng.integers(32, 127, size=(n, 5)).astype(np.uint8)
    junk_len = rng.integers(0, 6, size=n)
    strings = []
    for i in range(n):
        kind = kinds[i]
        if kind == 0:
            strings.append(SPECIAL[picks[i]])
        elif kind == 1:
            strings.append(str(int(wide[i])).encode())
        elif kind == 2:
            strings.append(b"0" * int(zeros[i]) + str(int(small[i])).encode())
        elif kind == 3:
            strings.append(b"-" + b"0" * int(zeros[i] % 3) + str(int(mid[i])).encode())
        elif kind == 4:
            strings.append(junk[i, : junk_len[i]].tobytes())
        else:
            strings.append(b"+" + str(int(big[i])).encode())
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = np.frombuffer(b"".join(strings), dtype=np.uint8).copy()
    return strings, offsets, chars


def reference(strings, n):
    out = np.zeros(n, dtype=np.int64)
    bits = np.zeros(n, dtype=bool)
    for i, s in enumerate(strings):
        if PATTERN.fullmatch(s):
            v = int(s)
            if INT64_MIN <= v <= INT64_MAX:
                out[i] = v
                bits[i] = True
    return out, np.packbits(bits, bitorder="little")


def check(n, seed):
    strings, offsets, chars = make(n, seed)
    d_out = torch.full((n,), 7, dtype=torch.int64, device="cuda")
    d_valid = torch.zeros((n + 7) // 8, dtype=torch.uint8, device="cuda")
    mod.run(torch.from_numpy(offsets).cuda(), torch.from_numpy(chars).cuda(), d_out, d_valid, n)
    torch.cuda.synchronize()
    out, valid = reference(strings, n)
    got_out, got_valid = d_out.cpu().numpy(), d_valid.cpu().numpy()
    assert np.array_equal(got_out, out), f"n={n}: out mismatch at {np.nonzero(got_out != out)[0][:10]}"
    assert np.array_equal(got_valid, valid), f"n={n}: valid mismatch at bytes {np.nonzero(got_valid != valid)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("string_to_int triton: ok")
