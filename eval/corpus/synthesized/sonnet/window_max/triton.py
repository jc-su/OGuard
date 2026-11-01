import torch
import triton
import triton.language as tl


@triton.jit
def _segmented_max(s1, v1, s2, v2):
    return s2, tl.where(s1 == s2, tl.maximum(v1, v2), v2)


@triton.jit
def kernel(values, out, n, w, BLOCK: tl.constexpr, WMAX: tl.constexpr):
    # Van Herk / Gil-Werman: out[i] = max(suffix max of the block holding
    # i - w + 1, prefix max of the block holding i), blocks of w positions.
    # A tile of BLOCK + 2 * WMAX positions holds the outputs of the program in
    # its middle: WMAX positions of left context for the prefix maxima and
    # WMAX positions of right context for the suffix maxima.
    pid = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BLOCK + 2 * WMAX)
    p = pid * BLOCK - WMAX + t
    u = p - w + 1
    neg = -9223372036854775807 - 1
    xp = tl.load(values + p, mask=(p >= 0) & (p < n), other=neg)
    xu = tl.load(values + u, mask=(u >= 0) & (u < n), other=neg)
    _, pre = tl.associative_scan((((p + 4096) // w), xp), 0, _segmented_max)
    _, suf = tl.associative_scan((((u + 4096) // w), xu), 0, _segmented_max, reverse=True)
    tl.store(out + p, tl.maximum(pre, suf), mask=(t >= WMAX) & (t < WMAX + BLOCK) & (p < n))


TYPES = {"values": "*i64", "out": "*i64", "n": "i64", "w": "i64"}
CONSTANTS = {"BLOCK": 2048, "WMAX": 1024}
NUM_WARPS = 8


def run(values, out, n, w):
    assert 1 <= w <= CONSTANTS["WMAX"]
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, out, n, w, **CONSTANTS, num_warps=NUM_WARPS)
