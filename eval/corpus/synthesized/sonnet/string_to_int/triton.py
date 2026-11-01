import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, valid, n, BLOCK: tl.constexpr):
    # One lane per string; the tile loops over the digit positions up to its longest string.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = i < n
    start = tl.load(offsets + i, mask=inside, other=0).to(tl.int64)
    end = tl.load(offsets + i + 1, mask=inside, other=0).to(tl.int64)
    first = tl.load(chars + start, mask=inside & (end > start), other=0).to(tl.int32)
    negative = first == 45
    digits_from = start + (negative | (first == 43)).to(tl.int64)
    count = end - digits_from
    ok = inside & (count > 0)
    ten = tl.full([BLOCK], 10, tl.uint64)
    # The magnitude of INT64_MIN is one more than INT64_MAX.
    limit = tl.full([BLOCK], 9223372036854775807, tl.uint64) + negative.to(tl.uint64)
    acc = tl.zeros([BLOCK], dtype=tl.uint64)
    for k in range(0, tl.max(tl.where(ok, count, 0), axis=0)):
        active = ok & (k < count)
        digit = tl.load(chars + digits_from + k, mask=active, other=48).to(tl.int32) - 48
        bad = (digit < 0) | (digit > 9)
        d = tl.where(bad, 0, digit).to(tl.uint64)
        overflow = acc > (limit - d) // ten
        ok = ok & ~(active & (bad | overflow))
        acc = tl.where(active & ok, acc * ten + d, acc)
    magnitude = acc.to(tl.int64)
    value = tl.where(ok, tl.where(negative, 0 - magnitude, magnitude), 0)
    tl.store(out + i, value, mask=inside)
    bit = tl.arange(0, 8).to(tl.int32)
    ok_bits = tl.reshape(ok.to(tl.int32), [BLOCK // 8, 8]) << bit[None, :]
    byte_value = tl.sum(ok_bits, axis=1).to(tl.uint8)
    byte = tl.program_id(0).to(tl.int64) * (BLOCK // 8) + tl.arange(0, BLOCK // 8).to(tl.int64)
    tl.store(valid + byte, byte_value, mask=byte < (n + 7) // 8)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*i64", "valid": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 4


def run(offsets, chars, out, valid, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, out, valid, n, **CONSTANTS, num_warps=NUM_WARPS)
