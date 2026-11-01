import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, valid, n, BLOCK: tl.constexpr):
    # Each program handles BLOCK bytes of the validity bitmap, that is a
    # [BLOCK, 8] grid of strings, one lane per string, and writes each byte
    # once, so no atomics on the bitmap are needed. The lanes step over the
    # bytes together until the longest string of the program is consumed.
    # The magnitude accumulates in unsigned arithmetic with the overflow test
    # done before each multiply-add, so that -9223372036854775808 parses and
    # 9223372036854775808 does not.
    pid = tl.program_id(0).to(tl.int64)
    bytes_ = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nbytes = (n + 7) // 8
    bmask = bytes_ < nbytes
    bit = tl.arange(0, 8).to(tl.int64)
    rows = bytes_[:, None] * 8 + bit[None, :]
    rmask = rows < n
    pos = tl.load(offsets + rows, mask=rmask, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=rmask, other=0).to(tl.int64)
    c0 = tl.load(chars + pos, mask=rmask & (pos < end), other=0).to(tl.int32)
    neg = c0 == 45
    pos = pos + ((c0 == 43) | neg).to(tl.int64)
    ok = rmask & (pos < end)
    limit = tl.where(neg, tl.full([BLOCK, 8], 0x8000000000000000, tl.uint64),
                     tl.full([BLOCK, 8], 0x7FFFFFFFFFFFFFFF, tl.uint64))
    mag = tl.zeros([BLOCK, 8], tl.uint64)
    live = ok
    while tl.max(tl.where(live, end - pos, 0)) > 0:
        c = tl.load(chars + pos, mask=live, other=48).to(tl.int32)
        d = (c - 48).to(tl.uint64)
        bad = live & ((c < 48) | (c > 57) | (mag > (limit - d) // 10))
        ok = ok & (bad == 0)
        mag = tl.where(live & (bad == 0), mag * 10 + d, mag)
        pos = pos + live.to(tl.int64)
        live = ok & (pos < end)
    value = tl.where(neg, 0 - mag, mag).to(tl.int64, bitcast=True)
    tl.store(out + rows, tl.where(ok, value, 0), mask=rmask)
    weight = (1 << bit).to(tl.int32)
    acc = tl.sum(tl.where(ok, weight[None, :], 0), axis=1)
    tl.store(valid + bytes_, acc.to(tl.uint8), mask=bmask)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*i64", "valid": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 128}
NUM_WARPS = 4


def run(offsets, chars, out, valid, n):
    nbytes = (n + 7) // 8
    grid = (max(1, triton.cdiv(nbytes, CONSTANTS["BLOCK"])),)
    kernel[grid](offsets, chars, out, valid, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
