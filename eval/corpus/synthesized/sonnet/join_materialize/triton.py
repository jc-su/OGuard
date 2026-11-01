import triton
import triton.language as tl


@triton.jit
def kernel(build, probe, offsets, out_build, out_probe, b, p, BLOCK: tl.constexpr):
    # One lane per output position e. The probe row i is the last row with offsets[i] <= e,
    # found by binary search over offsets, and the first build row of its key by a second one.
    pid = tl.program_id(0).to(tl.int64)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(offsets + p).to(tl.int64)
    mask = e < total

    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.zeros([BLOCK], dtype=tl.int64) + p
    searching = mask & (lo < hi)
    while tl.max(searching.to(tl.int32), axis=0) > 0:
        mid = lo + ((hi - lo) >> 1)
        le = tl.load(offsets + mid, mask=searching, other=0).to(tl.int64) <= e
        lo = tl.where(searching & le, mid + 1, lo)
        hi = tl.where(searching & (~le), mid, hi)
        searching = mask & (lo < hi)
    row = tl.where(mask, lo - 1, 0)

    start = tl.load(offsets + row, mask=mask, other=0).to(tl.int64)
    key = tl.load(probe + row, mask=mask, other=0)

    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.zeros([BLOCK], dtype=tl.int64) + b
    searching = mask & (lo < hi)
    while tl.max(searching.to(tl.int32), axis=0) > 0:
        mid = lo + ((hi - lo) >> 1)
        less = tl.load(build + mid, mask=searching, other=0) < key
        lo = tl.where(searching & less, mid + 1, lo)
        hi = tl.where(searching & (~less), mid, hi)
        searching = mask & (lo < hi)

    tl.store(out_build + e, (lo + (e - start)).to(tl.int32), mask=mask)
    tl.store(out_probe + e, row.to(tl.int32), mask=mask)


TYPES = {
    "build": "*i64",
    "probe": "*i64",
    "offsets": "*i32",
    "out_build": "*i32",
    "out_probe": "*i32",
    "b": "i64",
    "p": "i64",
}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(build, probe, offsets, out_build, out_probe, b, p):
    # The number of output rows is offsets[p], which the grid must cover.
    total = int(offsets[p].item())
    grid = (max(1, triton.cdiv(total, CONSTANTS["BLOCK"])),)
    kernel[grid](build, probe, offsets, out_build, out_probe, b, p, **CONSTANTS, num_warps=NUM_WARPS)
