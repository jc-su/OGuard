import triton
import triton.language as tl


@triton.jit
def kernel(build, probe, offsets, out_build, out_probe, b, p, BLOCK: tl.constexpr):
    # Each program writes BLOCK output pairs. The probe row that owns a pair is
    # found by a binary search over offsets, and its first match in build by a
    # binary search over build, so a probe row with many matches is balanced.
    pid = tl.program_id(0).to(tl.int64)
    q = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    total = tl.load(offsets + p).to(tl.int64)
    qmask = q < total
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.where(qmask, p, 1) + tl.zeros([BLOCK], tl.int64)
    while tl.max(hi - lo, axis=0) > 1:
        mid = (lo + hi) >> 1
        v = tl.load(offsets + mid, mask=qmask, other=0).to(tl.int64)
        take = v <= q
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)
    i = lo
    key = tl.load(probe + i, mask=qmask, other=0)
    flo = tl.zeros([BLOCK], tl.int64)
    fhi = tl.where(qmask, b, 0) + tl.zeros([BLOCK], tl.int64)
    while tl.max(fhi - flo, axis=0) > 0:
        live = flo < fhi
        mid = (flo + fhi) >> 1
        v = tl.load(build + mid, mask=live, other=0)
        less = live & (v < key)
        flo = tl.where(less, mid + 1, flo)
        fhi = tl.where(live & (v >= key), mid, fhi)
    start = tl.load(offsets + i, mask=qmask, other=0).to(tl.int64)
    tl.store(out_build + q, (flo + (q - start)).to(tl.int32), mask=qmask)
    tl.store(out_probe + q, i.to(tl.int32), mask=qmask)


TYPES = {"build": "*i64", "probe": "*i64", "offsets": "*i32", "out_build": "*i32", "out_probe": "*i32",
         "b": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(build, probe, offsets, out_build, out_probe, b, p):
    total = out_build.shape[0]
    grid = (max(1, triton.cdiv(total, CONSTANTS["BLOCK"])),)
    kernel[grid](build, probe, offsets, out_build, out_probe, b, p,
                 BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
