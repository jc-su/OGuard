import triton
import triton.language as tl


@triton.jit
def search(ts, n, key, left: tl.constexpr, BLOCK: tl.constexpr):
    # Vector binary search over ts: the first row with ts >= key when left,
    # else the first row with ts > key.
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.zeros([BLOCK], tl.int64) + n
    while tl.max(hi - lo, axis=0) > 0:
        live = lo < hi
        mid = (lo + hi) >> 1
        v = tl.load(ts + mid, mask=live, other=0)
        if left:
            go = live & (v < key)
        else:
            go = live & (v <= key)
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(live & (go == 0), mid, hi)
    return lo


@triton.jit
def kernel(ts, values, out, n, range, BLOCK: tl.constexpr):
    # Each program handles one tile of BLOCK rows. Every row's frame [lo, hi]
    # comes from two binary searches over ts. The frame is split into the
    # part inside the tile, the part before it, and the part after it; each
    # part is read in chunks of BLOCK rows whose prefix sums are gathered, so
    # a frame of any length costs one pass over it per program, and no global
    # scratch is needed. Sums are int64 and wrap on overflow.
    pid = tl.program_id(0).to(tl.int64)
    t0 = pid * BLOCK
    lane = tl.arange(0, BLOCK).to(tl.int64)
    offs = t0 + lane
    m = offs < n
    t1 = tl.minimum(t0 + BLOCK, n)
    x = tl.load(ts + offs, mask=m, other=0)
    lo = search(ts, n, x - range, True, BLOCK)
    hi = search(ts, n, x, False, BLOCK) - 1
    lo = tl.where(m, lo, offs)
    hi = tl.where(m, hi, offs - 1)
    # Part inside the tile.
    v = tl.load(values + offs, mask=m, other=0)
    c = tl.cumsum(v, axis=0)
    e = c - v
    j0 = tl.maximum(lo, t0) - t0
    j1 = tl.minimum(hi, t1 - 1) - t0
    j0 = tl.minimum(tl.maximum(j0, 0), BLOCK - 1).to(tl.int32)
    j1 = tl.minimum(tl.maximum(j1, 0), BLOCK - 1).to(tl.int32)
    acc = tl.gather(c, j1, 0) - tl.gather(e, j0, 0)
    # Part before the tile, in chunks going backward.
    first_lo = tl.min(tl.where(m, lo, 0x7FFFFFFFFFFFFFFF), axis=0)
    b = t0
    while b > first_lo:
        a = tl.maximum(b - BLOCK, first_lo)
        idx = a + lane
        w = tl.load(values + idx, mask=idx < b, other=0)
        cw = tl.cumsum(w, axis=0)
        total = tl.sum(w, axis=0)
        ew = cw - w
        inside = m & (lo >= a) & (lo < b)
        g = tl.minimum(tl.maximum(lo - a, 0), BLOCK - 1).to(tl.int32)
        part = tl.where(inside, total - tl.gather(ew, g, 0), total)
        acc = acc + tl.where(m & (lo < b), part, 0)
        b = a
    # Part after the tile, in chunks going forward.
    last_hi = tl.max(tl.where(m, hi, -1), axis=0)
    a = t1
    while a <= last_hi:
        bb = tl.minimum(a + BLOCK, last_hi + 1)
        idx = a + lane
        w = tl.load(values + idx, mask=idx < bb, other=0)
        cw = tl.cumsum(w, axis=0)
        total = tl.sum(w, axis=0)
        inside = m & (hi >= a) & (hi < bb)
        g = tl.minimum(tl.maximum(hi - a, 0), BLOCK - 1).to(tl.int32)
        part = tl.where(inside, tl.gather(cw, g, 0), total)
        acc = acc + tl.where(m & (hi >= a), part, 0)
        a = bb
    tl.store(out + offs, acc, mask=m)


TYPES = {"ts": "*i64", "values": "*i64", "out": "*i64", "n": "i64", "range": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(ts, values, out, n, range):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](ts, values, out, n, range, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
