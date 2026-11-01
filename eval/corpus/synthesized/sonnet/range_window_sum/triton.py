import torch
import triton
import triton.language as tl

# One program does the whole column in three passes over `out` because the kernel has
# no scratch memory for a cross-program carry. With P the inclusive prefix sums of
# `values`, lo(i) the first row with ts >= ts[i] - range and h(i) the last row with
# ts == ts[i]:
#   pass 0 (ascending tiles):  out[i] = P[i]
#   pass 1 (descending tiles): out[i] = P[i] - P[lo(i) - 1]   (sum of rows lo(i)..i)
#   pass 2 (ascending tiles):  out[i] = out[h(i)]             (rows of one ts share a frame)
# Every pass reads a whole tile before it writes the tile (a barrier separates them).
# Pass 1 reads only positions below the tile or inside it, pass 2 only positions inside
# or above it, so no value that is still needed has been overwritten. The first and last
# row of a tile bracket the search of every row of the tile. The loops use while because
# the argument `range` shadows the builtin.


@triton.jit
def _lower_bound(ts, key, lo, hi, BLOCK: tl.constexpr):
    # Smallest p in [lo, hi] with ts[p] >= key, given ts[lo - 1] < key (or lo = 0) and that
    # the answer is at most hi. Each round probes BLOCK positions.
    lane = tl.arange(0, BLOCK).to(tl.int64)
    while hi > lo:
        step = (hi - lo + (BLOCK - 1)) // BLOCK
        q = lo + lane * step
        inside = q < hi
        less = inside & (tl.load(ts + q, mask=inside, other=0) < key)
        c = tl.sum(less.to(tl.int64), axis=0)
        new_lo = tl.where(c > 0, lo + (c - 1) * step + 1, lo)
        hi = tl.minimum(lo + c * step, hi)
        lo = new_lo
    return lo


@triton.jit
def _bracket(ts, key, lo, hi, m):
    # Same answer per row for keys whose answers lie in [lo, hi].
    lo_v = tl.zeros_like(key) + lo
    hi_v = tl.zeros_like(key) + hi
    active = m & (lo_v < hi_v)
    while tl.max(active.to(tl.int32), axis=0) > 0:
        mid = lo_v + (hi_v - lo_v) // 2
        x = tl.load(ts + mid, mask=active, other=0)
        lo_v = tl.where(active & (x < key), mid + 1, lo_v)
        hi_v = tl.where(active & (x >= key), mid, hi_v)
        active = m & (lo_v < hi_v)
    return lo_v


@triton.jit(do_not_specialize=["n", "range"])
def kernel(ts, values, out, n, range, BLOCK: tl.constexpr):
    # a launch passes a scalar that fits in 32 bits as int32
    n = n.to(tl.int64)
    range = range.to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    zero = n * 0
    num_tiles = (n + (BLOCK - 1)) // BLOCK

    # pass 0: inclusive prefix sums modulo 2^64
    carry = zero
    k = zero
    while k < num_tiles:
        rows = k * BLOCK + lane
        m = rows < n
        v = tl.load(values + rows, mask=m, other=0)
        running = tl.cumsum(v, axis=0) + carry
        tl.store(out + rows, running, mask=m)
        carry = tl.sum(tl.where(lane == BLOCK - 1, running, 0), axis=0)
        k += 1
    tl.debug_barrier()

    # pass 1: out[i] = sum of values[lo(i) .. i]
    k = zero
    while k < num_tiles:
        s = (num_tiles - 1 - k) * BLOCK
        e = tl.minimum(s + BLOCK, n)
        lo_first = _lower_bound(ts, tl.load(ts + s) - range, zero, s, BLOCK)
        lo_last = _lower_bound(ts, tl.load(ts + e - 1) - range, lo_first, e - 1, BLOCK)
        rows = s + lane
        m = rows < n
        key = tl.load(ts + rows, mask=m, other=0) - range
        lo = _bracket(ts, key, lo_first, lo_last, m)
        p = tl.load(out + rows, mask=m, other=0)
        below = tl.load(out + lo - 1, mask=m & (lo > 0), other=0)
        res = p - below
        tl.debug_barrier()
        tl.store(out + rows, res, mask=m)
        tl.debug_barrier()
        k += 1

    # pass 2: every row of one ts takes the value of the last row of that ts
    k = zero
    while k < num_tiles:
        s = k * BLOCK
        e = tl.minimum(s + BLOCK, n)
        up_first = _lower_bound(ts, tl.load(ts + s) + 1, s + 1, n, BLOCK)
        up_last = _lower_bound(ts, tl.load(ts + e - 1) + 1, tl.maximum(up_first, e), n, BLOCK)
        rows = s + lane
        m = rows < n
        key = tl.load(ts + rows, mask=m, other=0) + 1
        up = _bracket(ts, key, up_first, up_last, m)
        res = tl.load(out + up - 1, mask=m, other=0)
        tl.debug_barrier()
        tl.store(out + rows, res, mask=m)
        tl.debug_barrier()
        k += 1


TYPES = {
    "ts": "*i64",
    "values": "*i64",
    "out": "*i64",
    "n": "i64",
    "range": "i64",
}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(ts, values, out, n, range):
    kernel[(1,)](ts, values, out, n, range, **CONSTANTS, num_warps=NUM_WARPS)
