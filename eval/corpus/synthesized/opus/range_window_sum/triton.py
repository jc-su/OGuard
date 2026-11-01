"""range_window_sum: out[i] = sum of values[j] over the rows j with
ts[i] - range <= ts[j] <= ts[i] (RANGE BETWEEN range PRECEDING AND CURRENT ROW),
modulo 2^64.

With P(q) = values[0] + ... + values[q - 1], out[i] = P(U_i) - P(L_i), where
L_i = first row with ts >= ts[i] - range and U_i = first row with ts > ts[i]. A
window can cover the whole column, and Triton has no grid-wide barrier and the
kernel has no scratch memory for prefix sums, so one program processes the column
in tiles of BLOCK rows with two cursors (position, P(position)), one for the L_i and
one for the U_i. Both are nondecreasing in i, so each cursor only moves forward,
adding BLOCK values at a time with a cumulative sum, from which each lane gathers P
at its query. For a tile, galloping searches from its first and last rows bound
L_i and U_i, and vector binary searches find them within those bounds. Every value
is read at most twice per cursor. Sums use int64 arithmetic, which wraps modulo 2^64.
"""

import triton
import triton.language as tl

BLOCK = 2048
NUM_WARPS = 8

TYPES = {
    "ts": "*i64",
    "values": "*i64",
    "out": "*i64",
    "n": "i64",
    "range": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _lower_bound_back(ts, i, x):
    """First j in [0, i] with ts[j] >= x, given ts[i] >= x."""
    hi = i
    step = tl.full((), 1, tl.int64)
    lo = hi - step
    more = (lo >= 0) & (tl.load(ts + tl.maximum(lo, 0)) >= x)
    while more:
        hi = lo
        step = step * 2
        lo = hi - step
        more = (lo >= 0) & (tl.load(ts + tl.maximum(lo, 0)) >= x)
    lo = tl.maximum(lo, -1) + 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        go = tl.load(ts + mid) < x
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
    return lo


@triton.jit
def _upper_bound_fwd(ts, i, n, x):
    """First j in (i, n] with ts[j] > x (n when none), given ts[i] <= x."""
    lo = i
    step = tl.full((), 1, tl.int64)
    hi = lo + step
    more = (hi < n) & (tl.load(ts + tl.minimum(hi, n - 1)) <= x)
    while more:
        lo = hi
        step = step * 2
        hi = lo + step
        more = (hi < n) & (tl.load(ts + tl.minimum(hi, n - 1)) <= x)
    hi = tl.minimum(hi, n)
    lo = lo + 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        go = tl.load(ts + mid) <= x
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
    return lo


@triton.jit
def _search(ts, lo, hi, x, STRICT: tl.constexpr):
    """Per lane: first k in [lo, hi) with ts[k] >= x (ts[k] > x if STRICT), or hi."""
    while tl.max(hi - lo, axis=0) > 0:
        live = lo < hi
        mid = lo + ((hi - lo) >> 1)
        v = tl.load(ts + mid, mask=live, other=0)
        if STRICT:
            go = v <= x
        else:
            go = v < x
        lo = tl.where(live & go, mid + 1, lo)
        hi = tl.where(live & ~go, mid, hi)
    return lo


@triton.jit
def _prefix_at(values, pos, val, q, q_max, BLOCK: tl.constexpr):
    """Moves the cursor (pos, val = P(pos)) to q_max and returns P(q) per lane for the
    queries q in [pos, q_max], with the new cursor."""
    lane = tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.int64) + val
    while pos < q_max:
        step = tl.minimum(q_max - pos, BLOCK)
        x = tl.load(values + pos + lane, mask=lane < step, other=0)
        incl = tl.cumsum(x, 0)
        hit = (q > pos) & (q <= pos + step)
        picked = tl.gather(incl, tl.where(hit, q - pos - 1, 0).to(tl.int32), 0)
        result = tl.where(hit, val + picked, result)
        val = val + tl.sum(x, axis=0)
        pos = pos + step
    return result, pos, val


@triton.jit
def kernel(ts, values, out, n, range, BLOCK: tl.constexpr):
    n = tl.cast(n, tl.int64)
    width = tl.cast(range, tl.int64)
    lane = tl.arange(0, BLOCK)
    l_pos = tl.zeros((), tl.int64)
    l_val = tl.zeros((), tl.int64)
    u_pos = tl.zeros((), tl.int64)
    u_val = tl.zeros((), tl.int64)
    t0 = tl.zeros((), tl.int64)
    while t0 < n:  # the argument `range` shadows the builtin
        rows = t0 + lane
        active = rows < n
        x = tl.load(ts + rows, mask=active, other=0)
        t_last = tl.minimum(t0 + BLOCK, n) - 1
        first_ts = tl.load(ts + t0)
        last_ts = tl.load(ts + t_last)
        l_first = _lower_bound_back(ts, t0, first_ts - width)
        l_last = _lower_bound_back(ts, t_last, last_ts - width)
        u_first = _upper_bound_fwd(ts, t0, n, first_ts)
        u_last = _upper_bound_fwd(ts, t_last, n, last_ts)

        lo = tl.zeros((BLOCK,), tl.int64) + l_first
        lq = _search(ts, lo, tl.where(active, l_last, lo), x - width, False)
        lo = tl.zeros((BLOCK,), tl.int64) + u_first
        uq = _search(ts, lo, tl.where(active, u_last, lo), x, True)
        lq = tl.where(active, lq, l_last)
        uq = tl.where(active, uq, u_last)

        pl, l_pos, l_val = _prefix_at(values, l_pos, l_val, lq, l_last, BLOCK)
        pu, u_pos, u_val = _prefix_at(values, u_pos, u_val, uq, u_last, BLOCK)
        tl.store(out + rows, pu - pl, mask=active)
        t0 += BLOCK


def run(ts, values, out, n, range):
    kernel[(1,)](ts, values, out, n, range, BLOCK=BLOCK, num_warps=NUM_WARPS)
