import triton
import triton.language as tl


@triton.jit
def kernel(values, run_ends, run_values, count, n, BLOCK: tl.constexpr):
    # Each program encodes one tile of BLOCK rows. A row is a run end when it
    # is the last row or differs from the next one. The index of a run in the
    # output is the number of run ends before it; within the tile that is a
    # cumulative sum, and across tiles it comes from count[0], which holds the
    # number of runs written so far. A tile that holds run ends finds the last
    # run end e before it, waits until the tile owning e has published
    # (count[0] = P with run_ends[P - 1] = e + 1), writes its runs at P,
    # P + 1, ..., and then adds its run count to count[0]. Tiles without run
    # ends write nothing and wait for nothing. Programs are dispatched in
    # index order, so the tile a program waits for is always resident or
    # finished.
    pid = tl.program_id(0).to(tl.int64)
    t0 = pid * BLOCK
    lane = tl.arange(0, BLOCK).to(tl.int64)
    offs = t0 + lane
    m = offs < n
    v = tl.load(values + offs, mask=m, other=0)
    nxt = tl.load(values + offs + 1, mask=offs + 1 < n, other=0)
    is_end = m & ((offs == n - 1) | (v != nxt))
    flag = is_end.to(tl.int32)
    rank = tl.cumsum(flag, axis=0) - flag
    total = tl.sum(flag, axis=0)
    if total > 0:
        e = tl.zeros((), tl.int64) - 1
        b = t0
        while (b > 0) & (e < 0):
            a = tl.maximum(b - BLOCK, 0)
            idx = a + lane
            lm = idx < b
            vv = tl.load(values + idx, mask=lm, other=0)
            nn = tl.load(values + idx + 1, mask=lm, other=0)
            e = tl.max(tl.where(lm & (vv != nn), idx, -1), axis=0)
            b = a
        base = tl.zeros((), tl.int32)
        if e >= 0:
            p = tl.atomic_add(count, 0, sem="acquire")
            last = tl.load(run_ends + (p - 1).to(tl.int64), mask=p > 0, other=0, volatile=True)
            while (p == 0) | (last != e + 1):
                p = tl.atomic_add(count, 0, sem="acquire")
                last = tl.load(run_ends + (p - 1).to(tl.int64), mask=p > 0, other=0, volatile=True)
            base = p
        j = base.to(tl.int64) + rank.to(tl.int64)
        tl.store(run_ends + j, (offs + 1).to(tl.int32), mask=is_end)
        tl.store(run_values + j, v, mask=is_end)
        tl.debug_barrier()
        tl.atomic_add(count, total, sem="release")


TYPES = {"values": "*i64", "run_ends": "*i32", "run_values": "*i64", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 4096}
NUM_WARPS = 8


def run(values, run_ends, run_values, count, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, run_ends, run_values, count, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
