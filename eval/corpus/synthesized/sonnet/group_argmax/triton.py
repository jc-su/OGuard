import triton
import triton.language as tl


@triton.jit
def kernel(group, value, best, n, g, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs.to(tl.int32)
    k = tl.load(group + offs, mask=mask, other=0)
    v = tl.load(value + offs, mask=mask, other=0)
    slot = best + k

    # best[k] holds the row of the current maximum of group k, so value supplies its value.
    # Lanes that do not attempt a swap issue a compare-and-swap of cur to cur, which changes nothing.
    cur = tl.load(slot, mask=mask, other=-1)
    active = mask
    while tl.max(active.to(tl.int32), axis=0) > 0:
        cv = tl.load(value + tl.maximum(cur, 0), mask=active, other=0)
        cur_wins = (cur >= 0) & ((cv > v) | ((cv == v) & (cur < row)))
        attempt = active & (~cur_wins)
        old = tl.atomic_cas(slot, cur, tl.where(attempt, row, cur))
        active = attempt & (old != cur)
        cur = old


TYPES = {"group": "*i32", "value": "*i64", "best": "*i32", "n": "i64", "g": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(group, value, best, n, g):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](group, value, best, n, g, **CONSTANTS, num_warps=NUM_WARPS)
