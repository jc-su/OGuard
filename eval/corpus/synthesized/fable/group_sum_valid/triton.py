import triton
import triton.language as tl


@triton.jit
def kernel(group, value, valid, counts, sums, n, g, BLOCK: tl.constexpr):
    # Each program handles BLOCK rows and adds the valid ones to the global
    # accumulators of their groups; int64 atomic adds wrap on overflow.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    vb = tl.load(valid + (offs >> 3), mask=m, other=0).to(tl.int32)
    ok = m & (((vb >> (offs & 7).to(tl.int32)) & 1) != 0)
    k = tl.load(group + offs, mask=ok, other=0).to(tl.int64)
    v = tl.load(value + offs, mask=ok, other=0)
    tl.atomic_add(counts + k, tl.full([BLOCK], 1, tl.int64), mask=ok)
    tl.atomic_add(sums + k, v, mask=ok)


TYPES = {"group": "*i32", "value": "*i64", "valid": "*u8", "counts": "*i64", "sums": "*i64",
         "n": "i64", "g": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(group, value, valid, counts, sums, n, g):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](group, value, valid, counts, sums, n, g, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
