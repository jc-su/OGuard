"""groupby_sum: hash aggregation of value by group into an open-addressing table.

Home slot = mix64(group) & (capacity - 1) with linear probing. Each program
aggregates a tile of BLOCK rows; the probe loop runs until every row of the
tile has found its slot. A row first reads the slot: a matching key gets an
atomicAdd on the sum; an empty slot is claimed with atomicCAS (a lost race
against the same key still lands on the right slot); any other key moves to
the next slot. Keys never change once set, so the plain read before the CAS
is safe.

Triton has no block-private shared-memory table, so every row goes to the
global table; the CUDA version's block-private path only pays off below
about 1000 groups.

tl.atomic_cas has no mask, so the CAS is a predicated PTX atom.cas through
tl.inline_asm_elementwise; rows that are done or whose slot is not empty issue
nothing.

table_keys must be filled with -1 and table_sums with zero on entry.
"""

import triton
import triton.language as tl

# A small tile keeps the tile-wide probe loop short: it runs as many rounds
# as the longest probe chain in the tile.
BLOCK = 64
NUM_WARPS = 2

TYPES = {
    "group": "*i32",
    "value": "*fp64",
    "table_keys": "*i32",
    "table_sums": "*fp64",
    "capacity": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def home_slot(group, mask):
    h = group.to(tl.int64).to(tl.uint64)
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return (h & mask).to(tl.int64)


@triton.jit
def cas_i32(ptr, cmp, val, pred):
    """atom.cas.b32 where pred; returns the old value, or cmp where not pred."""
    return tl.inline_asm_elementwise(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, $4, 0;\n"
        "  mov.b32 $0, $2;\n"
        "  @p atom.global.cas.b32 $0, [$1], $2, $3;\n"
        "}",
        "=r,l,r,r,r",
        [ptr, cmp, val, pred.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def kernel(
    group, value, table_keys, table_sums, capacity: tl.int64, n: tl.int64, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pending = rows < n
    g = tl.load(group + rows, mask=pending, other=0)
    v = tl.load(value + rows, mask=pending, other=0.0)
    mask = (capacity - 1).to(tl.uint64)
    slot = home_slot(g, mask)
    empty = tl.full([BLOCK], -1, tl.int32)
    while tl.sum(pending.to(tl.int32), axis=0) > 0:
        cur = tl.load(table_keys + slot, mask=pending, other=0, volatile=True)
        hit = pending & (cur == g)
        attempt = pending & (cur == -1)
        old = cas_i32(table_keys + slot, empty, g, attempt)
        hit = hit | (attempt & ((old == -1) | (old == g)))
        tl.atomic_add(table_sums + slot, v, mask=hit, sem="relaxed")
        pending = pending & (hit == 0)
        slot = ((slot + 1).to(tl.uint64) & mask).to(tl.int64)


def run(group, value, table_keys, table_sums, capacity, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](
        group, value, table_keys, table_sums, capacity, n, **CONSTANTS, num_warps=NUM_WARPS
    )
