"""hash_build: insert (key, row id) into an open-addressing table with linear probing.

Home slot = mix64(key) & (capacity - 1), where mix64 is the MurmurHash3 64-bit
finalizer, so nearby keys spread over the table. Each program inserts a tile
of BLOCK rows; the probe loop runs until every row of the tile has a slot. A
row reads its slot first and issues the atomicCAS only on an empty slot; when
the CAS succeeds the row owns the slot and writes its row id. Duplicate keys,
if any, get separate slots, which is the multimap semantics a join build needs.

tl.atomic_cas has no mask, so the CAS is a predicated PTX atom.cas through
tl.inline_asm_elementwise; rows that are done or whose slot is not empty issue
nothing.

table_keys must be filled with -1 on entry.
"""

import triton
import triton.language as tl

# A one-warp tile keeps the tile-wide probe loop short: it runs as many
# rounds as the longest probe chain in the tile, and the loop's reductions
# stay inside the warp.
BLOCK = 32
NUM_WARPS = 1

TYPES = {"keys": "*i64", "table_keys": "*i64", "table_rows": "*i32", "capacity": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def mix64(key):
    h = key.to(tl.uint64)
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


@triton.jit
def cas_i64(ptr, cmp, val, pred):
    """atom.cas.b64 where pred; returns the old value, or cmp where not pred."""
    return tl.inline_asm_elementwise(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, $4, 0;\n"
        "  mov.b64 $0, $2;\n"
        "  @p atom.global.cas.b64 $0, [$1], $2, $3;\n"
        "}",
        "=l,l,l,l,r",
        [ptr, cmp, val, pred.to(tl.int32)],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def kernel(keys, table_keys, table_rows, capacity: tl.int64, n: tl.int64, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pending = rows < n
    key = tl.load(keys + rows, mask=pending, other=0)
    mask = (capacity - 1).to(tl.uint64)
    slot = (mix64(key) & mask).to(tl.int64)
    empty = tl.full([BLOCK], -1, tl.int64)
    while tl.sum(pending.to(tl.int32), axis=0) > 0:
        cur = tl.load(table_keys + slot, mask=pending, other=0, volatile=True)
        attempt = pending & (cur == -1)
        old = cas_i64(table_keys + slot, empty, key, attempt)
        won = attempt & (old == -1)
        tl.store(table_rows + slot, rows.to(tl.int32), mask=won)
        pending = pending & (won == 0)
        slot = ((slot + 1).to(tl.uint64) & mask).to(tl.int64)


def run(keys, table_keys, table_rows, capacity, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](keys, table_keys, table_rows, capacity, n, **CONSTANTS, num_warps=NUM_WARPS)
