"""hash_distinct: insert the distinct keys into an open-addressing table with linear
probing from slot h(key) & (capacity - 1), h(x) = (x * 0x9E3779B97F4A7C15) >> 32 on
the unsigned 64-bit value; table_rows[s] = smallest row with the key of slot s.

One lane per row; the program probes until every lane has found or inserted its
key. A slot changes once, from -1 to a key, by compare-and-swap, so a key is never
stored twice and never moves. A plain read of a slot can only be stale by showing
-1; the CAS then returns the key that is there. tl.atomic_cas has no mask, so
lanes that need no CAS issue CAS(slot, key, key), which leaves the slot unchanged.
table_rows[s] is lowered with a masked atomic min.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "n": "i64",
    "capacity": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(keys, table_keys, table_rows, n, capacity, BLOCK: tl.constexpr):
    mask = tl.cast(capacity, tl.int64) - 1
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = rows < n
    key = tl.load(keys + rows, mask=active, other=0)
    h = (key.to(tl.uint64, bitcast=True) * 0x9E3779B97F4A7C15) >> 32
    slot = h.to(tl.int64, bitcast=True) & mask
    while tl.max(active.to(tl.int32), axis=0) > 0:
        cur = tl.load(table_keys + slot, mask=active, other=0)
        need_cas = active & (cur == -1)
        old = tl.atomic_cas(table_keys + slot, tl.where(need_cas, -1, key), key)
        inserted = need_cas & (old == -1)
        seen = tl.where(need_cas, old, cur)
        found = active & (inserted | (seen == key))
        tl.atomic_min(table_rows + slot, rows.to(tl.int32), mask=found)
        active = active & ~found
        slot = tl.where(active, (slot + 1) & mask, slot)


def run(keys, table_keys, table_rows, n, capacity):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](keys, table_keys, table_rows, n, capacity, BLOCK=BLOCK, num_warps=NUM_WARPS)
