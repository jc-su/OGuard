"""Triton kernel for hash_build.

Inserts each row of keys into an open-addressing hash table with
linear probing. capacity is a power of two, at least 2 * n, so the
table never fills and every probe sequence is guaranteed to reach an
empty slot.

Precondition: the caller fills table_keys with -1 before each launch.

Each program handles a block of ROWS rows and probes them together:
a row that has already claimed its slot is "redirected" to a harmless
compare-and-swap on table_keys[0] with cmp == val == EMPTY, which
never changes memory (whatever table_keys[0] holds), so it cannot
disturb a still-active row's slot.
"""

import triton
import triton.language as tl

TYPES = {
    "keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "capacity": "i64",
    "n": "i64",
}

CONSTANTS = {"ROWS": 128}

NUM_WARPS = 4

EMPTY = -1


@triton.jit
def _hash64(key):
    """splitmix64 finalizer: mixes a 64-bit key for table placement."""
    h = key.to(tl.uint64)
    h = (h ^ (h >> 30)) * tl.full((), 0xBF58476D1CE4E5B9, dtype=tl.uint64)
    h = (h ^ (h >> 27)) * tl.full((), 0x94D049BB133111EB, dtype=tl.uint64)
    h = h ^ (h >> 31)
    return h


@triton.jit
def kernel(keys, table_keys, table_rows, capacity, n, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < n

    empty64 = tl.full((ROWS,), -1, dtype=tl.int64)
    key = tl.load(keys + rows, mask=row_mask, other=-1)

    cap_mask = (capacity - 1).to(tl.int64)
    slot = (_hash64(key) & cap_mask.to(tl.uint64)).to(tl.int64)

    active = row_mask
    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        use_key = tl.where(active, key, empty64)
        old = tl.atomic_cas(table_keys + use_slot, empty64, use_key)
        got = old == empty64
        write_mask = active & got
        tl.store(table_rows + use_slot, rows, mask=write_mask)
        active = active & (~got)
        slot = tl.where(active, (slot + 1) & cap_mask, slot)


def run(keys, table_keys, table_rows, capacity, n):
    """Launch kernel on PyTorch CUDA tensors. table_keys must already be -1."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](
        keys, table_keys, table_rows, capacity, n, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS
    )
