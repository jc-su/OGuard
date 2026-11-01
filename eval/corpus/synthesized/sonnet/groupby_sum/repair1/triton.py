"""Triton kernel for groupby_sum.

Hash aggregation: sums value per group into an open-addressing table
(table_keys, table_sums) with linear probing. capacity is a power of
two, at least 2 * (number of distinct groups), so the table never
fills.

Precondition: the caller fills table_keys with -1 and zeroes
table_sums before each launch.

Each program handles a block of ROWS rows and probes them together. A
row that has already claimed (or matched) its slot is redirected to a
harmless compare-and-swap on table_keys[0] with cmp == val == EMPTY,
and its contribution to table_sums is masked out, so it cannot
disturb a still-active row's slot or sum. Two rows that race for the
same empty slot for the same group both see the winner's key (via
compare_and_swap or an equal already-installed key) and both add
their value; rows with different, colliding groups simply retry at
the next slot.

Repair: rows was computed as pid * ROWS + arange in 32-bit
arithmetic. Since the checker must prove safety for any grid, pid is
not bounded by the grid this file happens to launch, so pid * ROWS
can overflow a 32-bit index and wrap into a value that still passes
`rows < n` while addressing outside group/value. rows is now built in
64-bit arithmetic throughout, so no such wraparound is possible.
slot/use_slot were already 64-bit and are unaffected.
"""

import triton
import triton.language as tl

TYPES = {
    "group": "*i32",
    "value": "*fp64",
    "table_keys": "*i32",
    "table_sums": "*fp64",
    "capacity": "i64",
    "n": "i64",
}

CONSTANTS = {"ROWS": 128}

NUM_WARPS = 4


@triton.jit
def _hash32(key):
    """Murmur3 finalizer: mixes a 32-bit key for table placement."""
    h = key.to(tl.uint32)
    h = h ^ (h >> 16)
    h = h * tl.full((), 0x85EBCA6B, dtype=tl.uint32)
    h = h ^ (h >> 13)
    h = h * tl.full((), 0xC2B2AE35, dtype=tl.uint32)
    h = h ^ (h >> 16)
    return h


@triton.jit
def kernel(group, value, table_keys, table_sums, capacity, n, ROWS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    row_mask = rows < n

    empty32 = tl.full((ROWS,), -1, dtype=tl.int32)
    g = tl.load(group + rows, mask=row_mask, other=-1)
    v = tl.load(value + rows, mask=row_mask, other=0.0)

    cap_mask = (capacity - 1).to(tl.int32)
    slot = (_hash32(g) & cap_mask.to(tl.uint32)).to(tl.int64)
    cap_mask64 = cap_mask.to(tl.int64)

    active = row_mask
    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        use_key = tl.where(active, g, empty32)
        old = tl.atomic_cas(table_keys + use_slot, empty32, use_key)
        got_slot = (old == empty32) | (old == use_key)

        add_mask = active & got_slot
        tl.atomic_add(table_sums + use_slot, tl.where(active, v, 0.0), mask=add_mask)

        active = active & (~got_slot)
        slot = tl.where(active, (slot + 1) & cap_mask64, slot)


def run(group, value, table_keys, table_sums, capacity, n):
    """Launch kernel on PyTorch CUDA tensors.

    table_keys must already be -1 and table_sums must already be zero.
    """
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](
        group,
        value,
        table_keys,
        table_sums,
        capacity,
        n,
        ROWS=CONSTANTS["ROWS"],
        num_warps=NUM_WARPS,
    )
