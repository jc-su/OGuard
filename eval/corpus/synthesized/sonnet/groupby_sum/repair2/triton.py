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

Repair 1 (kept): rows is built in 64-bit arithmetic throughout (pid *
ROWS + arange, all as int64), so pid * ROWS cannot overflow and wrap
into a value that still passes `rows < n` while addressing outside
group/value, for any grid.

Repair 2: the checker still refused the compare-and-swap on
table_keys, because `tl.atomic_cas` has no mask argument at all, so
its address must be provably inside table_keys for *every* lane,
active or not -- and it could not prove that `slot` (and hence
`use_slot`) stays inside [0, capacity) just from `_hash32(g) &
cap_mask`: that bound holds mathematically, but it is carried through
a bitwise AND on a value produced by a custom shift/xor/multiply
mixer and then threaded through a data-dependent while loop, which is
more than the checker's range analysis follows. So the slot index is
now built so its bound is explicit and local to the instruction that
uses it, not just implied by how it was derived:

  * the modulus is computed as `h % cap` instead of `h & (cap - 1)` --
    `%` by a positive value directly yields a result in
    [0, cap), which is the bound the checker needs, whereas `&` must
    be recognized as a range-narrowing operation to give the same
    fact, which this checker apparently does not do;
  * `cap` itself is `max(capacity, 1)`, so the modulus is always a
    positive divisor even if capacity's declared lower bound (>= 2)
    were not tracked by the checker;
  * `slot` is re-clamped with tl.maximum/tl.minimum against
    [0, capacity - 1] immediately after the hash+mod, again after the
    active/inactive redirect (`use_slot`), and again after the
    linear-probe increment -- so every value that ever reaches a
    `table_keys + ...` or `table_sums + ...` address is bounded right
    at its point of use, by a min/max pair the checker can trace
    without needing to follow it back through the hash mixer or the
    loop.

table_sums' address (used by atomic_add, which does take a mask) is
built from the same clamped `use_slot`, so it is unconditionally in
bounds too, not only correctly masked.
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

    # cap_pos is a positive divisor/clamp bound even if the checker does
    # not track capacity's contractual lower bound of 2.
    cap_pos = tl.maximum(capacity, 1)
    cap_last = cap_pos - 1

    slot = (_hash32(g).to(tl.int64)) % cap_pos
    slot = tl.minimum(tl.maximum(slot, 0), cap_last)

    active = row_mask
    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        use_slot = tl.minimum(tl.maximum(use_slot, 0), cap_last)
        use_key = tl.where(active, g, empty32)
        old = tl.atomic_cas(table_keys + use_slot, empty32, use_key)
        got_slot = (old == empty32) | (old == use_key)

        add_mask = active & got_slot
        tl.atomic_add(table_sums + use_slot, tl.where(active, v, 0.0), mask=add_mask)

        active = active & (~got_slot)
        next_slot = (slot + 1) % cap_pos
        next_slot = tl.minimum(tl.maximum(next_slot, 0), cap_last)
        slot = tl.where(active, next_slot, slot)


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
