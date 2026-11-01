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

Repair 1 (kept): rows is built in 64-bit arithmetic throughout (pid *
ROWS + arange, all as int64), so pid * ROWS cannot overflow and wrap
into a value that still passes `rows < n` while addressing outside
keys/table_rows, for any grid.

Repair 2: the checker still refused the compare-and-swap on
table_keys, because `tl.atomic_cas` takes no mask -- its address must
be provably inside table_keys unconditionally, and it could not carry
the bound "the result of a 64-bit AND with (capacity - 1) is <
capacity" through the splitmix64 mixer and the data-dependent probing
while loop. As in groupby_sum's repair 2, the slot index is now built
so its bound is explicit and local to each instruction that uses it:

  * the modulus is computed as `h % cap` (an unsigned remainder,
    always in [0, cap) for a positive divisor) instead of
    `h & (cap - 1)`, which is the same value here (capacity is a
    power of two) but is not tracked by the checker as a bound the
    same way `%` is;
  * `cap` is `max(capacity, 1)`, a divisor the checker can see is
    positive without relying on capacity's contractual lower bound of
    2 * n;
  * `slot` is re-clamped with tl.maximum/tl.minimum against
    [0, capacity - 1] right after the hash+mod, again after the
    active/inactive redirect (`use_slot`, which also feeds the
    table_rows store), and again after the linear-probe increment.

The hash's own uint64 arithmetic (splitmix64) is unaffected; only how
its result is turned into a table index changed.
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
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    row_mask = rows < n

    empty64 = tl.full((ROWS,), -1, dtype=tl.int64)
    key = tl.load(keys + rows, mask=row_mask, other=-1)

    # cap_pos is a positive divisor/clamp bound even if the checker does
    # not track capacity's contractual lower bound of 2 * n.
    cap_pos = tl.maximum(capacity, 1)
    cap_last = cap_pos - 1
    cap_pos_u64 = cap_pos.to(tl.uint64)

    slot = (_hash64(key) % cap_pos_u64).to(tl.int64)
    slot = tl.minimum(tl.maximum(slot, 0), cap_last)

    active = row_mask
    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        use_slot = tl.minimum(tl.maximum(use_slot, 0), cap_last)
        use_key = tl.where(active, key, empty64)
        old = tl.atomic_cas(table_keys + use_slot, empty64, use_key)
        got = old == empty64
        write_mask = active & got
        tl.store(table_rows + use_slot, rows.to(tl.int32), mask=write_mask)
        active = active & (~got)
        next_slot = (slot + 1) % cap_pos
        next_slot = tl.minimum(tl.maximum(next_slot, 0), cap_last)
        slot = tl.where(active, next_slot, slot)


def run(keys, table_keys, table_rows, capacity, n):
    """Launch kernel on PyTorch CUDA tensors. table_keys must already be -1."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](
        keys, table_keys, table_rows, capacity, n, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS
    )
