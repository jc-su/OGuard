"""Triton kernel for hash_probe.

Probes the open-addressing hash table built by hash_build (table_keys,
table_rows, capacity) with probe_keys, and for each match emits the
pair (build row id, probe row id) into out_build / out_probe.

Precondition: the caller zeroes out_count before each launch.

Build keys are distinct, so at most one build row matches each probe
key: the probe stops at the first match. An empty slot (table_keys ==
-1) is reached within at most capacity steps, because capacity is at
least 2 * (number of build keys), so the table is never full.

Each program handles a block of ROWS rows and probes them together: a
row that already stopped (found a match, or reached an empty slot) is
redirected to a harmless read of table_keys[0]; reads have no side
effects, so this cannot disturb a still-searching row.

Repair 1 (kept): rows is built in 64-bit arithmetic throughout (pid *
ROWS + arange, all as int64), so pid * ROWS cannot overflow and wrap
into a value that still passes `rows < m` while addressing outside
probe_keys/out_probe, for any grid.

Repair 2: the checker still refused a plain (unmasked) load of
table_keys[use_slot] -- reading the probe slot has no mask at all, so
the checker must prove `use_slot` is inside table_keys
unconditionally, and (as in hash_build's repair 2) it could not carry
that bound through splitmix64 plus a 64-bit AND against capacity - 1
threaded through the data-dependent probing while loop. The slot
index is now built the same way as hash_build's repair 2:

  * the modulus is computed as `h % cap` (always in [0, cap) for a
    positive divisor) instead of `h & (cap - 1)`;
  * `cap` is `max(capacity, 1)`, a divisor the checker can see is
    positive on its own;
  * `slot` is re-clamped against [0, capacity - 1] right after the
    hash+mod, again after the active/inactive redirect (`use_slot`,
    which now feeds both the table_keys and table_rows loads), and
    again after the linear-probe increment.

out_build/out_probe are written at `pos`, the local prefix count from
atomic_add on out_count -- a value the checker cannot bound from the
grid alone, since worst case every lane across every program could
match. `pos` is masked with `pos < out_cap` for correctness (as
before), and is now also re-clamped into [0, out_cap - 1] before it
is used to form the out_build/out_probe addresses, so those two
stores are unconditionally in bounds and not only correctly masked.
"""

import triton
import triton.language as tl

TYPES = {
    "probe_keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "out_build": "*i32",
    "out_probe": "*i32",
    "out_count": "*i32",
    "capacity": "i64",
    "m": "i64",
    "out_cap": "i64",
}

CONSTANTS = {"ROWS": 128}

NUM_WARPS = 4


@triton.jit
def _hash64(key):
    """splitmix64 finalizer: mixes a 64-bit key for table placement."""
    h = key.to(tl.uint64)
    h = (h ^ (h >> 30)) * tl.full((), 0xBF58476D1CE4E5B9, dtype=tl.uint64)
    h = (h ^ (h >> 27)) * tl.full((), 0x94D049BB133111EB, dtype=tl.uint64)
    h = h ^ (h >> 31)
    return h


@triton.jit
def kernel(
    probe_keys,
    table_keys,
    table_rows,
    out_build,
    out_probe,
    out_count,
    capacity,
    m,
    out_cap,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    row_mask = rows < m

    key = tl.load(probe_keys + rows, mask=row_mask, other=-1)

    # cap_pos is a positive divisor/clamp bound even if the checker does
    # not track capacity's contractual lower bound.
    cap_pos = tl.maximum(capacity, 1)
    cap_last = cap_pos - 1
    cap_pos_u64 = cap_pos.to(tl.uint64)

    slot = (_hash64(key) % cap_pos_u64).to(tl.int64)
    slot = tl.minimum(tl.maximum(slot, 0), cap_last)

    active = row_mask
    found = tl.zeros((ROWS,), dtype=tl.int1)
    build_row = tl.zeros((ROWS,), dtype=tl.int32)

    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        use_slot = tl.minimum(tl.maximum(use_slot, 0), cap_last)
        cur_key = tl.load(table_keys + use_slot)
        is_empty = cur_key == -1
        is_match = active & (~is_empty) & (cur_key == key)

        newly_found = active & is_match
        found = found | newly_found
        cur_row = tl.load(table_rows + use_slot)
        build_row = tl.where(newly_found, cur_row, build_row)

        stop = is_empty | is_match
        active = active & (~stop)
        next_slot = (slot + 1) % cap_pos
        next_slot = tl.minimum(tl.maximum(next_slot, 0), cap_last)
        slot = tl.where(active, next_slot, slot)

    match_mask = row_mask & found
    count_block = out_count + tl.zeros((ROWS,), dtype=tl.int32)
    pos = tl.atomic_add(count_block, 1, mask=match_mask).to(tl.int64)

    # out_cap_pos is a positive clamp bound even if the checker does not
    # track out_cap's contractual lower bound.
    out_cap_pos = tl.maximum(out_cap, 1)
    out_cap_last = out_cap_pos - 1
    cap_ok = match_mask & (pos >= 0) & (pos < out_cap)
    safe_pos = tl.minimum(tl.maximum(pos, 0), out_cap_last)
    tl.store(out_build + safe_pos, build_row, mask=cap_ok)
    tl.store(out_probe + safe_pos, rows.to(tl.int32), mask=cap_ok)


def run(
    probe_keys,
    table_keys,
    table_rows,
    out_build,
    out_probe,
    out_count,
    capacity,
    m,
    out_cap,
):
    """Launch kernel on PyTorch CUDA tensors. out_count must already be zeroed."""
    if m == 0:
        return
    grid = (triton.cdiv(m, CONSTANTS["ROWS"]),)
    kernel[grid](
        probe_keys,
        table_keys,
        table_rows,
        out_build,
        out_probe,
        out_count,
        capacity,
        m,
        out_cap,
        ROWS=CONSTANTS["ROWS"],
        num_warps=NUM_WARPS,
    )
