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
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < m

    key = tl.load(probe_keys + rows, mask=row_mask, other=-1)

    cap_mask = (capacity - 1).to(tl.int64)
    slot = (_hash64(key) & cap_mask.to(tl.uint64)).to(tl.int64)

    active = row_mask
    found = tl.zeros((ROWS,), dtype=tl.int1)
    build_row = tl.zeros((ROWS,), dtype=tl.int32)

    while tl.max(active.to(tl.int32)) == 1:
        use_slot = tl.where(active, slot, 0)
        cur_key = tl.load(table_keys + use_slot)
        is_empty = cur_key == -1
        is_match = active & (~is_empty) & (cur_key == key)

        newly_found = active & is_match
        found = found | newly_found
        cur_row = tl.load(table_rows + use_slot)
        build_row = tl.where(newly_found, cur_row, build_row)

        stop = is_empty | is_match
        active = active & (~stop)
        slot = tl.where(active, (slot + 1) & cap_mask, slot)

    match_mask = row_mask & found
    count_block = out_count + tl.zeros((ROWS,), dtype=tl.int32)
    pos = tl.atomic_add(count_block, 1, mask=match_mask)
    cap_ok = match_mask & (pos < out_cap)
    tl.store(out_build + pos, build_row, mask=cap_ok)
    tl.store(out_probe + pos, rows, mask=cap_ok)


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
