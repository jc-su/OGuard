"""hash_probe: emit (build row, probe row) for every probe key found in the table.

Each program probes a tile of BLOCK rows with linear probing from
mix64(key) & (capacity - 1); a row stops at the first empty slot, so a build
side with duplicate keys emits every match. Output writes are aggregated per
program: on every probe step the program counts its matches, reserves the
slots with one atomicAdd on out_count, and each matching row writes at
base + (matches before it in the tile), from an exclusive tile scan.

The output position is formed in 64-bit from base and compared against 0 and
out_cap before the store, so every store stays inside out_build and out_probe
(out_cap slots) whatever out_count holds on entry. The store mask also keeps
the row bound rows < m, so every stored probe row id is a row of probe_keys.
Pairs past out_cap are counted but not written, so the caller can size the
output from out_count and probe again.

out_count must be zero on entry.
"""

import triton
import triton.language as tl

# A one-warp tile keeps the tile-wide probe loop short: it runs as many
# rounds as the longest probe chain in the tile, and the loop's reductions
# and scans stay inside the warp.
BLOCK = 32
NUM_WARPS = 1

TYPES = {
    "probe_keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "capacity": "i64",
    "out_build": "*i32",
    "out_probe": "*i32",
    "out_count": "*i32",
    "m": "i64",
    "out_cap": "i64",
}
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
def kernel(
    probe_keys,
    table_keys,
    table_rows,
    capacity: tl.int64,
    out_build,
    out_probe,
    out_count,
    m: tl.int64,
    out_cap: tl.int64,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = rows < m
    pending = in_bounds
    key = tl.load(probe_keys + rows, mask=pending, other=0)
    mask = (capacity - 1).to(tl.uint64)
    slot = (mix64(key) & mask).to(tl.int64)
    while tl.sum(pending.to(tl.int32), axis=0) > 0:
        tk = tl.load(table_keys + slot, mask=pending, other=-1)
        found = pending & (tk == key)
        pending = pending & (tk != -1)
        flags = found.to(tl.int32)
        num = tl.sum(flags, axis=0)
        if num > 0:
            build_row = tl.load(table_rows + slot, mask=found, other=0)
            base = tl.atomic_add(out_count, num, sem="relaxed")
            pos = base.to(tl.int64) + (tl.cumsum(flags, axis=0) - flags)
            writable = found & in_bounds & (pos >= 0) & (pos < out_cap)
            tl.store(out_build + pos, build_row, mask=writable)
            tl.store(out_probe + pos, rows.to(tl.int32), mask=writable)
        slot = ((slot + 1).to(tl.uint64) & mask).to(tl.int64)


def run(probe_keys, table_keys, table_rows, capacity, out_build, out_probe, out_count, m, out_cap):
    grid = (max(1, triton.cdiv(m, BLOCK)),)
    kernel[grid](
        probe_keys,
        table_keys,
        table_rows,
        capacity,
        out_build,
        out_probe,
        out_count,
        m,
        out_cap,
        **CONSTANTS,
        num_warps=NUM_WARPS,
    )
