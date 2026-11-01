import triton
import triton.language as tl


@triton.jit
def kernel(keys, table_keys, table_rows, n, capacity, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    row = offs.to(tl.int32)
    # -1 is not a key. Lanes that are not inserting issue a compare-and-swap of -1 to -1,
    # which changes nothing.
    key = tl.load(keys + offs, mask=mask, other=-1)
    empty = tl.full([BLOCK], -1, tl.int64)
    multiplier = tl.full([BLOCK], 0x9E3779B97F4A7C15, tl.uint64)
    slot_mask = capacity - 1
    slot = ((key.to(tl.uint64) * multiplier) >> 32).to(tl.int64) & slot_mask

    active = mask
    while tl.max(active.to(tl.int32), axis=0) > 0:
        old = tl.atomic_cas(table_keys + slot, empty, tl.where(active, key, empty))
        done = active & ((old == -1) | (old == key))
        tl.atomic_min(table_rows + slot, row, mask=done)
        active = active & (~done)
        slot = tl.where(active, (slot + 1) & slot_mask, slot)


TYPES = {"keys": "*i64", "table_keys": "*i64", "table_rows": "*i32", "n": "i64", "capacity": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(keys, table_keys, table_rows, n, capacity):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](keys, table_keys, table_rows, n, capacity, **CONSTANTS, num_warps=NUM_WARPS)
