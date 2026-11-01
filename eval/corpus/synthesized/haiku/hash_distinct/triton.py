import triton
import triton.language as tl
import torch

@triton.jit
def kernel(keys_ptr, table_keys_ptr, table_rows_ptr, n, capacity, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    key = tl.load(keys_ptr + offsets, mask=mask, other=-1)
    h = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
    slot = h & (capacity - 1)
    
    for _ in tl.range(0, 1000):
        table_key = tl.load(table_keys_ptr + slot, mask=mask, other=-1)
        if table_key == -1 or table_key == key:
            break
        slot = (slot + 1) & (capacity - 1)
    
    table_key = tl.load(table_keys_ptr + slot, mask=mask, other=-1)
    if table_key == -1:
        tl.atomic_cas(table_keys_ptr + slot, -1, key)
        tl.store(table_rows_ptr + slot, offsets, mask=mask)
    else:
        tl.atomic_min(table_rows_ptr + slot, offsets, mask=mask)

TYPES = {
    "keys_ptr": "*i64",
    "table_keys_ptr": "*i64",
    "table_rows_ptr": "*i32",
    "n": "i64",
    "capacity": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(keys, table_keys, table_rows, n, capacity):
    grid = ((n + 255) // 256,)
    kernel[grid](keys, table_keys, table_rows, n, capacity, num_warps=NUM_WARPS)
