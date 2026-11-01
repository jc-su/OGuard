import torch
import triton
import triton.language as tl

TYPES = {
    "keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "capacity": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(keys, n, capacity, table_keys, table_rows, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    key = tl.load(keys + idx, mask=mask, other=-1)
    
    # Linear probing for each key
    hash_val = key & (capacity - 1)
    for _ in range(capacity):
        old_key = tl.atomic_cas(table_keys + hash_val, -1, key)
        if old_key == -1 or old_key == key:
            tl.store(table_rows + hash_val, tl.cast(idx, tl.int32), mask=mask)
            break
        hash_val = (hash_val + 1) & (capacity - 1)

def run(keys: torch.Tensor, table_keys: torch.Tensor, table_rows: torch.Tensor) -> None:
    n = keys.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    capacity = table_keys.numel()
    kernel[grid](keys, n, capacity, table_keys, table_rows, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
