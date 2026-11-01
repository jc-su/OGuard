import torch
import triton
import triton.language as tl

TYPES = {
    "group": "*i32",
    "value": "*fp64",
    "table_keys": "*i32",
    "table_sums": "*fp64",
    "capacity": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(group, value, n, capacity, table_keys, table_sums, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    
    key = tl.load(group + idx, mask=mask, other=-1)
    val = tl.load(value + idx, mask=mask, other=0.0)
    
    hash_val = key & (capacity - 1)
    
    # Linear probing
    for _ in range(capacity):
        old_key = tl.atomic_cas(table_keys + hash_val, -1, key)
        if old_key == -1 or old_key == key:
            tl.atomic_add(table_sums + hash_val, val, mask=mask)
            break
        hash_val = (hash_val + 1) & (capacity - 1)

def run(group: torch.Tensor, value: torch.Tensor, table_keys: torch.Tensor, table_sums: torch.Tensor) -> None:
    n = group.numel()
    capacity = table_keys.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](group, value, n, capacity, table_keys, table_sums, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
