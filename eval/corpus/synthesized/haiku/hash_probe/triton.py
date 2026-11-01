import torch
import triton
import triton.language as tl

TYPES = {
    "probe_keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "out_build": "*i32",
    "out_probe": "*i32",
    "out_count": "*i32",
    "m": "i64",
    "capacity": "i64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(probe_keys, m, table_keys, table_rows, capacity, out_build, out_probe, out_count, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < m
    
    probe_key = tl.load(probe_keys + idx, mask=mask, other=-1)
    
    # Linear probing
    hash_val = probe_key & (capacity - 1)
    for _ in range(capacity):
        key = tl.load(table_keys + hash_val, mask=mask, other=-1)
        if key == probe_key:
            build_row = tl.load(table_rows + hash_val, mask=mask, other=-1)
            count_idx = tl.atomic_add(out_count, 0, 1)
            if count_idx < capacity:
                tl.store(out_build + count_idx, build_row, mask=mask)
                tl.store(out_probe + count_idx, tl.cast(idx, tl.int32), mask=mask)
            break
        elif key == -1:
            break
        hash_val = (hash_val + 1) & (capacity - 1)

def run(probe_keys: torch.Tensor, table_keys: torch.Tensor, table_rows: torch.Tensor,
        out_build: torch.Tensor, out_probe: torch.Tensor, out_count: torch.Tensor) -> None:
    m = probe_keys.numel()
    capacity = table_keys.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((m + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](probe_keys, m, table_keys, table_rows, capacity, out_build, out_probe, out_count, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
