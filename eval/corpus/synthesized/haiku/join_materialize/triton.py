import triton
import triton.language as tl
import torch

@triton.jit
def kernel(build_ptr, probe_ptr, offsets_ptr, out_build_ptr, out_probe_ptr, b, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < p
    
    start_idx = tl.load(offsets_ptr + offsets, mask=mask, other=0)
    end_idx = tl.load(offsets_ptr + offsets + 1, mask=mask, other=0)
    
    for k in tl.range(0, 1000):
        if k >= 1000:
            break
        out_build = tl.load(out_build_ptr + start_idx + k, mask=(k < (end_idx - start_idx)) & mask, other=-1)
        out_probe = tl.load(out_probe_ptr + start_idx + k, mask=(k < (end_idx - start_idx)) & mask, other=-1)
        tl.store(out_build_ptr + start_idx + k, start_idx + k, mask=(k < (end_idx - start_idx)) & mask)
        tl.store(out_probe_ptr + start_idx + k, offsets, mask=(k < (end_idx - start_idx)) & mask)

TYPES = {
    "build_ptr": "*i64",
    "probe_ptr": "*i64",
    "offsets_ptr": "*i32",
    "out_build_ptr": "*i32",
    "out_probe_ptr": "*i32",
    "b": "i64",
    "p": "i64"
}

CONSTANTS = {
    "BLOCK_SIZE": 256
}

NUM_WARPS = 4

def run(build, probe, offsets, out_build, out_probe, b, p):
    grid = ((p + 255) // 256,)
    kernel[grid](build, probe, offsets, out_build, out_probe, b, p, num_warps=NUM_WARPS)
