import triton
import triton.language as tl

TYPES = {
    "year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "out_brazil_ptr": "*fp64",
    "out_total_ptr": "*fp64",
    "n_elements": "i32",
    "num_groups": "i32",
    "year_offset": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 8


@triton.jit
def kernel(
    year_ptr, volume_ptr, is_brazil_ptr,
    out_brazil_ptr, out_total_ptr,
    n_elements, num_groups, year_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    year = tl.load(year_ptr + offs, mask=mask, other=0)
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    
    keep = mask & (year >= 1990)
    brazil_vol = tl.where(is_brazil, volume, 0.0)
    
    # Group index: year - year_offset (e.g., 1995 -> 0, 1996 -> 1)
    gi = year - year_offset
    
    # Per-element atomic add (few groups, moderate elements)
    # Use conditional to avoid unnecessary atomic ops
    tl.atomic_add(out_brazil_ptr + gi, brazil_vol, mask=keep, sem='relaxed')
    tl.atomic_add(out_total_ptr + gi, volume, mask=keep, sem='relaxed')
