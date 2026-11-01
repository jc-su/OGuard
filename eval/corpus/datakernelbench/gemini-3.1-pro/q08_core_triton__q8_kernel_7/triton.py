import triton
import triton.language as tl

TYPES = {
    "o_year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i8",
    "sum_brazil_ptr": "*fp64",
    "sum_total_ptr": "*fp64",
    "present_ptr": "*i32",
    "min_year": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_year_ptr,
    volume_ptr,
    is_brazil_ptr,
    sum_brazil_ptr,
    sum_total_ptr,
    present_ptr,
    min_year,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    year = tl.load(o_year_ptr + offs, mask=mask, other=0)
    vol = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    
    is_br = brazil > 0
    group_idx = year - min_year
    
    valid_group_idx = tl.where(mask, group_idx, 0)
    vol_br = tl.where(is_br, vol, 0.0)
    
    tl.atomic_add(sum_brazil_ptr + valid_group_idx, vol_br, mask=mask)
    tl.atomic_add(sum_total_ptr + valid_group_idx, vol, mask=mask)
    tl.atomic_max(present_ptr + valid_group_idx, 1, mask=mask)
