import triton
import triton.language as tl

TYPES = {
    "year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "sum_total_ptr": "*fp64",
    "sum_brazil_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"min_year": 1995, "BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    year_ptr, volume_ptr, is_brazil_ptr,
    sum_total_ptr, sum_brazil_ptr, n_elements,
    min_year: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data
    year = tl.load(year_ptr + offs, mask=mask, other=0)
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    
    # Compute year index (0-based from min_year)
    year_idx = year - min_year
    
    # Create masks for each year
    is_1995 = (year_idx == 0) & mask
    is_1996 = (year_idx == 1) & mask
    
    # Volume masks
    vol_1995 = tl.where(is_1995, volume, 0.0)
    vol_1996 = tl.where(is_1996, volume, 0.0)
    
    # Brazil volume masks
    brazil_vol = tl.where(is_brazil, volume, 0.0)
    brazil_1995 = tl.where(is_1995, brazil_vol, 0.0)
    brazil_1996 = tl.where(is_1996, brazil_vol, 0.0)
    
    # Sum across the block using tl.sum (vectorized reduction)
    sum_1995 = tl.sum(vol_1995)
    sum_1996 = tl.sum(vol_1996)
    sum_brazil_1995 = tl.sum(brazil_1995)
    sum_brazil_1996 = tl.sum(brazil_1996)
    
    # Atomic add to global sums
    tl.atomic_add(sum_total_ptr + 0, sum_1995)
    tl.atomic_add(sum_total_ptr + 1, sum_1996)
    tl.atomic_add(sum_brazil_ptr + 0, sum_brazil_1995)
    tl.atomic_add(sum_brazil_ptr + 1, sum_brazil_1996)
