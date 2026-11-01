import triton
import triton.language as tl

TYPES = {
    "year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    year_ptr,
    volume_ptr,
    is_brazil_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q8 market share calculation.
    
    Memory-bound optimization: single pass through data to compute
    brazil_volume and total_volume accumulators per year (1995, 1996).
    
    out_ptr layout (4 float64 accumulators):
        [0] = brazil volume for year 1995
        [1] = total volume for year 1995
        [2] = brazil volume for year 1996
        [3] = total volume for year 1996
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load input columns with safe defaults for OOB lanes
    year = tl.load(year_ptr + offs, mask=mask, other=1995)
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0)
    
    # Filter: only process years 1995 and 1996
    valid = (year == 1995) | (year == 1996)
    
    # Compute year index (0 for 1995, 1 for 1996), clamp invalid to 0
    year_idx = tl.where(valid, year - 1995, 0)
    
    # Compute contributions (zero for invalid rows)
    brazil_vol = tl.where(valid & is_brazil, volume, 0.0)
    total_vol = tl.where(valid, volume, 0.0)
    
    # Atomic add to per-year accumulators
    # Brazil: out[year_idx * 2], Total: out[year_idx * 2 + 1]
    tl.atomic_add(out_ptr + year_idx * 2, brazil_vol)
    tl.atomic_add(out_ptr + year_idx * 2 + 1, total_vol)
