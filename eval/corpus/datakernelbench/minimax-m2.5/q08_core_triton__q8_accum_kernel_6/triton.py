import triton
import triton.language as tl

TYPES = {
    "year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "min_year": "i32",
    "num_groups": "i32",
    "sum_brazil_ptr": "*fp64",
    "sum_total_ptr": "*fp64",
    "n_rows": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    year_ptr, volume_ptr, is_brazil_ptr,
    min_year, num_groups,
    sum_brazil_ptr, sum_total_ptr,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: accumulate total and Brazil volume per year."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    year = tl.load(year_ptr + offs, mask=mask, other=0)
    idx = year - min_year
    valid = (idx >= 0) & (idx < num_groups)
    idx_safe = tl.where(valid, idx, 0)

    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    volume_safe = tl.where(valid, volume, 0.0)

    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=False)
    brazil_vol = tl.where(is_brazil, volume, 0.0)
    brazil_vol_safe = tl.where(valid, brazil_vol, 0.0)

    tl.atomic_add(sum_total_ptr + idx_safe, volume_safe)
    tl.atomic_add(sum_brazil_ptr + idx_safe, brazil_vol_safe)
