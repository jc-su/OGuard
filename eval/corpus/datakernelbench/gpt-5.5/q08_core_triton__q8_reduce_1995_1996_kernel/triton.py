import triton
import triton.language as tl

TYPES = {
    "o_year_ptr": "*i32",
    "volume_ptr": "*fp64",
    "is_brazil_ptr": "*i1",
    "agg_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_year_ptr,
    volume_ptr,
    is_brazil_ptr,
    agg_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    years = tl.load(o_year_ptr + offs, mask=mask, other=-1)
    volume = tl.load(volume_ptr + offs, mask=mask, other=0.0)
    is_brazil = tl.load(is_brazil_ptr + offs, mask=mask, other=0) != 0

    is_1995 = mask & (years == 1995)
    is_1996 = mask & (years == 1996)

    total_1995 = tl.sum(tl.where(is_1995, volume, 0.0))
    total_1996 = tl.sum(tl.where(is_1996, volume, 0.0))
    brazil_1995 = tl.sum(tl.where(is_1995 & is_brazil, volume, 0.0))
    brazil_1996 = tl.sum(tl.where(is_1996 & is_brazil, volume, 0.0))

    tl.atomic_add(agg_ptr + 0, brazil_1995, sem="relaxed")
    tl.atomic_add(agg_ptr + 1, total_1995, sem="relaxed")
    tl.atomic_add(agg_ptr + 2, brazil_1996, sem="relaxed")
    tl.atomic_add(agg_ptr + 3, total_1996, sem="relaxed")
