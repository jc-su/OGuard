import triton
import triton.language as tl

TYPES = {
    "code_ptr": "*i32",
    "high_ptr": "*i32",
    "out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    code_ptr,
    high_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    code = tl.load(code_ptr + offs, mask=mask, other=-1)
    high = tl.load(high_ptr + offs, mask=mask, other=0)

    is_high = high != 0
    is_code0 = mask & (code == 0)
    is_code1 = mask & (code == 1)

    one = tl.full([BLOCK_SIZE], 1, tl.int32)
    zero = tl.full([BLOCK_SIZE], 0, tl.int32)

    hi0 = tl.sum(tl.where(is_code0 & is_high, one, zero), axis=0).to(tl.int64)
    hi1 = tl.sum(tl.where(is_code1 & is_high, one, zero), axis=0).to(tl.int64)
    lo0 = tl.sum(tl.where(is_code0 & (~is_high), one, zero), axis=0).to(tl.int64)
    lo1 = tl.sum(tl.where(is_code1 & (~is_high), one, zero), axis=0).to(tl.int64)

    tl.atomic_add(out_ptr + 0, hi0)
    tl.atomic_add(out_ptr + 1, hi1)
    tl.atomic_add(out_ptr + 2, lo0)
    tl.atomic_add(out_ptr + 3, lo1)
