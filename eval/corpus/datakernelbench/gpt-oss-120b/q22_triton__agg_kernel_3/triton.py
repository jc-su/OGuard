import triton
import triton.language as tl

TYPES = {
    "bal_ptr": "*fp64",
    "code_ptr": "*i32",
    "mask_ptr": "*u8",
    "out_cnt": "*fp64",
    "out_sum": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 2048}
NUM_WARPS = 4


@triton.jit
def kernel(
    bal_ptr,            # *float64   c_acctbal
    code_ptr,           # *int32    cntrycode as int (e.g. 13, 31 ...)
    mask_ptr,           # *uint8    1 where row passes final filter
    out_cnt,            # *float64  per‑code count array (size = MAX_CODE+1)
    out_sum,            # *float64  per‑code sum array (size = MAX_CODE+1)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    m = tl.load(mask_ptr + offs, mask=mask, other=0)
    keep = mask & (m != 0)

    # load only for kept rows
    code = tl.load(code_ptr + offs, mask=keep, other=0)
    bal  = tl.load(bal_ptr + offs, mask=keep, other=0.0)

    # atomic updates per country code (code values < 100)
    tl.atomic_add(out_cnt + code, tl.where(keep, 1.0, 0.0))
    tl.atomic_add(out_sum + code, tl.where(keep, bal, 0.0))
