import triton
import triton.language as tl

TYPES = {
    "cntrycode_ptr": "*i32",
    "acctbal_ptr": "*fp64",
    "codes_ptr": "*i32",
    "out_sum_count_ptr": "*fp64",
    "n_customers": "i32",
}

CONSTANTS = {"N_CODES": 7, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    cntrycode_ptr,
    acctbal_ptr,
    codes_ptr,
    out_sum_count_ptr,
    n_customers,
    N_CODES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=-999999)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=-1.0)

    in_set = tl.full((BLOCK_SIZE,), False, tl.int1)
    for i in tl.static_range(0, N_CODES):
        code = tl.load(codes_ptr + i)
        in_set = in_set | (cntry == code)

    keep = mask & in_set & (bal > 0.0)

    zf = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    of = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    part_sum = tl.sum(tl.where(keep, bal, zf))
    part_cnt = tl.sum(tl.where(keep, of, zf))

    tl.atomic_add(out_sum_count_ptr + 0, part_sum)
    tl.atomic_add(out_sum_count_ptr + 1, part_cnt)
