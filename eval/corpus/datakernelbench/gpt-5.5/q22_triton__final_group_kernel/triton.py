import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "cntrycode_ptr": "*i32",
    "acctbal_ptr": "*fp64",
    "bitmap_ptr": "*u8",
    "codes_ptr": "*i32",
    "avg_sum_count_ptr": "*fp64",
    "out_count_f64_ptr": "*fp64",
    "out_sum_ptr": "*fp64",
    "n_customers": "i32",
    "max_key": "i32",
}

CONSTANTS = {"N_CODES": 7, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    cntrycode_ptr,
    acctbal_ptr,
    bitmap_ptr,
    codes_ptr,
    avg_sum_count_ptr,
    out_count_f64_ptr,
    out_sum_ptr,
    n_customers,
    max_key,
    N_CODES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    avg_sum = tl.load(avg_sum_count_ptr + 0)
    avg_cnt = tl.load(avg_sum_count_ptr + 1)
    avg_val = avg_sum / avg_cnt

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    custkey = tl.load(c_custkey_ptr + offs, mask=mask, other=-1)
    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=-999999)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)

    key_valid = mask & (custkey >= 0) & (custkey <= max_key)
    has_order = tl.load(bitmap_ptr + custkey, mask=key_valid, other=1)
    base_keep = mask & key_valid & (has_order == 0) & (bal > avg_val)

    zf = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    of = tl.full((BLOCK_SIZE,), 1.0, tl.float64)

    for i in tl.static_range(0, N_CODES):
        code = tl.load(codes_ptr + i)
        keep = base_keep & (cntry == code)
        part_cnt = tl.sum(tl.where(keep, of, zf))
        part_sum = tl.sum(tl.where(keep, bal, zf))
        tl.atomic_add(out_count_f64_ptr + i, part_cnt)
        tl.atomic_add(out_sum_ptr + i, part_sum)
