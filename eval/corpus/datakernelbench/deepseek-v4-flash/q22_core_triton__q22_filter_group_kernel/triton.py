import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i1",
    "o_custkey_ptr": "*i64",
    "o_custkey_n": "i32",
    "avg_val_ptr": "*fp64",
    "out_numcust_ptr": "*i64",
    "out_totacctbal_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr, acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    o_custkey_ptr, o_custkey_n,
    avg_val_ptr, out_numcust_ptr, out_totacctbal_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_custkey = tl.load(custkey_ptr + offs, mask=mask, other=-1)
    c_acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    c_cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    c_in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    avg_val = tl.load(avg_val_ptr)

    # NOT EXISTS check
    lo = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    hi = tl.full([BLOCK_SIZE], o_custkey_n, dtype=tl.int64)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        mid_val = tl.load(o_custkey_ptr + mid, mask=mid < o_custkey_n, other=-1)
        lo = tl.where(mid_val < c_custkey, mid + 1, lo)
        hi = tl.where(mid_val < c_custkey, hi, mid)
    found = (lo < o_custkey_n) & (tl.load(o_custkey_ptr + lo, mask=lo < o_custkey_n, other=-1) == c_custkey)
    no_order = ~found

    outer_mask = (c_in_mask == 1) & (c_acctbal > avg_val) & no_order & mask

    code_map = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    code_map = tl.where(c_cntrycode == 13, 0, code_map)
    code_map = tl.where(c_cntrycode == 17, 1, code_map)
    code_map = tl.where(c_cntrycode == 18, 2, code_map)
    code_map = tl.where(c_cntrycode == 23, 3, code_map)
    code_map = tl.where(c_cntrycode == 29, 4, code_map)
    code_map = tl.where(c_cntrycode == 30, 5, code_map)
    code_map = tl.where(c_cntrycode == 31, 6, code_map)

    valid = outer_mask & (code_map >= 0)

    g0_cnt = tl.sum(tl.where(valid & (code_map == 0), 1, 0))
    g1_cnt = tl.sum(tl.where(valid & (code_map == 1), 1, 0))
    g2_cnt = tl.sum(tl.where(valid & (code_map == 2), 1, 0))
    g3_cnt = tl.sum(tl.where(valid & (code_map == 3), 1, 0))
    g4_cnt = tl.sum(tl.where(valid & (code_map == 4), 1, 0))
    g5_cnt = tl.sum(tl.where(valid & (code_map == 5), 1, 0))
    g6_cnt = tl.sum(tl.where(valid & (code_map == 6), 1, 0))

    g0_sum = tl.sum(tl.where(valid & (code_map == 0), c_acctbal, 0.0))
    g1_sum = tl.sum(tl.where(valid & (code_map == 1), c_acctbal, 0.0))
    g2_sum = tl.sum(tl.where(valid & (code_map == 2), c_acctbal, 0.0))
    g3_sum = tl.sum(tl.where(valid & (code_map == 3), c_acctbal, 0.0))
    g4_sum = tl.sum(tl.where(valid & (code_map == 4), c_acctbal, 0.0))
    g5_sum = tl.sum(tl.where(valid & (code_map == 5), c_acctbal, 0.0))
    g6_sum = tl.sum(tl.where(valid & (code_map == 6), c_acctbal, 0.0))

    tl.atomic_add(out_numcust_ptr + 0, g0_cnt)
    tl.atomic_add(out_numcust_ptr + 1, g1_cnt)
    tl.atomic_add(out_numcust_ptr + 2, g2_cnt)
    tl.atomic_add(out_numcust_ptr + 3, g3_cnt)
    tl.atomic_add(out_numcust_ptr + 4, g4_cnt)
    tl.atomic_add(out_numcust_ptr + 5, g5_cnt)
    tl.atomic_add(out_numcust_ptr + 6, g6_cnt)

    tl.atomic_add(out_totacctbal_ptr + 0, g0_sum)
    tl.atomic_add(out_totacctbal_ptr + 1, g1_sum)
    tl.atomic_add(out_totacctbal_ptr + 2, g2_sum)
    tl.atomic_add(out_totacctbal_ptr + 3, g3_sum)
    tl.atomic_add(out_totacctbal_ptr + 4, g4_sum)
    tl.atomic_add(out_totacctbal_ptr + 5, g5_sum)
    tl.atomic_add(out_totacctbal_ptr + 6, g6_sum)
