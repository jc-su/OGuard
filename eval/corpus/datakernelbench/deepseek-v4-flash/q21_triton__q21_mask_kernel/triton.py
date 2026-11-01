import triton
import triton.language as tl

TYPES = {
    "l1_orderkey_ptr": "*i64",
    "l1_late_ptr": "*i1",
    "supp_ids_ptr": "*i64",
    "lo_orderkeys_ptr": "*i64",
    "lo_distinct_supp_ptr": "*i64",
    "lo_late_distinct_supp_ptr": "*i64",
    "precomputed_idx_ptr": "*i64",
    "mask_out_ptr": "*i32",
    "n_elements": "i32",
    "num_suppliers": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 4096,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l1_orderkey_ptr, l1_late_ptr, supp_ids_ptr,
    lo_orderkeys_ptr, lo_distinct_supp_ptr, lo_late_distinct_supp_ptr,
    precomputed_idx_ptr,
    mask_out_ptr, n_elements, num_suppliers,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    l1_orderkey = tl.load(l1_orderkey_ptr + offs, mask=mask, other=-1)
    l1_late = tl.load(l1_late_ptr + offs, mask=mask, other=0)
    supp_id = tl.load(supp_ids_ptr + offs, mask=mask, other=-1)
    idx = tl.load(precomputed_idx_ptr + offs, mask=mask, other=0)
    
    distinct_supp = tl.load(lo_distinct_supp_ptr + idx, mask=mask, other=0)
    late_distinct_supp = tl.load(lo_late_distinct_supp_ptr + idx, mask=mask, other=0)
    
    exists_other = distinct_supp >= 2
    not_exists_other_late = late_distinct_supp <= 1
    
    keep = mask & (l1_late != 0) & exists_other & not_exists_other_late & (supp_id >= 0) & (supp_id < num_suppliers)
    tl.store(mask_out_ptr + offs, tl.where(keep, 1, 0), mask=mask)
