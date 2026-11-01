import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q19_kernel(
    l_quantity_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    p_brand_id_ptr,
    p_container_grp_ptr,
    p_size_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load condition-determining fields first
    brand = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    cont = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    qty = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    size = tl.load(p_size_ptr + offs, mask=mask, other=0)
    
    # Evaluate disjoint conditions (hardcoded integer IDs match the host encoding mapping)
    cond1 = (brand == 12) & (cont == 1) & (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == 23) & (cont == 2) & (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == 34) & (cont == 3) & (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)
    
    keep = mask & (cond1 | cond2 | cond3)
    
    # Conditional load of price and discount saves HBM bandwidth for non-matching rows
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    revenue = tl.where(keep, price * (1.0 - discount), 0.0)
    
    tl.atomic_add(out_ptr, tl.sum(revenue))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_q19_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
) -> float:
    """
    SQL: 
        SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE
         (p_partkey=l_partkey AND p_brand='Brand#12' AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
          AND l_quantity BETWEEN 1 AND 11 AND p_size BETWEEN 1 AND 5
          AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
         OR (... Brand#23, MED containers, qty 10-20, size 1-10 ...)
         OR (... Brand#34, LG containers, qty 20-30, size 1-15 ...).
         
    Description: 
        TPC-H Q19 - Discounted Revenue.
        Uses exact baseline cuDF broad filters, followed by an aggressive early projection.
        Crucially, we encode string attributes to INT keys on the deeply-filtered tiny part table,
        and drop string columns BEFORE joining. This shrinks the resulting join size immensely.
        Finally, the fast inner join feeds into a fused memory-bound Triton reduction kernel
        that conditionally evaluates predicates and scales sum.
        
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # 1. Pre-filter lineitem using exact reference boolean logic
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    # Project to avoid carrying unused columns and string objects
    li = df_lineitem.loc[li_mask, ["l_partkey", "l_quantity", "l_extendedprice", "l_discount"]]

    # 2. Pre-filter part using broad brand inclusion
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part.loc[part_mask, ["p_partkey", "p_brand", "p_container", "p_size"]]

    # 3. Encode part string rules to INT32 *BEFORE* the join 
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = pt["p_brand"]
    pt["p_brand_id"] = (
        (p_brand_col == brand1).astype("int32") * brand12_id +
        (p_brand_col == brand2).astype("int32") * brand23_id +
        (p_brand_col == brand3).astype("int32") * brand34_id
    )

    sm_grp, md_grp, lg_grp = 1, 2, 3
    cont = pt["p_container"]
    pt["p_container_grp"] = (
        ((cont == "SM CASE") | (cont == "SM BOX") | (cont == "SM PACK") | (cont == "SM PKG")).astype("int32") * sm_grp +
        ((cont == "MED BAG") | (cont == "MED BOX") | (cont == "MED PKG") | (cont == "MED PACK")).astype("int32") * md_grp +
        ((cont == "LG CASE") | (cont == "LG BOX") | (cont == "LG PACK") | (cont == "LG PKG")).astype("int32") * lg_grp
    )

    # Project string columns out to speed up the merge massively
    pt = pt[["p_partkey", "p_brand_id", "p_container_grp", "p_size"]]

    # 4. Fast numerical-only inner join
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey", how="inner")

    n = len(joined)
    if n == 0:
        return 0.0

    # 5. Extract to zero-copy device tensors
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(joined["p_brand_id"].to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(joined["p_container_grp"].to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    # 6. Launch fused single-pass reduction kernel
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity_t.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._q19_kernel[grid](
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        p_brand_id_t,
        p_container_grp_t,
        p_size_t,
        out,
        n
    )

    return float(out.item())
