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
    l_quantity_ptr, l_extendedprice_ptr, l_discount_ptr,
    p_brand_id_ptr, p_container_grp_ptr, p_size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    thresh1_lo, thresh1_hi,
    thresh2_lo, thresh2_hi,
    thresh3_lo, thresh3_hi,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    cont_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    size_val = tl.load(p_size_ptr + offs, mask=mask, other=0)
    quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    
    cond1 = (
        (brand_id == brand12_id)
        & (cont_grp == sm_grp)
        & (quantity >= 1.0)
        & (quantity <= 11.0)
        & (size_val >= 1)
        & (size_val <= 5)
    )
    cond2 = (
        (brand_id == brand23_id)
        & (cont_grp == md_grp)
        & (quantity >= 10.0)
        & (quantity <= 20.0)
        & (size_val >= 1)
        & (size_val <= 10)
    )
    cond3 = (
        (brand_id == brand34_id)
        & (cont_grp == lg_grp)
        & (quantity >= 20.0)
        & (quantity <= 30.0)
        & (size_val >= 1)
        & (size_val <= 15)
    )
    keep = mask & (cond1 | cond2 | cond3)
    
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    revenue = price * (1.0 - discount)
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, revenue, 0.0)))
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
    SQL: SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE
         (p_partkey=l_partkey AND p_brand='Brand#12' AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
          AND l_quantity BETWEEN 1 AND 11 AND p_size BETWEEN 1 AND 5
          AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
         OR (... Brand#23, MED containers, qty 10-20, size 1-10 ...)
         OR (... Brand#34, LG containers, qty 20-30, size 1-15 ...).
    Description: TPC-H Q19 - Discounted Revenue with multi-disjunction predicates on part/lineitem join.
    Optimized with a single fused Triton kernel that combines all predicates, projection, and sum.
    cuDF handles the join and string encoding; the fused kernel does the rest in one pass.
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Pre-filter lineitem on shipmode/shipinstruct
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Pre-filter part on brand set
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # Join on partkey
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # Encode brand as integer id
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
    brand_encoded = cudf.Series([0] * len(joined), dtype="int32")
    b12 = (p_brand_col == brand1).astype("int32") * brand12_id
    b23 = (p_brand_col == brand2).astype("int32") * brand23_id
    b34 = (p_brand_col == brand3).astype("int32") * brand34_id
    brand_encoded = b12 + b23 + b34

    # Encode container into groups: 1=SM, 2=MED, 3=LG
    sm_grp, md_grp, lg_grp = 1, 2, 3
    cont = joined["p_container"]
    sm_mask = (
        (cont == "SM CASE") | (cont == "SM BOX") | (cont == "SM PACK") | (cont == "SM PKG")
    ).astype("int32") * sm_grp
    md_mask = (
        (cont == "MED BAG") | (cont == "MED BOX") | (cont == "MED PKG") | (cont == "MED PACK")
    ).astype("int32") * md_grp
    lg_mask = (
        (cont == "LG CASE") | (cont == "LG BOX") | (cont == "LG PACK") | (cont == "LG PKG")
    ).astype("int32") * lg_grp
    container_grp = sm_mask + md_mask + lg_mask

    # Convert to tensors
    l_quantity = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    n = l_quantity.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q19_kernel[grid](
        l_quantity, l_extendedprice, l_discount,
        p_brand_id, p_container_grp, p_size,
        brand12_id, brand23_id, brand34_id,
        sm_grp, md_grp, lg_grp,
        1.0, 11.0,
        10.0, 20.0,
        20.0, 30.0,
        out, n,
    )
    return float(out.item())
