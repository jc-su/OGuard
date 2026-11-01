import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

# Triton kernel source: fused filter/project/aggregate for TPC‑H Q19
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
    l_quantity_ptr: tl.tensor,
    l_extendedprice_ptr: tl.tensor,
    l_discount_ptr: tl.tensor,
    p_brand_id_ptr: tl.tensor,
    p_container_grp_ptr: tl.tensor,
    p_size_ptr: tl.tensor,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    q1_lo,
    q1_hi,
    q2_lo,
    q2_hi,
    q3_lo,
    q3_hi,
    s1_lo,
    s1_hi,
    s2_lo,
    s2_hi,
    s3_lo,
    s3_hi,
    out_ptr: tl.tensor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements

    # Load column data
    qty = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    price = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(p_size_ptr + offs, mask=mask, other=0)

    # Disjunct 1: Brand#12, SM containers, qty 1‑11, size 1‑5
    cond1 = (brand_id == brand12_id) & (container_grp == sm_grp) & (qty >= q1_lo) & (qty <= q1_hi) & (size >= s1_lo) & (size <= s1_hi)

    # Disjunct 2: Brand#23, MED containers, qty 10‑20, size 1‑10
    cond2 = (brand_id == brand23_id) & (container_grp == md_grp) & (qty >= q2_lo) & (qty <= q2_hi) & (size >= s2_lo) & (size <= s2_hi)

    # Disjunct 3: Brand#34, LG containers, qty 20‑30, size 1‑15
    cond3 = (brand_id == brand34_id) & (container_grp == lg_grp) & (qty >= q3_lo) & (qty <= q3_hi) & (size >= s3_lo) & (size <= s3_hi)

    keep = mask & (cond1 | cond2 | cond3)

    revenue = price * (1.0 - discount)
    revenue_keep = tl.where(keep, revenue, 0.0)

    block_sum = tl.sum(revenue_keep, axis=0)
    tl.atomic_add(out_ptr, block_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
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
    Description: TPC-H Q19 - Discounted Revenue with multi‑disjunction predicates on part/lineitem join.
    This implementation uses cuDF for join and encoding, then a single fused Triton kernel for filter/project/aggregate.
    Params:
        tables (dict): Dictionary of cuDF DataFrames.
        brand1, brand2, brand3 (str): Brand literals.
        shipinstruct (str): Ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Pre‑filter lineitem on shipmode and shipinstruct
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Pre‑filter part on the three brands
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # Join lineitem and part on partkey
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # Encode brand as integer id (12, 23, 34)
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
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

    # Convert relevant columns to GPU tensors
    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    n = l_quantity_t.numel()

    # Output tensor (single value)
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity_t.device)

    # Grid launch using keyword arguments
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q19_kernel[grid](
        l_quantity_ptr=l_quantity_t,
        l_extendedprice_ptr=l_extendedprice_t,
        l_discount_ptr=l_discount_t,
        p_brand_id_ptr=p_brand_id_t,
        p_container_grp_ptr=p_container_grp_t,
        p_size_ptr=p_size_t,
        brand12_id=brand12_id,
        brand23_id=brand23_id,
        brand34_id=brand34_id,
        sm_grp=sm_grp,
        md_grp=md_grp,
        lg_grp=lg_grp,
        q1_lo=1.0,
        q1_hi=11.0,
        q2_lo=10.0,
        q2_hi=20.0,
        q3_lo=20.0,
        q3_hi=30.0,
        s1_lo=1,
        s1_hi=5,
        s2_lo=1,
        s2_hi=10,
        s3_lo=1,
        s3_hi=15,
        out_ptr=out,
        n_elements=n,
    )

    return float(out.item())
