import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf

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
    qty_ptr, price_ptr, discount_ptr,
    brand_id_ptr, container_grp_ptr, size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns (rely on pointer dtype inference)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=mask, other=0)
    container = tl.load(container_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)

    # condition groups (exact logic as reference Torch plan)
    cond1 = (brand == brand12_id) & (container == sm_grp) & (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == brand23_id) & (container == md_grp) & (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == brand34_id) & (container == lg_grp) & (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)

    keep = mask & (cond1 | cond2 | cond3)

    revenue = price * (1.0 - discount)
    row_contrib = tl.where(keep, revenue, 0.0)
    block_sum = tl.sum(row_contrib)
    tl.atomic_add(out_ptr, block_sum)
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

def _launch_q19(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_brand_id: torch.Tensor,
    p_container_grp: torch.Tensor,
    p_size: torch.Tensor,
    brand12_id: int,
    brand23_id: int,
    brand34_id: int,
    sm_grp: int,
    md_grp: int,
    lg_grp: int,
) -> torch.Tensor:
    """
    Fuse filter → project → aggregate into a single Triton kernel.
    Returns a scalar tensor with the summed revenue.
    """
    n = l_quantity.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q19_kernel[grid](
        l_quantity, l_extendedprice, l_discount,
        p_brand_id, p_container_grp, p_size,
        brand12_id, brand23_id, brand34_id,
        sm_grp, md_grp, lg_grp,
        out, n,
    )
    return out.squeeze()

def _query_core(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_brand_id: torch.Tensor,
    p_container_grp: torch.Tensor,
    p_size: torch.Tensor,
    brand12_id: int,
    brand23_id: int,
    brand34_id: int,
    sm_grp: int,
    md_grp: int,
    lg_grp: int,
) -> torch.Tensor:
    """
    Public API (fixed signature). Delegates to the fused Triton kernel.
    """
    return _launch_q19(
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )

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
    Params:
        brand1, brand2, brand3 (str): the three brand literals.
        shipinstruct (str): ship instruction literal.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Pre-filter lineitem on shipmode/shipinstruct (common to all disjuncts)
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Pre-filter part on brand set (common narrowing)
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
    p_brand_id = cudf.Series(
        [0] * len(joined), dtype="int32"
    )
    # Use where-based encoding via masks
    brand_encoded = cudf.Series([0] * len(joined), dtype="int32")
    b12 = (p_brand_col == brand1).astype("int32") * brand12_id
    b23 = (p_brand_col == brand2).astype("int32") * brand23_id
    b34 = (p_brand_col == brand3).astype("int32") * brand34_id
    brand_encoded = b12 + b23 + b34

    # Encode container into groups: 1=SM, 2=MED, 3=LG, 0=other
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

    l_quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_brand_id_t = torch.as_tensor(brand_encoded.to_cupy(), dtype=torch.int32)
    p_container_grp_t = torch.as_tensor(container_grp.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(joined["p_size"].to_cupy(), dtype=torch.int32)

    res = _query_core(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        p_brand_id_t,
        p_container_grp_t,
        p_size_t,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
    )
    return float(res.item())
