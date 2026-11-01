import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# --------------------------------------------------------------
# Triton kernel source – written to a temporary file and loaded.
# --------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _revenue_kernel(
    # column pointers
    l_quantity_ptr,          # fp64
    l_extendedprice_ptr,    # fp64
    l_discount_ptr,         # fp64
    p_brand_id_ptr,         # int32
    p_container_grp_ptr,    # int32
    p_size_ptr,             # int32
    # scalar ids
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    # output per‑block array
    out_ptr,                # fp64[]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (masked)
    l_q   = tl.load(l_quantity_ptr + offs,        mask=mask, other=0.0)
    l_ep  = tl.load(l_extendedprice_ptr + offs,  mask=mask, other=0.0)
    l_d   = tl.load(l_discount_ptr + offs,       mask=mask, other=0.0)
    b_id  = tl.load(p_brand_id_ptr + offs,       mask=mask, other=0)
    grp   = tl.load(p_container_grp_ptr + offs,  mask=mask, other=0)
    sz    = tl.load(p_size_ptr + offs,           mask=mask, other=0)

    # Three disjunctive predicates
    cond1 = (b_id == brand12_id) & (grp == sm_grp) & (l_q >= 1.0) & (l_q <= 11.0) & (sz >= 1) & (sz <= 5)
    cond2 = (b_id == brand23_id) & (grp == md_grp) & (l_q >= 10.0) & (l_q <= 20.0) & (sz >= 1) & (sz <= 10)
    cond3 = (b_id == brand34_id) & (grp == lg_grp) & (l_q >= 20.0) & (l_q <= 30.0) & (sz >= 1) & (sz <= 15)

    keep = cond1 | cond2 | cond3

    # Revenue for kept rows
    revenue = tl.where(keep, l_ep * (1.0 - l_d), 0.0)

    # Reduce within the block and write to per‑block output
    block_sum = tl.sum(revenue)
    tl.store(out_ptr + pid, block_sum)
'''

def _load_kernels():
    """Write the Triton source to a temp file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_q19_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels at module import
_K = _load_kernels()

def _launch_revenue(
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
    Execute the fused Triton kernel and sum the per‑block results on the GPU.
    This avoids atomic contention and improves throughput on H100.
    """
    n = l_quantity.shape[0]
    BLOCK_SIZE = 1024  # optimal for H100 FP64 bandwidth
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Per‑block accumulation buffer
    block_sums = torch.zeros(num_blocks, dtype=torch.float64, device=l_quantity.device)

    grid = (num_blocks,)

    _K._revenue_kernel[grid](
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
        block_sums,
        n,
        BLOCK_SIZE,
    )

    # Final reduction on the GPU
    return torch.sum(block_sums)

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
    Public API – unchanged signature.
    Delegates to the fused Triton kernel for a single‑pass scan.
    """
    return _launch_revenue(
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

    # Common ship‑mode / instruction filter
    li_mask = (
        ((df_lineitem["l_shipmode"] == "AIR") | (df_lineitem["l_shipmode"] == "AIR REG"))
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    )
    li = df_lineitem[li_mask]

    # Common brand filter on part
    part_mask = (
        (df_part["p_brand"] == brand1)
        | (df_part["p_brand"] == brand2)
        | (df_part["p_brand"] == brand3)
    )
    pt = df_part[part_mask]

    # Join on partkey
    joined = li.merge(pt, left_on="l_partkey", right_on="p_partkey")

    # Encode brand ids
    brand12_id, brand23_id, brand34_id = 12, 23, 34
    p_brand_col = joined["p_brand"]
    b12 = (p_brand_col == brand1).astype("int32") * brand12_id
    b23 = (p_brand_col == brand2).astype("int32") * brand23_id
    b34 = (p_brand_col == brand3).astype("int32") * brand34_id
    brand_encoded = b12 + b23 + b34

    # Encode container groups
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

    # Transfer columns to torch tensors on GPU
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
