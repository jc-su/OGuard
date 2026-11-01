import os
import tempfile
import importlib.util
import linecache
from types import ModuleType
from typing import Any

import torch
import triton
import cudf

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q19_core_kernel(
    quantity_ptr,
    extendedprice_ptr,
    discount_ptr,
    brand_id_ptr,
    container_grp_ptr,
    size_ptr,
    out_ptr,
    n_elements,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    qty = tl.load(quantity_ptr + offs, mask=valid, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=valid, other=0)
    cont = tl.load(container_grp_ptr + offs, mask=valid, other=0)
    size = tl.load(size_ptr + offs, mask=valid, other=0)

    cond1 = (
        (brand == brand12_id)
        & (cont == sm_grp)
        & (qty >= 1.0)
        & (qty <= 11.0)
        & (size >= 1)
        & (size <= 5)
    )
    cond2 = (
        (brand == brand23_id)
        & (cont == md_grp)
        & (qty >= 10.0)
        & (qty <= 20.0)
        & (size >= 1)
        & (size <= 10)
    )
    cond3 = (
        (brand == brand34_id)
        & (cont == lg_grp)
        & (qty >= 20.0)
        & (qty <= 30.0)
        & (size >= 1)
        & (size <= 15)
    )
    keep = valid & (cond1 | cond2 | cond3)

    price = tl.load(extendedprice_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    revenue = tl.where(keep, price * (1.0 - disc), 0.0)
    block_sum = tl.sum(revenue, axis=0)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_q19_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q19_core(
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
    block_size: int = 2048,
) -> torch.Tensor:
    """
    GPU hot path launcher.

    The cuDF work in run_query has already performed string filters, encoding, and the join.
    This function launches one Triton GPU kernel that fuses the remaining numeric predicates,
    projection, and global SUM reduction. It is faster than eager PyTorch here because Q19's
    tensor core is a memory-bound scan: the fused kernel avoids materializing three predicate
    masks, the combined mask, and the filtered revenue vector, and conditionally loads price
    and discount only for rows that satisfy the cheap predicates.
    """
    n = int(l_quantity.numel())
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity.device)
    if n == 0:
        return out.squeeze(0)

    grid = (triton.cdiv(n, block_size),)
    _K._q19_core_kernel[grid](
        l_quantity,
        l_extendedprice,
        l_discount,
        p_brand_id,
        p_container_grp,
        p_size,
        out,
        n,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=4,
    )
    return out.squeeze(0)


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
    Run the numeric Q19 predicate/project/aggregate stage on the GPU with Triton.

    run_query performs cuDF I/O, string filtering, joining, and categorical encoding, then
    passes aligned 1D tensors here. This core replaces the eager PyTorch mask construction
    and torch.sum with a single fused Triton scan/reduction, reducing HBM traffic and temporary
    allocations while preserving the reference fp64 aggregate result.
    """
    return _launch_q19_core(
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
