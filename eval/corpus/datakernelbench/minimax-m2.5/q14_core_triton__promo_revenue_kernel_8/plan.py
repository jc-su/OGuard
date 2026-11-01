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
    reset_to_zero=["out_total_ptr", "out_promo_ptr"],
)
@triton.jit
def _promo_revenue_kernel(
    shipdate_ptr, price_ptr, discount_ptr, promo_ptr,
    date_lo, date_hi, out_total_ptr, out_promo_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate with a sentinel value that fails the date predicate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    
    # Apply date filter: shipdate_lo <= shipdate < shipdate_hi
    date_keep = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load price and discount only for rows that pass date filter
    # This conditional load is a key optimization for low-selectivity filters
    price = tl.load(price_ptr + offs, mask=date_keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=date_keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=date_keep, other=0.0)
    
    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)
    
    # Accumulate total revenue
    block_total = tl.sum(revenue)
    tl.atomic_add(out_total_ptr, block_total)
    
    # Accumulate promo revenue (where promo flag is True)
    promo_revenue = tl.where(promo, revenue, 0.0)
    block_promo = tl.sum(promo_revenue)
    tl.atomic_add(out_promo_ptr, block_promo)
'''


def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q15_promo_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_promo_revenue(
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Launch fused Triton kernel for filter + project + aggregate.
    Returns (total_revenue, promo_revenue) as a tuple of tensors.
    """
    n = l_extendedprice.numel()
    
    # Output tensors for atomic accumulation
    out_total = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    out_promo = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._promo_revenue_kernel[grid](
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        out_total,
        out_promo,
        n,
    )
    
    return out_total, out_promo


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Compute promotional revenue percentage using fused Triton kernel.
    
    This replaces the reference's PyTorch-only implementation with a 
    fused Triton kernel that:
    1. Filters by shipdate range (memory-bound: conditional load)
    2. Projects to revenue = price * (1 - discount)
    3. Aggregates total and promo revenue in a single pass
    
    Using Triton atomic_add with reset_to_zero ensures correct accumulation
    across blocks without Python float32 precision issues.
    """
    total, promo = _launch_promo_revenue(
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    # Compute final ratio: 100.0 * promo / total
    return 100.0 * promo / total


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
