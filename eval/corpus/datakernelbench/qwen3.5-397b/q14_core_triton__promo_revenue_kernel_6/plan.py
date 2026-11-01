import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

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
def _promo_revenue_kernel(
    extendedprice_ptr,
    discount_ptr,
    shipdate_ptr,
    promo_ptr,
    date_lo,
    date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q14 promo revenue calculation.
    
    Single pass through all columns:
    1. Load shipdate, apply date filter (cheap, early reject)
    2. Load extendedprice, discount, compute revenue = price * (1 - discount)
    3. Load promo flag, compute promo revenue where date AND promo
    4. Block-level sum, atomic add to global accumulators
    
    Memory-bound optimization: fusion reduces HBM passes from 4+ (eager) to 1.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate first for date filter (int32 epoch days)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    
    # Date filter: shipdate_lo <= shipdate < shipdate_hi
    date_keep = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load remaining columns (all rows, but revenue zeroed for non-matching dates)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    is_promo = tl.load(promo_ptr + offs, mask=mask, other=False)
    
    # Revenue calculation: extendedprice * (1 - discount)
    revenue = extendedprice * (1.0 - discount)
    
    # Total revenue (date-filtered)
    filtered_revenue = tl.where(date_keep, revenue, 0.0)
    
    # Promo revenue (date-filtered AND promo flag)
    promo_revenue = tl.where(date_keep & is_promo, revenue, 0.0)
    
    # Block-level sums
    total_block = tl.sum(filtered_revenue)
    promo_block = tl.sum(promo_revenue)
    
    # Atomic add to global output [total, promo]
    # Note: out_ptr + 0 and out_ptr + 1 are pointer offsets, not mask args
    tl.atomic_add(out_ptr + 0, total_block)
    tl.atomic_add(out_ptr + 1, promo_block)
'''


def _load_kernels():
    """
    Load Triton kernels from temp file (required for @triton.jit in exec'd modules).
    """
    tmpdir = tempfile.mkdtemp(prefix="q14_triton_")
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
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Launch fused Triton kernel for promo revenue calculation.
    
    GPU: Single kernel launch processes all rows with fused filter+project+reduce.
    Output: 2-element tensor [total_revenue, promo_revenue], then computes percentage.
    
    Args:
        l_extendedprice: float64 tensor, lineitem extended prices
        l_discount: float64 tensor, lineitem discounts
        l_shipdate_days: int32 tensor, ship dates as epoch days
        p_type_is_promo: bool tensor, promo flag per row
        shipdate_lo_days: int, inclusive lower date bound (epoch days)
        shipdate_hi_days: int, exclusive upper date bound (epoch days)
    
    Returns:
        Scalar float64 tensor: 100 * promo_revenue / total_revenue
    """
    n = l_extendedprice.numel()
    # Output accumulator: [total_revenue, promo_revenue]
    out = torch.zeros(2, dtype=torch.float64, device=l_extendedprice.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._promo_revenue_kernel[grid](
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        out,
        n,
    )
    
    # Compute percentage: 100 * promo_sum / total_sum
    total = out[0]
    promo = out[1]
    result = 100.0 * promo / total
    
    return result


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Fused Triton kernel for TPC-H Q14 promo revenue calculation.
    
    GPU: Single fused pass through all columns with date filter, revenue calculation,
    and dual accumulation (total + promo). Reduces HBM traffic vs. reference eager ops
    by fusing filter → project → reduce into one kernel launch.
    
    Memory-bound optimization: ~60M rows × 4 columns × 8 bytes = ~1.9 GB HBM read.
    Fusion eliminates intermediate tensors and multiple kernel launches.
    
    Args:
        l_extendedprice: float64 tensor, lineitem extended prices
        l_discount: float64 tensor, lineitem discounts
        l_shipdate_days: int32 tensor, ship dates as epoch days
        p_type_is_promo: bool tensor, promo flag per row
        shipdate_lo_days: int, inclusive lower date bound (epoch days)
        shipdate_hi_days: int, exclusive upper date bound (epoch days)
    
    Returns:
        Scalar float64 tensor: promo revenue percentage (100 * promo_sum / total_sum)
    """
    return _launch_promo_revenue(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )


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
