import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd


# ============================================================================
# TRITON KERNEL SOURCE - fused filter + project + aggregate
# ============================================================================
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
    shipdate_ptr, discount_ptr, price_ptr, promo_ptr,
    date_lo, date_hi,
    out_total_ptr, out_promo_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns - all at once for coalesced access
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=mask, other=0)

    # Filter predicate: shipdate in range [date_lo, date_hi)
    keep = (shipdate >= date_lo) & (shipdate < date_hi)

    # Project: revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # Apply filter: zero out rows outside date range
    revenue_filtered = tl.where(keep, revenue, 0.0)

    # Aggregate: total sum
    total = tl.sum(revenue_filtered)
    tl.atomic_add(out_total_ptr, total)

    # Aggregate: promo sum (only where promo flag is True AND date filter passes)
    promo_revenue = tl.where(keep & promo, revenue, 0.0)
    promo_sum = tl.sum(promo_revenue)
    tl.atomic_add(out_promo_ptr, promo_sum)
'''


def _load_kernels():
    """Load Triton kernels from on-disk source file."""
    tmpdir = tempfile.mkdtemp(prefix="q6_promo_triton_")
    path = os.path.join(tmpdir, "_promo_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_promo_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_promo_revenue(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_extendedprice: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Launch fused Triton kernel for filter + project + aggregate.
    Returns (total_revenue, promo_revenue) as a tensor.
    """
    n = l_extendedprice.numel()
    
    # Allocate output tensors (fp64 for precision)
    out_total = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    out_promo = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._promo_revenue_kernel[grid](
        l_shipdate_days, l_discount, l_extendedprice, p_type_is_promo,
        shipdate_lo_days, shipdate_hi_days,
        out_total, out_promo,
        n,
    )
    
    return torch.cat([out_total, out_promo])


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Core computation: filter by shipdate, compute revenue, aggregate total and promo.
    Returns the final percentage: 100.0 * promo_sum / total_sum
    """
    results = _launch_promo_revenue(
        l_shipdate_days,
        l_discount,
        l_extendedprice,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    
    total = results[0]
    promo = results[1]
    
    # Avoid division by zero
    return 100.0 * promo / total


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    This implementation uses a fused Triton kernel for the hot path (filter + project + aggregate),
    avoiding multiple passes over the data and minimizing HBM traffic by loading columns once
    and performing all computations in registers before atomic adds.

    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Convert date strings to int32 epoch days
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Join lineitem with part to get promo flag
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Convert to torch tensors (GPU)
    # shipdate as int32 days for efficient comparison
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(
        joined["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        joined["l_discount"].to_cupy(), dtype=torch.float64
    )
    p_type_is_promo_t = torch.as_tensor(
        joined["p_type_is_promo"].astype("bool").to_cupy()
    )

    # Execute fused kernel
    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    
    return float(res.item())
