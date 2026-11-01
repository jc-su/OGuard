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
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q14_kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load and filter by shipdate
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = (shipdate >= date_lo) & (shipdate < date_hi)
    keep_mask = mask & keep
    
    # Conditionally load expensive columns (hardware skips fetch if keep_mask is false for the warp)
    price = tl.load(price_ptr + offs, mask=keep_mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep_mask, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep_mask, other=0)
    
    # Compute revenue and mask out invalid lanes
    revenue = price * (1.0 - discount)
    revenue = tl.where(keep_mask, revenue, 0.0)
    
    # Compute promotional revenue
    promo_revenue = tl.where(promo != 0, revenue, 0.0)
    
    # Compute block sums
    total_rev = tl.sum(revenue)
    total_promo = tl.sum(promo_revenue)
    
    # Accumulate globally
    tl.atomic_add(out_ptr, total_promo)
    tl.atomic_add(out_ptr + 1, total_rev)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q14_triton_")
    path = os.path.join(tmpdir, "_q14_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q14_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Executes a fused Triton kernel over the joined tables to calculate the total and promotional 
    revenue in a single memory-bound pass, avoiding several intermediate tensor allocations.
    """
    n = l_extendedprice.numel()
    
    # Array to hold both sums: [promo_sum, total_sum]
    out = torch.zeros(2, dtype=torch.float64, device=l_extendedprice.device)
    
    if n > 0:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _K._q14_kernel[grid](
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            p_type_is_promo,
            shipdate_lo_days,
            shipdate_hi_days,
            out,
            n,
        )
    
    # Compute output matching reference return type (scalar tensor)
    return 100.0 * out[0] / out[1]


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
