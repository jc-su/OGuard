import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# ----------------------------------------------------------------------
# Triton kernel source (written to a temporary .py file at import time)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configurations focused on H100: larger BLOCK_SIZE gives fewer blocks
# and thus fewer atomic adds. 4‑warps per block is a good default for FP64.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [2048, 4096, 8192, 16384]
    ],
    key=["n_elements"],
    reset_to_zero=["out_total_ptr", "out_promo_ptr"],
)
@triton.jit
def _promo_rev_kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi,
    out_total_ptr, out_promo_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused scan:
      * filter by shipdate range
      * compute revenue = price * (1 - discount)
      * atomic sum of total revenue
      * atomic sum of revenue for promotional parts
    All arithmetic is fp64; promo flag is uint8 (0/1).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate (int32) for all active lanes.
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)

    # Conditional loads – only rows passing the date filter are fetched.
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep, other=0)

    # Revenue per kept row (fp64)
    revenue = price * (1.0 - discount)

    # Reduce within the block and atomically add to global accumulators.
    total_sum = tl.sum(revenue)
    promo_sum = tl.sum(tl.where(promo != 0, revenue, 0.0))

    tl.atomic_add(out_total_ptr, total_sum)
    tl.atomic_add(out_promo_ptr, promo_sum)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="promo_rev_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_promo_rev_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load Triton kernels once at import time.
_K = _load_kernels()

def _launch_promo_rev(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the fused Triton kernel.
    Returns (total_revenue, promo_revenue) as 1‑element fp64 tensors on the same device.
    """
    n = l_extendedprice.numel()
    out_total = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    out_promo = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)

    # Triton prefers integer types for the promo flag; cast once outside the kernel.
    promo_uint8 = p_type_is_promo.to(torch.uint8)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._promo_rev_kernel[grid](
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        promo_uint8,
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
    Triton‑fused implementation of the promo‑revenue percentage.
    Returns a scalar fp64 tensor containing 100 * promo / total.
    """
    total, promo = _launch_promo_rev(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return (100.0 * promo / total).squeeze()

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

    # Pre‑compute promotional flag on part side before join.
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Convert columns to GPU tensors.
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    # Compute promo revenue percentage.
    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
