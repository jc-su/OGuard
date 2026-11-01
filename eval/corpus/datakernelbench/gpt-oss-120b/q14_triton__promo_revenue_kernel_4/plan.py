import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source – must be written to a temporary file before
# compilation (see framework guidelines).
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune for H100 – a range of block sizes, warps and stages.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=tw, num_stages=ns)
        for bs in [1024, 2048, 4096, 8192]
        for tw in [4, 8]
        for ns in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["total_out_ptr", "promo_out_ptr"],
)
@triton.jit
def _promo_revenue_kernel(
    ext_ptr,                # *float64* – l_extendedprice
    disc_ptr,               # *float64* – l_discount
    shipdate_ptr,           # *int32*   – l_shipdate (days)
    promo_ptr,              # *int8*    – p_type_is_promo (0/1)
    date_lo,                # int32    – lower bound (inclusive)
    date_hi,                # int32    – upper bound (exclusive)
    total_out_ptr,          # *float64* – atomic sum of all revenue
    promo_out_ptr,          # *float64* – atomic sum of promo revenue
    n_elements,             # total number of rows
    BLOCK_SIZE: tl.constexpr,
):
    """Fused scan:
    1. Filter rows by shipdate window.
    2. Compute revenue = extendedprice * (1 - discount).
    3. Accumulate total revenue.
    4. Conditionally accumulate promo revenue where promo flag == 1.
    All work is done in a single pass to minimise HBM traffic.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate and apply date filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)

    # Load values needed for revenue computation
    discount = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    extprice = tl.load(ext_ptr + offs, mask=mask, other=0.0)

    revenue = extprice * (1.0 - discount)

    # Total revenue contribution (only rows that survive the date filter)
    total_contrib = tl.where(mask & keep_date, revenue, 0.0)

    # Load promo flag (stored as int8, 0 or 1)
    promo_flag = tl.load(promo_ptr + offs, mask=mask, other=0)

    # Promo revenue contribution
    promo_contrib = tl.where(mask & keep_date & (promo_flag != 0), revenue, 0.0)

    # Reduce within the block and atomically add to global scalars
    tl.atomic_add(total_out_ptr, tl.sum(total_contrib))
    tl.atomic_add(promo_out_ptr, tl.sum(promo_contrib))
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="promo_rev_triton_")
    path = os.path.join(tmpdir, "_promo_rev_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_promo_rev_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time.
_K = _load_kernels()

def _launch_promo_revenue(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """Execute the fused Triton kernel and return the final scalar result."""
    n = l_extendedprice.numel()
    device = l_extendedprice.device

    # Output buffers – single-element FP64 tensors for atomic accumulation.
    total = torch.zeros(1, dtype=torch.float64, device=device)
    promo = torch.zeros(1, dtype=torch.float64, device=device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._promo_revenue_kernel[grid](
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        total,
        promo,
        n,
    )
    # Compute the percentage as defined in the SQL query.
    result = 100.0 * promo.squeeze() / total.squeeze()
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
    Hot path entry point – delegates to the fused Triton kernel.
    Parameters match the reference plan; the return is a scalar ``torch.Tensor``.
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
    SQL:
        SELECT 100.00 *
               sum(CASE WHEN p_type LIKE 'PROMO%' THEN
                           l_extendedprice * (1 - l_discount)
                       ELSE 0 END) /
               sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= date '1995-09-01'
          AND l_shipdate < CAST('1995-10-01' AS date);
    Description:
        The join and string predicate (p_type startswith 'PROMO') are handled by cuDF.
        The remaining scan – date filter, revenue projection, and the two aggregates –
        are fused into a single Triton kernel, reducing memory traffic from
        several passes to just one. This fused pass is the main speedup over the
        reference eager Torch implementation.
    Params:
        shipdate_lo (str): inclusive lower bound of l_shipdate.
        shipdate_hi (str): exclusive upper bound of l_shipdate.
    Returns:
        float: the promotional revenue percentage.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Convert date bounds to int32 epoch days.
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Build the promo flag on the part side (boolean -> int8 for Triton).
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Perform the equi‑join.
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Materialize columns as GPU tensors.
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(
        joined["p_type_is_promo"].astype("int8").to_cupy(),
        dtype=torch.int8,
    )

    # Execute fused GPU computation.
    res_tensor = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res_tensor.item())
