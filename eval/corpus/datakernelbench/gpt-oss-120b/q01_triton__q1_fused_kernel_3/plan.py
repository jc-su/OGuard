import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source (written to disk and imported)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["N_ELEM"],
    reset_to_zero=[
        "OUT_SUM_QTY", "OUT_SUM_BASE", "OUT_SUM_DISC_PRICE",
        "OUT_SUM_CHARGE", "OUT_SUM_DISC", "OUT_COUNT"
    ],
)
@triton.jit
def _q1_fused_kernel(
    # input column pointers
    RF_PTR, LS_PTR, QTY_PTR, PRICE_PTR, DISC_PTR, TAX_PTR, SHIPDATE_PTR,
    # scalar filter bound (int32 days)
    SHIPDATE_HI,
    # output per‑group accumulators (dense key = rf*256+ls)
    OUT_SUM_QTY, OUT_SUM_BASE, OUT_SUM_DISC_PRICE,
    OUT_SUM_CHARGE, OUT_SUM_DISC, OUT_COUNT,
    # meta
    N_ELEM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM

    # Load shipdate and apply filter
    shipdate = tl.load(SHIPDATE_PTR + offs, mask=mask, other=0)
    keep = mask & (shipdate <= SHIPDATE_HI)

    # Load remaining columns only for rows that satisfy the filter
    rf = tl.load(RF_PTR + offs, mask=keep, other=0)
    ls = tl.load(LS_PTR + offs, mask=keep, other=0)
    qty = tl.load(QTY_PTR + offs, mask=keep, other=0.0)
    price = tl.load(PRICE_PTR + offs, mask=keep, other=0.0)
    disc = tl.load(DISC_PTR + offs, mask=keep, other=0.0)
    tax = tl.load(TAX_PTR + offs, mask=keep, other=0.0)

    # Dense group key (fits in int32)
    group = rf * 256 + ls

    # Derived expressions (float64 arithmetic)
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Atomic accumulation per group (float64 sums)
    tl.atomic_add(OUT_SUM_QTY + group, qty)
    tl.atomic_add(OUT_SUM_BASE + group, price)
    tl.atomic_add(OUT_SUM_DISC_PRICE + group, disc_price)
    tl.atomic_add(OUT_SUM_CHARGE + group, charge)
    tl.atomic_add(OUT_SUM_DISC + group, disc)

    # Conditional count increment (int32)
    inc = tl.where(keep, 1, 0)
    tl.atomic_add(OUT_COUNT + group, inc)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q1_triton_")
    path = os.path.join(tmpdir, "q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("q1_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper / launch functions
# ----------------------------------------------------------------------
_MAX_GROUPS = 256 * 256  # enough for uint8 returnflag & linestatus codes

def _launch_fused_agg(
    rf_code: torch.Tensor,
    ls_code: torch.Tensor,
    qty: torch.Tensor,
    price: torch.Tensor,
    disc: torch.Tensor,
    tax: torch.Tensor,
    shipdate_days: torch.Tensor,
    shipdate_hi: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Launch the fused Triton kernel that performs:
      * filter on l_shipdate <= shipdate_hi
      * dense group‑by on (returnflag, linestatus)
      * all required sum aggregations
    Returns dense per‑group accumulator tensors sized ``_MAX_GROUPS``.
    """
    device = rf_code.device
    n = rf_code.numel()

    # Allocate per‑group buffers
    sum_qty = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_base = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_disc_price = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_charge = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_disc = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    count = torch.zeros(_MAX_GROUPS, dtype=torch.int32, device=device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._q1_fused_kernel[grid](
        rf_code,
        ls_code,
        qty,
        price,
        disc,
        tax,
        shipdate_days,
        shipdate_hi,
        sum_qty,
        sum_base,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count,
        n,
    )
    return sum_qty, sum_base, sum_disc_price, sum_charge, sum_disc, count

def _postprocess_aggregates(
    sum_qty: torch.Tensor,
    sum_base: torch.Tensor,
    sum_disc_price: torch.Tensor,
    sum_charge: torch.Tensor,
    sum_disc: torch.Tensor,
    count_int: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Convert dense per‑group buffers into the compact, sorted result
    matching the reference implementation.
    """
    # Identify groups with at least one row
    mask = count_int > 0
    keys = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # Decode dense key back to categorical codes
    rf_codes = (keys // 256).to(torch.int64)
    ls_codes = (keys % 256).to(torch.int64)

    # Gather aggregates for active groups
    sum_qty = sum_qty[keys]
    sum_base = sum_base[keys]
    sum_disc_price = sum_disc_price[keys]
    sum_charge = sum_charge[keys]
    sum_disc = sum_disc[keys]
    count = count_int[keys].to(torch.int64)

    # Compute averages (float64 division)
    cnt_f = count.to(torch.float64)
    avg_qty = sum_qty / cnt_f
    avg_price = sum_base / cnt_f
    avg_disc = sum_disc / cnt_f

    # Sort by numeric key (ensures order by returnflag then linestatus)
    order = torch.argsort(keys)

    return (
        rf_codes[order],
        ls_codes[order],
        sum_qty[order],
        sum_base[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        count[order],
    )

# ----------------------------------------------------------------------
# Core query (public)
# ----------------------------------------------------------------------
def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Fused aggregation for TPC‑H Q1 using a single Triton kernel.
    Mirrors the reference semantics while drastically reducing HBM traffic.
    """
    (
        sum_qty,
        sum_base,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count,
    ) = _launch_fused_agg(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
    )
    return _postprocess_aggregates(
        sum_qty, sum_base, sum_disc_price, sum_charge, sum_disc, count
    )

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_hi: str = "1998-09-02",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            l_returnflag,
            l_linestatus,
            sum(l_quantity) AS sum_qty,
            sum(l_extendedprice) AS sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            avg(l_quantity) AS avg_qty,
            avg(l_extendedprice) AS avg_price,
            avg(l_discount) AS avg_disc,
            count(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= CAST('1998-09-02' AS date)
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus;
    Description:
        The reference plan materialized each column separately and performed
        several PyTorch kernels.  This implementation fuses the filter,
        projection, and all aggregates into a single Triton kernel, reducing
        memory traffic to a single pass over the table and removing
        Python‑level temporaries.
    Params:
        tables (dict): must contain a ``lineitem`` cuDF DataFrame.
        shipdate_hi (str): inclusive upper bound for ``l_shipdate``.
    Returns:
        cudf.DataFrame with the ten columns listed in the SELECT clause,
        sorted by ``l_returnflag`` then ``l_linestatus``.
    """
    df_lineitem = tables["lineitem"]

    # Convert filter bound to integer days since epoch
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns as categorical integer codes (uint8 range)
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    # Transfer required columns to GPU tensors
    l_returnflag_code_t = torch.as_tensor(
        rf_codes.cat.codes.astype("int32").to_cupy()
    )
    l_linestatus_code_t = torch.as_tensor(
        ls_codes.cat.codes.astype("int32").to_cupy()
    )
    l_quantity_t = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64
    )
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_tax_t = torch.as_tensor(
        df_lineitem["l_tax"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400)
         .astype("int32")).to_cupy()
    )

    # Execute fused aggregation
    (
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode integer codes back to original strings
    rf_strings = [rf_categories[i] for i in rf_out.cpu().tolist()]
    ls_strings = [ls_categories[i] for i in ls_out.cpu().tolist()]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strings),
        "l_linestatus": cudf.Series(ls_strings),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_order,
    })
