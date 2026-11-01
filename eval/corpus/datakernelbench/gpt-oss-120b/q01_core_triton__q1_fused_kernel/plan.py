import os
import tempfile
import importlib.util
import linecache
import torch
import triton  # needed for triton.cdiv in host code
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernels (written to a temporary .py file and imported)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# --------------------------------------------------------------
# Fully fused kernel for TPC‑H Q1
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=[
        "out_sum_qty", "out_sum_base", "out_sum_disc_price",
        "out_sum_charge", "out_sum_disc", "out_count"
    ],
)
@triton.jit
def _q1_fused_kernel(
    rf_ptr,                # int32 *
    ls_ptr,                # int32 *
    qty_ptr,               # fp64 *
    price_ptr,             # fp64 *
    disc_ptr,              # fp64 *
    tax_ptr,               # fp64 *
    shipdate_ptr,          # int32 *
    shipdate_hi,           # int32 (scalar)
    out_sum_qty,           # fp64 *
    out_sum_base,          # fp64 *
    out_sum_disc_price,    # fp64 *
    out_sum_charge,        # fp64 *
    out_sum_disc,          # fp64 *
    out_count,             # int64 *
    n_elements,            # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load cheap columns (always)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0).to(tl.int32)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0).to(tl.int32)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Apply date predicate
    keep = mask & (shipdate <= shipdate_hi)

    # Conditional loads of expensive columns (zero when not kept)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Group key (fits in int32)
    grp = rf * 256 + ls

    # Derived columns
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Atomic per‑group accumulation
    tl.atomic_add(out_sum_qty + grp, qty)
    tl.atomic_add(out_sum_base + grp, price)
    tl.atomic_add(out_sum_disc_price + grp, disc_price)
    tl.atomic_add(out_sum_charge + grp, charge)
    tl.atomic_add(out_sum_disc + grp, disc)

    # Conditional count increment (only for rows that satisfy predicate)
    add_one = tl.where(keep, 1, 0).to(tl.int64)
    tl.atomic_add(out_count + grp, add_one)
'''

def _load_kernels() -> object:
    """Write the kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Launch the fused kernel and post‑process results
# ----------------------------------------------------------------------
def _launch_q1(
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
    Execute a single Triton kernel that performs the date filter, computes the
    derived columns and atomically aggregates all required metrics per (returnflag,
    linestatus) group. Returns tensors in the same order/dtype as the reference.
    """
    device = l_quantity.device
    n = l_quantity.shape[0]

    # Upper bound on distinct groups (256 * 256)
    MAX_GROUPS = 256 * 256

    # Allocate per‑group accumulators (fp64 for sums, int64 for count)
    sum_qty = torch.zeros(MAX_GROUPS, dtype=torch.float64, device=device)
    sum_base = torch.zeros_like(sum_qty)
    sum_disc_price = torch.zeros_like(sum_qty)
    sum_charge = torch.zeros_like(sum_qty)
    sum_disc = torch.zeros_like(sum_qty)
    count = torch.zeros(MAX_GROUPS, dtype=torch.int64, device=device)

    # Ensure inputs are int32 where the kernel expects them
    rf_i32 = l_returnflag_code.to(torch.int32)
    ls_i32 = l_linestatus_code.to(torch.int32)
    ship_i32 = l_shipdate_days.to(torch.int32)

    # Grid launch (autotuned BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._q1_fused_kernel[grid](
        rf_i32,
        ls_i32,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        ship_i32,
        shipdate_hi_days,
        sum_qty,
        sum_base,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count,
        n,
    )

    # --------------------------------------------------------------
    # Post‑processing on the GPU
    # --------------------------------------------------------------
    nonzero_mask = count > 0
    group_idx = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(1)

    sum_qty = sum_qty[group_idx]
    sum_base = sum_base[group_idx]
    sum_disc_price = sum_disc_price[group_idx]
    sum_charge = sum_charge[group_idx]
    sum_disc = sum_disc[group_idx]
    count = count[group_idx]

    # Compute averages (float64)
    cnt_f = count.to(torch.float64)
    avg_qty = sum_qty / cnt_f
    avg_price = sum_base / cnt_f
    avg_disc = sum_disc / cnt_f

    # Decode group key back to original columns
    rf_out = (group_idx // 256).to(torch.int64)
    ls_out = (group_idx % 256).to(torch.int64)

    # Sort by (rf, ls) to match reference ordering
    sort_key = rf_out * 256 + ls_out
    order = torch.argsort(sort_key)

    return (
        rf_out[order],
        ls_out[order],
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
# Public query core – signature identical to the reference
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
    Core compute for TPC‑H Q1.
    Replaces the eager PyTorch aggregation with a single fused Triton kernel
    that atomically accumulates all aggregates per (l_returnflag,
    l_linestatus) group, delivering exact results while dramatically reducing
    memory traffic.
    """
    return _launch_q1(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
    )

# ----------------------------------------------------------------------
# run_query – unchanged from the reference (copy‑paste)
# ----------------------------------------------------------------------
def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns l_returnflag, l_linestatus to integer codes (single-char strings).
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    # Use categorical codes for encoding
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

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

    # Decode category codes back to strings
    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_order,
    })
