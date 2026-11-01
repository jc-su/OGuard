import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source (will be written to disk and loaded)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Group multiplier for encoding (rf_code * 256 + ls_code)
_GROUP_MULT = 256

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=[
        "sum_qty_ptr", "sum_base_price_ptr", "sum_disc_price_ptr",
        "sum_charge_ptr", "sum_disc_ptr", "count_ptr"
    ],
)
@triton.jit
def _q1_kernel(
    rf_ptr, ls_ptr,
    qty_ptr, price_ptr, discount_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi,
    sum_qty_ptr, sum_base_price_ptr, sum_disc_price_ptr,
    sum_charge_ptr, sum_disc_ptr, count_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # load date threshold once per program
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load all columns (unconditional loads; filtered rows contribute zero)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=mask, other=0.0)

    # filter predicate: shipdate <= shipdate_hi
    keep = shipdate <= shipdate_hi

    # compute projected values only for kept rows (zero otherwise)
    qty_agg = tl.where(keep, qty, 0.0)
    price_agg = tl.where(keep, price, 0.0)
    discount_agg = tl.where(keep, discount, 0.0)
    # derived values
    disc_price_agg = tl.where(keep, price * (1.0 - discount), 0.0)
    charge_agg = tl.where(keep, disc_price_agg * (1.0 + tax), 0.0)
    count_agg = tl.where(keep, 1, 0)

    # group key = returnflag_code * 256 + linestatus_code
    # use inline constant 256 instead of global
    group_key = rf * 256 + ls

    # atomic adds for each aggregate
    tl.atomic_add(sum_qty_ptr + group_key, qty_agg)
    tl.atomic_add(sum_base_price_ptr + group_key, price_agg)
    tl.atomic_add(sum_disc_price_ptr + group_key, disc_price_agg)
    tl.atomic_add(sum_charge_ptr + group_key, charge_agg)
    tl.atomic_add(sum_disc_ptr + group_key, discount_agg)
    tl.atomic_add(count_ptr + group_key, count_agg)
'''

# ----------------------------------------------------------------------
# Load the Triton kernels from the on‑disk file
# ----------------------------------------------------------------------
def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

# ----------------------------------------------------------------------
# Launch wrapper
# ----------------------------------------------------------------------
def _run_fused_aggregation(
    rf: torch.Tensor,
    ls: torch.Tensor,
    qty: torch.Tensor,
    price: torch.Tensor,
    discount: torch.Tensor,
    tax: torch.Tensor,
    shipdate: torch.Tensor,
    shipdate_hi: int,
    max_groups: int,
):
    """Run the fused filter + aggregate kernel and return the raw aggregated tensors."""
    n = rf.shape[0]

    # allocate output buffers (max_groups = 256*256 = 65536)
    sum_qty = torch.zeros(max_groups, dtype=torch.float64, device=rf.device)
    sum_base_price = torch.zeros(max_groups, dtype=torch.float64, device=rf.device)
    sum_disc_price = torch.zeros(max_groups, dtype=torch.float64, device=rf.device)
    sum_charge = torch.zeros(max_groups, dtype=torch.float64, device=rf.device)
    sum_disc = torch.zeros(max_groups, dtype=torch.float64, device=rf.device)
    count = torch.zeros(max_groups, dtype=torch.int64, device=rf.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._q1_kernel[grid](
        rf, ls,
        qty, price, discount, tax, shipdate,
        shipdate_hi,
        sum_qty, sum_base_price, sum_disc_price,
        sum_charge, sum_disc, count,
        n,
    )
    return sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count


# ----------------------------------------------------------------------
# Main query entry point
# ----------------------------------------------------------------------
def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus,
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

    Description: TPC‑H Q1 – pricing summary report. Uses a fused Triton kernel
    for the hot path (filter + projection + aggregation) to minimise memory
    traffic and avoid multiple GPU–CPU round‑trips.

    Params:
        tables (dict): dictionary with key "lineitem" mapping to a cudf.DataFrame.
        shipdate_hi (str): upper bound (inclusive) for l_shipdate, default '1998-09-02'.
    """
    df_line = tables["lineitem"]

    # ------------------------------------------------------------------
    # Encode categorical columns to integer codes
    # ------------------------------------------------------------------
    rf_cat = df_line["l_returnflag"].astype("category")
    ls_cat = df_line["l_linestatus"].astype("category")
    rf_codes = rf_cat.cat.codes.astype("int32")
    ls_codes = ls_cat.cat.codes.astype("int32")

    # Keep category lists for later decoding
    rf_categories = rf_cat.cat.categories.to_pandas().tolist()
    ls_categories = ls_cat.cat.categories.to_pandas().tolist()

    # ------------------------------------------------------------------
    # Convert to GPU tensors
    # ------------------------------------------------------------------
    rf_t = torch.as_tensor(rf_codes.to_cupy(), dtype=torch.int32)
    ls_t = torch.as_tensor(ls_codes.to_cupy(), dtype=torch.int32)
    qty_t = torch.as_tensor(df_line["l_quantity"].to_cupy(), dtype=torch.float64)
    price_t = torch.as_tensor(df_line["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(df_line["l_discount"].to_cupy(), dtype=torch.float64)
    tax_t = torch.as_tensor(df_line["l_tax"].to_cupy(), dtype=torch.float64)

    # shipdate as int32 days since epoch
    shipdate_days = ((df_line["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    shipdate_t = torch.as_tensor(shipdate_days, dtype=torch.int32)

    # Upper bound date in epoch days
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # Run fused kernel
    # ------------------------------------------------------------------
    max_groups = 256 * 256  # 65536
    (
        sum_qty,
        sum_base_price,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count,
    ) = _run_fused_aggregation(
        rf_t, ls_t,
        qty_t, price_t, discount_t, tax_t,
        shipdate_t,
        shipdate_hi_days,
        max_groups,
    )

    # ------------------------------------------------------------------
    # Extract groups with non‑zero count and compute averages
    # ------------------------------------------------------------------
    # mask where count > 0
    valid_mask = count > 0
    # indices of active groups (they are already in ascending order)
    group_idx = torch.arange(max_groups, device=count.device)[valid_mask]

    # pull out aggregated values for those groups
    sum_qty_g = sum_qty[group_idx]
    sum_base_price_g = sum_base_price[group_idx]
    sum_disc_price_g = sum_disc_price[group_idx]
    sum_charge_g = sum_charge[group_idx]
    sum_disc_g = sum_disc[group_idx]
    count_g = count[group_idx]

    # averages (float64)
    count_f = count_g.to(torch.float64)
    avg_qty = sum_qty_g / count_f
    avg_price = sum_base_price_g / count_f
    avg_disc = sum_disc_g / count_f

    # decode group keys back to returnflag / linestatus strings
    rf_codes_out = (group_idx // 256).tolist()
    ls_codes_out = (group_idx % 256).tolist()
    rf_strs = [rf_categories[i] for i in rf_codes_out]
    ls_strs = [ls_categories[i] for i in ls_codes_out]

    # ------------------------------------------------------------------
    # Build result DataFrame
    # ------------------------------------------------------------------
    result = cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": cudf.Series(sum_qty_g.cpu().numpy()),
        "sum_base_price": cudf.Series(sum_base_price_g.cpu().numpy()),
        "sum_disc_price": cudf.Series(sum_disc_price_g.cpu().numpy()),
        "sum_charge": cudf.Series(sum_charge_g.cpu().numpy()),
        "avg_qty": cudf.Series(avg_qty.cpu().numpy()),
        "avg_price": cudf.Series(avg_price.cpu().numpy()),
        "avg_disc": cudf.Series(avg_disc.cpu().numpy()),
        "count_order": cudf.Series(count_g.cpu().numpy()),
    })

    return result
