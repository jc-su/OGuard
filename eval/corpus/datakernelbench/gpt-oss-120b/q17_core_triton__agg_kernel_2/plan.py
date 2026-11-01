import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for triton.cdiv in grid calculations

# ----------------------------------------------------------------------
# Triton kernels (written to disk and loaded at import time)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel 1: per‑part sum of quantity (fp64) and count (fp64)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_rows"],
    reset_to_zero=["sum_ptr", "cnt_ptr"],
)
@triton.jit
def _agg_kernel(
    idx_ptr,                # int64*  (part index for each selected row)
    qty_ptr,                # fp64*   (quantity for each selected row)
    sum_ptr,                # fp64*   (per‑part sum of quantity)
    cnt_ptr,                # fp64*   (per‑part count, stored as fp64)
    n_rows,                 # int64   (number of selected rows)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Guard against out‑of‑bounds contributions
    add_qty = tl.where(mask, qty, 0.0)
    add_cnt = tl.where(mask, 1.0, 0.0)

    tl.atomic_add(sum_ptr + idx, add_qty)   # fp64 sum
    tl.atomic_add(cnt_ptr + idx, add_cnt)   # fp64 count


# ------------------------------------------------------------------
# Kernel 2: filter rows by 0.2 * avg_quantity and sum extendedprice
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_rows"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _filter_sum_kernel(
    idx_ptr,                # int64*  (part index for each selected row)
    qty_ptr,                # fp64*   (quantity for each selected row)
    price_ptr,              # fp64*   (extendedprice for each selected row)
    thresh_ptr,             # fp64*   (per‑part 0.2 * avg_quantity)
    out_ptr,                # fp64*   (global accumulator)
    n_rows,                 # int64   (number of selected rows)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)

    thresh = tl.load(thresh_ptr + idx, mask=mask, other=0.0)

    keep = mask & (qty < thresh)

    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price, 0.0)))
'''
# ----------------------------------------------------------------------
# Load the kernels from the temporary module
# ----------------------------------------------------------------------
def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="tpch_q17_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()  # expose kernels as _K._agg_kernel, _K._filter_sum_kernel


# ----------------------------------------------------------------------
# Private helper that launches the fused Triton implementation
# ----------------------------------------------------------------------
def _launch_q17_fused(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Triton‑fused implementation of TPC‑H Q17 core compute.

    Steps (all on GPU):
    1. Sort ``part_keys`` to obtain a dense, monotonic list.
    2. Map each lineitem row to the index in the sorted list via
       ``torch.searchsorted``; keep rows whose partkey is in the filtered set.
    3. ``_agg_kernel`` atomically accumulates per‑part ``sum_qty`` (fp64) and
       ``cnt_qty`` (fp64).
    4. Host computes ``avg_qty`` and ``threshold = 0.2 * avg_qty`` (fp64).
    5. ``_filter_sum_kernel`` atomically adds ``l_extendedprice`` for rows where
       ``l_quantity < threshold``.
    6. Returns the global sum divided by 7.0.
    """
    # 1️⃣ Sort filtered part keys
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()

    # 2️⃣ Join via searchsorted and keep only matching rows
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)
    in_part = sorted_keys[idx_clamped] == l_partkey

    sel_idx = idx_clamped[in_part]
    sel_qty = l_quantity[in_part]
    sel_price = l_extendedprice[in_part]

    if sel_idx.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=l_partkey.device)

    # Allocate per‑part accumulators (both fp64)
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_partkey.device)

    # 3️⃣ Aggregate quantities & counts
    n_rows = sel_idx.numel()
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._agg_kernel[grid](
        sel_idx,
        sel_qty,
        sum_qty,
        cnt_qty,
        n_rows,
    )

    # 4️⃣ Compute per‑part average & threshold
    avg_qty = torch.where(cnt_qty > 0.0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))
    threshold = 0.2 * avg_qty

    # 5️⃣ Final reduction
    out = torch.zeros(1, dtype=torch.float64, device=l_partkey.device)
    _K._filter_sum_kernel[grid](
        sel_idx,
        sel_qty,
        sel_price,
        threshold,
        out,
        n_rows,
    )

    # 6️⃣ Scale as required by the query
    return out.squeeze(0) / 7.0


# ----------------------------------------------------------------------
# Public API – unchanged signature (core level)
# ----------------------------------------------------------------------
def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC‑H Q17 using a fused Triton pipeline.
    """
    return _launch_q17_fused(l_partkey, l_quantity, l_extendedprice, part_keys)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC‑H Q17. Computes the small‑quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
