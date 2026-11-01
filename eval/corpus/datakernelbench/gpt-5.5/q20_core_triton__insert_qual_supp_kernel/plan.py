import os
import tempfile
import importlib.util
import linecache
from types import ModuleType

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _insert_ps_kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    ht_part_ptr,
    ht_supp_ptr,
    ht_idx_ptr,
    table_mask,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    p = tl.load(ps_partkey_ptr + pid)
    s = tl.load(ps_suppkey_ptr + pid)

    x = (p * 1000003) ^ (s * 1009837)
    x = x ^ (x >> 16)
    h = x & table_mask

    done = False
    neg_one = tl.full((), -1, tl.int64)
    neg_two = tl.full((), -2, tl.int64)
    pid64 = pid.to(tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        slot = (h + probe) & table_mask
        cmp_val = tl.where(done, neg_two, neg_one)
        old = tl.atomic_cas(ht_idx_ptr + slot, cmp_val, pid64)
        success = (old == neg_one) & (~done)
        tl.store(ht_part_ptr + slot, p, mask=success)
        tl.store(ht_supp_ptr + slot, s, mask=success)
        done = done | success


@triton.jit
def _accum_lineitem_kernel(
    l_partkey_ptr,
    l_suppkey_ptr,
    l_quantity_ptr,
    ht_part_ptr,
    ht_supp_ptr,
    ht_idx_ptr,
    sum_qty_ptr,
    n_lineitem,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitem

    p = tl.load(l_partkey_ptr + offs, mask=mask, other=-1)
    s = tl.load(l_suppkey_ptr + offs, mask=mask, other=-1)
    q = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)

    x = (p * 1000003) ^ (s * 1009837)
    x = x ^ (x >> 16)
    h = x & table_mask

    found = tl.full((BLOCK_SIZE,), False, tl.int1)
    miss = tl.full((BLOCK_SIZE,), False, tl.int1)
    found_idx = tl.full((BLOCK_SIZE,), 0, tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        active = mask & (~found) & (~miss)
        slot = (h + probe) & table_mask
        cand = tl.load(ht_idx_ptr + slot, mask=active, other=-1)
        has_value = cand >= 0
        hp = tl.load(ht_part_ptr + slot, mask=active & has_value, other=-1)
        hs = tl.load(ht_supp_ptr + slot, mask=active & has_value, other=-1)

        match = active & has_value & (hp == p) & (hs == s)
        empty = active & (cand == -1)

        found_idx = tl.where(match, cand, found_idx)
        found = found | match
        miss = miss | empty

    tl.atomic_add(sum_qty_ptr + found_idx, q, mask=found, sem="relaxed")


@triton.jit
def _insert_qual_supp_kernel(
    ps_suppkey_ptr,
    ps_availqty_ptr,
    sum_qty_ptr,
    supp_set_ptr,
    table_mask,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    supp = tl.load(ps_suppkey_ptr + pid)
    avail = tl.load(ps_availqty_ptr + pid).to(tl.float64)
    qty = tl.load(sum_qty_ptr + pid)
    qual = (qty > 0.0) & (avail > (qty * 0.5))

    x = supp * 1009837
    x = x ^ (x >> 16)
    h = x & table_mask

    done = ~qual
    neg_one = tl.full((), -1, tl.int64)
    neg_two = tl.full((), -2, tl.int64)

    for probe in tl.static_range(0, MAX_PROBE):
        slot = (h + probe) & table_mask
        cmp_val = tl.where(done, neg_two, neg_one)
        old = tl.atomic_cas(supp_set_ptr + slot, cmp_val, supp)
        success = (old == neg_one) & (~done)
        existed = (old == supp) & (~done)
        done = done | success | existed


@triton.jit
def _select_supplier_kernel(
    s_suppkey_ptr,
    s_row_idx_ptr,
    supp_set_ptr,
    out_idx_ptr,
    count_ptr,
    n_supplier,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_supplier

    supp = tl.load(s_suppkey_ptr + offs, mask=mask, other=-1)

    x = supp * 1009837
    x = x ^ (x >> 16)
    h = x & table_mask

    found = tl.full((BLOCK_SIZE,), False, tl.int1)
    miss = tl.full((BLOCK_SIZE,), False, tl.int1)

    for probe in tl.static_range(0, MAX_PROBE):
        active = mask & (~found) & (~miss)
        slot = (h + probe) & table_mask
        cand = tl.load(supp_set_ptr + slot, mask=active, other=-1)
        match = active & (cand == supp)
        empty = active & (cand == -1)
        found = found | match
        miss = miss | empty

    found_i64 = found.to(tl.int64)
    prefix = tl.cumsum(found_i64, 0) - 1
    block_total = tl.sum(found_i64, 0)
    base = tl.atomic_add(count_ptr, block_total, sem="relaxed")
    row_idx = tl.load(s_row_idx_ptr + offs, mask=found, other=0)
    tl.store(out_idx_ptr + base + prefix, row_idx, mask=found)
'''


def _load_kernels() -> ModuleType:
    """Load Triton kernels from an on-disk module so Triton JIT can compile them."""
    tmpdir = tempfile.mkdtemp(prefix="q20_triton_")
    path = os.path.join(tmpdir, "_q20_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q20_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _next_power_of_2(x: int) -> int:
    """Return the next power of two for GPU open-addressed hash-table sizing."""
    if x <= 2:
        return 2
    return 1 << (x - 1).bit_length()


def _launch_q20_triton(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
    block_size: int = 256,
    max_probe: int = 64,
) -> torch.Tensor:
    """
    Run the numeric Q20 core on the GPU with Triton hash kernels.

    cuDF in run_query performs the string/date filters and joins. This launcher
    replaces the reference PyTorch unique/scatter/is-in sequence with GPU hash
    probes: partsupp keys are inserted once, filtered lineitem is scanned once to
    accumulate matching quantities, qualifying supplier keys are inserted into a
    set, and supplier row indices are compacted. The large lineitem phase is
    memory-bound; avoiding concatenated pair tensors, torch.unique sorting, and
    extra temporaries should reduce HBM traffic and launch overhead.
    """
    n_ps = int(ps_partkey.numel())
    n_li = int(l_partkey.numel())
    n_s = int(s_suppkey.numel())

    if n_ps == 0 or n_li == 0 or n_s == 0:
        return torch.empty((0,), dtype=s_row_idx.dtype, device=s_row_idx.device)

    device = ps_partkey.device

    ps_table_size = _next_power_of_2(max(16, n_ps * 8))
    ps_table_mask = ps_table_size - 1

    ht_idx = torch.full((ps_table_size,), -1, dtype=torch.int64, device=device)
    ht_part = torch.empty((ps_table_size,), dtype=torch.int64, device=device)
    ht_supp = torch.empty((ps_table_size,), dtype=torch.int64, device=device)
    sum_qty = torch.zeros((n_ps,), dtype=torch.float64, device=device)

    _K._insert_ps_kernel[(n_ps,)](
        ps_partkey,
        ps_suppkey,
        ht_part,
        ht_supp,
        ht_idx,
        ps_table_mask,
        MAX_PROBE=max_probe,
        num_warps=1,
    )

    _K._accum_lineitem_kernel[(triton.cdiv(n_li, block_size),)](
        l_partkey,
        l_suppkey,
        l_quantity,
        ht_part,
        ht_supp,
        ht_idx,
        sum_qty,
        n_li,
        ps_table_mask,
        BLOCK_SIZE=block_size,
        MAX_PROBE=max_probe,
        num_warps=4,
    )

    supp_table_size = _next_power_of_2(max(16, n_ps * 8))
    supp_table_mask = supp_table_size - 1
    supp_set = torch.full((supp_table_size,), -1, dtype=torch.int64, device=device)

    _K._insert_qual_supp_kernel[(n_ps,)](
        ps_suppkey,
        ps_availqty,
        sum_qty,
        supp_set,
        supp_table_mask,
        MAX_PROBE=max_probe,
        num_warps=1,
    )

    out_idx = torch.empty((n_s,), dtype=torch.int64, device=device)
    count = torch.zeros((1,), dtype=torch.int64, device=device)

    _K._select_supplier_kernel[(triton.cdiv(n_s, block_size),)](
        s_suppkey,
        s_row_idx,
        supp_set,
        out_idx,
        count,
        n_s,
        supp_table_mask,
        BLOCK_SIZE=block_size,
        MAX_PROBE=max_probe,
        num_warps=4,
    )

    selected_count = int(count.item())
    return out_idx[:selected_count]


def _query_core(
    # lineitem filtered tensors (already filtered by shipdate range in run_query)
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    # partsupp tensors (already filtered by part name in run_query)
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    # supplier (already filtered by nation = CANADA in run_query)
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Execute the fixed tensor-only Q20 core on GPU using Triton.

    The public signature matches the reference. All numeric work here is GPU-side:
    Triton kernels build/probe hash tables and perform FP64 quantity aggregation.
    This should be faster than eager PyTorch's torch.unique over stacked key pairs
    because it scans the filtered lineitem columns directly and avoids sort-like
    grouping plus large intermediate key tensors.
    """
    return _launch_q20_triton(
        l_partkey,
        l_suppkey,
        l_quantity,
        ps_partkey,
        ps_suppkey,
        ps_availqty,
        s_suppkey,
        s_row_idx,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (
                SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
            AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                               WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                               AND l_shipdate >= DATE '1994-01-01'
                               AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey AND n_name = 'CANADA'
          ORDER BY s_name;
    Description: TPC-H Q20 — finds suppliers in a specific nation who have excess inventory
                 of parts whose name starts with a given prefix during a date range.
    Params:
        name_prefix (str): prefix filter for p_name (LIKE '<prefix>%').
        nation_name (str): nation filter for supplier's nation.
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    # Convert date parameters to epoch days (int32)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # --- Filter part by p_name LIKE 'prefix%' ---
    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]

    # --- Filter partsupp by ps_partkey IN filtered_part ---
    filtered_partsupp = df_partsupp.merge(
        filtered_part, left_on="ps_partkey", right_on="p_partkey", how="inner"
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]]

    # --- Filter lineitem by shipdate range ---
    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    li_mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)
    filtered_lineitem = df_lineitem[li_mask][["l_partkey", "l_suppkey", "l_quantity"]]

    # --- Filter supplier by nation = CANADA ---
    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supplier_in_nation = df_supplier.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    ).reset_index(drop=True)

    # Add a row index column so we can recover s_name / s_address by tensor index
    supplier_in_nation = supplier_in_nation.reset_index(drop=True)
    s_row_idx_series = cudf.Series(
        cudf.core.column.arange(0, len(supplier_in_nation), dtype="int64")
    ) if False else cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert tensors
    l_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.int64)

    s_suppkey_t = torch.as_tensor(supplier_in_nation["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_row_idx_t = torch.as_tensor(s_row_idx_series.to_cupy(), dtype=torch.int64)

    selected_idx_t = _query_core(
        l_partkey_t, l_suppkey_t, l_quantity_t,
        ps_partkey_t, ps_suppkey_t, ps_availqty_t,
        s_suppkey_t, s_row_idx_t,
    )

    # Materialize result via gather on cuDF side
    selected_idx_cudf = cudf.Series(selected_idx_t)
    result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]
    result = result.sort_values("s_name").reset_index(drop=True)
    return result
