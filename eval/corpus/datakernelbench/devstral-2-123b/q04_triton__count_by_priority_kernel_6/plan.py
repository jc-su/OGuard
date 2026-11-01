import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
from typing import Tuple

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
    reset_to_zero=["counts_ptr"],
)
@triton.jit
def _count_by_priority_kernel(
    priority_ptr,
    counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    priority = tl.load(priority_ptr + offs, mask=mask, other=-1)
    tl.atomic_add(counts_ptr + priority, tl.where(mask, 1, 0))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt). The GPU path uses Triton for counting priority groups after cuDF handles the heavy EXISTS semijoin; this reduces Python round-trips and HBM traffic compared to eager Torch counts.
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    # Filter lineitems where commitdate < receiptdate
    line_mask = l_commitdate_days_t < l_receiptdate_days_t
    qualifying_l_orderkey = l_orderkey_t[line_mask]

    # Unique set of qualifying order keys (EXISTS)
    unique_keys = torch.unique(qualifying_l_orderkey)

    # Filter orders by date range
    date_mask = (o_orderdate_days_t >= date_lo_days) & (o_orderdate_days_t < date_hi_days)
    filt_orderkey = o_orderkey_t[date_mask]
    filt_priority = o_priority_codes_t[date_mask]

    # Check which orders exist in the qualifying set
    exists_mask = torch.isin(filt_orderkey, unique_keys)
    final_priority = filt_priority[exists_mask]

    # Tritton group-by counting
    n_elements = final_priority.numel()
    counts = torch.zeros(num_priorities, dtype=torch.int64, device=final_priority.device)
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _K._count_by_priority_kernel[grid](
            final_priority, counts, n_elements,
        )

    # Return non-zero groups
    nz = counts > 0
    priority_ids_t = torch.argsort(torch.arange(num_priorities, device=counts.device)[nz])
    counts_t = counts[nz]

    # Map back to priority strings and sort by priority
    priority_strs = [priority_categories[i.item()] for i in torch.argsort(torch.arange(num_priorities)[nz.to(torch.bool).cpu()])]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t.cpu(),
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
