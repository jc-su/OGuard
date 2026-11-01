import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

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
    key=["n_elements"]
)
@triton.jit
def mark_exists_kernel(
    l_okey_ptr, l_commit_ptr, l_receipt_ptr, exists_map_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    commit = tl.load(l_commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(l_receipt_ptr + offs, mask=mask, other=0)
    
    keep = mask & (commit < receipt)
    okey = tl.load(l_okey_ptr + offs, mask=keep, other=0)
    
    # Store 1 in the map for qualifying orderkeys
    val = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    tl.store(exists_map_ptr + okey, val, mask=keep)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["counts_out_ptr"]
)
@triton.jit
def count_priorities_kernel(
    o_okey_ptr, o_odate_ptr, o_priority_ptr, exists_map_ptr, counts_out_ptr,
    date_lo, date_hi, max_key, n_elements,
    NUM_PRIORITIES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    odate = tl.load(o_odate_ptr + offs, mask=mask, other=0)
    date_keep = mask & (odate >= date_lo) & (odate < date_hi)
    
    okey = tl.load(o_okey_ptr + offs, mask=date_keep, other=0)
    okey_valid = date_keep & (okey >= 0) & (okey <= max_key)
    
    # Check exists_map
    exists = tl.load(exists_map_ptr + okey, mask=okey_valid, other=0)
    final_keep = okey_valid & (exists == 1)
    
    prio = tl.load(o_priority_ptr + offs, mask=final_keep, other=0)
    
    # Accumulate locally and atomic add for small fixed group sizes
    for i in tl.static_range(NUM_PRIORITIES):
        p_mask = final_keep & (prio == i)
        count_i = tl.sum(tl.where(p_mask, 1, 0))
        if count_i > 0:
            tl.atomic_add(counts_out_ptr + i, count_i.to(tl.int64))
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

def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_o = o_orderkey.max().item() if o_orderkey.numel() > 0 else 0
    max_l = l_orderkey.max().item() if l_orderkey.numel() > 0 else 0
    max_key = max(max_o, max_l)
    
    exists_map = torch.zeros(max_key + 1, dtype=torch.int32, device=o_orderkey.device)
    
    n_line = l_orderkey.numel()
    if n_line > 0:
        grid1 = lambda meta: (triton.cdiv(n_line, meta["BLOCK_SIZE"]),)
        _K.mark_exists_kernel[grid1](
            l_orderkey, l_commitdate_days, l_receiptdate_days, exists_map, n_line
        )
    
    counts_out = torch.zeros(num_priorities, dtype=torch.int64, device=o_orderkey.device)
    
    n_orders = o_orderkey.numel()
    if n_orders > 0:
        grid2 = lambda meta: (triton.cdiv(n_orders, meta["BLOCK_SIZE"]),)
        _K.count_priorities_kernel[grid2](
            o_orderkey, o_orderdate_days, o_orderpriority_codes, exists_map, counts_out,
            date_lo_days, date_hi_days, max_key, n_orders,
            NUM_PRIORITIES=num_priorities
        )
    
    priority_ids = torch.arange(num_priorities, device=counts_out.device, dtype=torch.int64)
    nz = counts_out > 0
    
    return priority_ids[nz], counts_out[nz]

def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt).
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for groupby on GPU
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

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    # Map codes back to priority strings and sort
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
