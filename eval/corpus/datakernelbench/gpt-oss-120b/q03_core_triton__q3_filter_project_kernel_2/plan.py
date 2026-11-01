import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# ----------------------------------------------------------------------
# Triton kernel source (written to a temporary file and imported)
# ----------------------------------------------------------------------
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
    reset_to_zero=["cnt_ptr"],
)
@triton.jit
def _q3_filter_project_kernel(
    # input columns
    lk_ptr,                 # int64 *
    price_ptr,              # float64 *
    discount_ptr,           # float64 *
    l_ship_ptr,             # int32 *
    o_date_ptr,             # int32 *
    sp_ptr,                 # int32 *
    # scalar predicates
    shipdate_lo,            # int32
    orderdate_hi,           # int32
    # output buffers (pre‑allocated, size >= input rows)
    out_lk_ptr,             # int64 *
    out_rev_ptr,            # float64 *
    out_od_ptr,             # int32 *
    out_sp_ptr,             # int32 *
    # counter for total kept rows
    cnt_ptr,                # int64 *
    # runtime
    n_elements,             # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns needed for predicate
    l_ship = tl.load(l_ship_ptr + offs, mask=mask, other=0)
    o_date = tl.load(o_date_ptr + offs, mask=mask, other=0)

    # Predicate: l_shipdate > shipdate_lo AND o_orderdate < orderdate_hi
    keep = mask & (l_ship > shipdate_lo) & (o_date < orderdate_hi)

    # Compute revenue for kept rows
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    rev = price * (1.0 - discount)                     # float64

    # ---------- compact write index ----------
    keep_i = tl.where(keep, 1, 0).to(tl.int64)        # int64 flags per lane
    pref = tl.cumsum(keep_i, axis=0) - keep_i         # exclusive prefix sum
    block_sum = tl.sum(keep_i, axis=0)                # total kept in this block

    # Lane 0 atomically reserves space and receives previous total
    prev_total = tl.atomic_add(cnt_ptr, block_sum)    # scalar int64
    # Broadcast by adding scalar to vector prefix sum
    write_idx = prev_total + pref                     # global write positions

    # Load remaining columns for kept rows
    lk = tl.load(lk_ptr + offs, mask=keep, other=0)
    od = tl.load(o_date_ptr + offs, mask=keep, other=0)
    sp = tl.load(sp_ptr + offs, mask=keep, other=0)

    # Store compacted results
    tl.store(out_lk_ptr + write_idx, lk, mask=keep)
    tl.store(out_rev_ptr + write_idx, rev, mask=keep)
    tl.store(out_od_ptr + write_idx, od, mask=keep)
    tl.store(out_sp_ptr + write_idx, sp, mask=keep)
'''

def _load_kernels():
    """Write the Triton kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q3_triton_")
    path = os.path.join(tmpdir, "_q3_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q3_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper that runs the fused filter+project kernel
# ----------------------------------------------------------------------
def _launch_filter_project(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Executes a single Triton kernel that:
        * applies the two date predicates,
        * computes revenue = l_extendedprice * (1 - l_discount),
        * writes surviving rows to compact output buffers.
    Returns compacted tensors and the number of rows kept.
    """
    n = l_orderkey.numel()
    device = l_orderkey.device

    # Allocate maximum‑size output buffers
    out_lk = torch.empty(n, dtype=torch.int64, device=device)
    out_rev = torch.empty(n, dtype=torch.float64, device=device)
    out_od = torch.empty(n, dtype=torch.int32, device=device)
    out_sp = torch.empty(n, dtype=torch.int32, device=device)

    # Counter for total kept rows (device scalar)
    cnt = torch.zeros(1, dtype=torch.int64, device=device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # BLOCK_SIZE is a compile‑time constexpr; no need to pass explicitly.
    _K._q3_filter_project_kernel[grid](
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days,
        out_lk,
        out_rev,
        out_od,
        out_sp,
        cnt,
        n,
    )

    kept = int(cnt.item())
    return (
        out_lk[:kept],
        out_rev[:kept],
        out_od[:kept],
        out_sp[:kept],
        kept,
    )

# ----------------------------------------------------------------------
# Core compute function – replaces the reference eager implementation
# ----------------------------------------------------------------------
def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton‑accelerated hot path for TPC‑H Q3.
    Steps:
        1. Fused filter + projection (single memory pass).
        2. Group‑by aggregation via torch.unique + scatter_add.
        3. Sort by revenue (desc) then orderdate (asc) and limit to top_n.
    Returns (l_orderkey, revenue, o_orderdate, o_shippriority) tensors.
    """
    # 1️⃣ fused filter & projection
    lk, rev, od, sp, _ = _launch_filter_project(
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days,
    )

    if lk.numel() == 0:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=lk.device)
        empty_f64 = torch.empty(0, dtype=torch.float64, device=lk.device)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=lk.device)
        return empty_i64, empty_f64, empty_i32, empty_i32

    # 2️⃣ group‑by (l_orderkey, o_orderdate, o_shippriority)
    keys = torch.stack(
        [lk, od.to(torch.int64), sp.to(torch.int64)], dim=1
    )  # shape (N,3)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(
        unique_keys.shape[0], dtype=rev.dtype, device=rev.device
    )
    revenue_sum.scatter_add_(0, inverse, rev)

    u_lk = unique_keys[:, 0]                     # int64
    u_od = unique_keys[:, 1].to(torch.int32)    # back to int32
    u_sp = unique_keys[:, 2].to(torch.int32)

    # 3️⃣ order by revenue DESC, then orderdate ASC, limit top_n
    n_unique = unique_keys.shape[0]
    k = min(top_n, n_unique)

    # secondary sort (orderdate ASC) – stable
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]

    # primary sort (revenue DESC) – stable
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)

    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]

    return lk_sorted, rev_sorted, od_sorted, sp_sorted

# ----------------------------------------------------------------------
# run_query – unchanged from the reference plan (only imports added)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 - Shipping Priority. Joins customer, orders, lineitem;
                 filters by market segment and date predicates; aggregates revenue
                 per (orderkey, orderdate, shippriority); returns top-N by revenue.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert date literals to epoch‑day integers
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # Filter customers by market segment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # Join customer → orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # Join (customer, orders) → lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert datetime columns to int32 day counts
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
