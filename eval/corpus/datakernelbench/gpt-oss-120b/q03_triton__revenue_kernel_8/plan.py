import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# ----------------------------------------------------------------------
# Triton kernels (written to a temporary .py file and loaded at import time)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Compute per‑row revenue = l_extendedprice * (1 - l_discount)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ] + [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=8, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
)
@triton.jit
def _revenue_kernel(
    price_ptr,            # float64 *
    disc_ptr,             # float64 *
    out_rev_ptr,          # float64 *
    n_elements,           # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)

    # 1.0 will be promoted to fp64 because price and disc are fp64
    rev = price * (1.0 - disc)

    tl.store(out_rev_ptr + offs, rev, mask=mask)
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

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper: launch the revenue kernel
# ----------------------------------------------------------------------
def _launch_revenue(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per‑row revenue = l_extendedprice * (1 - l_discount) using a
    fused Triton kernel. Returns a tensor of the same shape (float64).
    """
    n = l_extendedprice.numel()
    out_rev = torch.empty_like(l_extendedprice, dtype=torch.float64)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._revenue_kernel[grid](
        l_extendedprice,
        l_discount,
        out_rev,
        n,
    )
    return out_rev

# ----------------------------------------------------------------------
# Core query logic – uses the Triton‑accelerated revenue computation
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
    Execute Q3’s aggregation on GPU:
    1. Compute revenue column with a fused Triton kernel.
    2. Apply the two date predicates (filter) using a torch mask.
    3. Group‑by (l_orderkey, o_orderdate, o_shippriority) via torch.unique +
       scatter_add.
    4. Sort by revenue DESC, o_orderdate ASC and return the top‑N rows.
    """
    # 1️⃣ Compute revenue in a single fused kernel
    rev = _launch_revenue(l_extendedprice, l_discount)

    # 2️⃣ Apply row‑level predicates
    mask = (l_shipdate_days > shipdate_lo_days) & (o_orderdate_days < orderdate_hi_days)

    lk = l_orderkey[mask]
    rev_f = rev[mask]
    od = o_orderdate_days[mask]
    sp = o_shippriority[mask]

    # 3️⃣ Group‑by using torch.unique + scatter_add
    keys = torch.stack([lk.to(torch.int64), od.to(torch.int64), sp.to(torch.int64)], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(
        unique_keys.shape[0], dtype=rev_f.dtype, device=rev_f.device
    )
    revenue_sum.scatter_add_(0, inverse, rev_f)

    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)

    # 4️⃣ ORDER BY revenue DESC, o_orderdate ASC, LIMIT top_n
    n_groups = unique_keys.shape[0]
    k = min(top_n, n_groups)

    # Stable sort: secondary (orderdate) first, then primary (revenue)
    idx_sec = torch.argsort(u_od, stable=True)
    rev_sec = revenue_sum[idx_sec]
    lk_sec = u_lk[idx_sec]
    od_sec = u_od[idx_sec]
    sp_sec = u_sp[idx_sec]

    idx_pri = torch.argsort(rev_sec, descending=True, stable=True)
    rev_sorted = rev_sec[idx_pri][:k]
    lk_sorted = lk_sec[idx_pri][:k]
    od_sorted = od_sec[idx_pri][:k]
    sp_sorted = sp_sec[idx_pri][:k]

    return lk_sorted, rev_sorted, od_sorted, sp_sorted

# ----------------------------------------------------------------------
# Public API – unchanged signature
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
    Description:
        cuDF performs the three‑table joins (customer‑orders‑lineitem), which are
        already highly optimized for GPU execution.  The only computationally
        intensive part—the per‑row revenue calculation—is offloaded to a
        Triton kernel, fusing the multiplication and subtraction into a single
        memory‑bound pass.  Filtering, grouping, and final sorting remain in
        PyTorch because they involve irregular memory accesses that are already
        efficient with torch’s scatter‑add and unique kernels.
    Params:
        mktsegment (str): customer market‑segment filter.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        shipdate_lo (str): exclusive lower bound for l_shipdate.
        top_n (int): LIMIT value.
    Returns:
        cudf.DataFrame with columns: l_orderkey (int64), revenue (float64),
        o_orderdate (datetime64[s]), o_shippriority (int32).
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert literal dates to epoch‑day integers
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # 1️⃣ Filter customers by market segment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # 2️⃣ Join customer → orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # 3️⃣ Join (customer, orders) → lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # 4️⃣ Cast date columns to int32 epoch‑day representation
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    # 5️⃣ Transfer needed columns to Torch tensors (GPU)
    lk_t = torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64)
    ep_t = torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64)
    od_t = torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32)
    sd_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    sp_t = torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32)

    # 6️⃣ Execute the core GPU query
    lk_res, rev_res, od_res, sp_res = _query_core(
        lk_t,
        ep_t,
        disc_t,
        sd_t,
        od_t,
        sp_t,
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # 7️⃣ Convert orderdate back to datetime64[s] for the final DataFrame
    od_series = cudf.Series(od_res.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_res,
        "revenue": rev_res,
        "o_orderdate": od_series,
        "o_shippriority": sp_res,
    })
