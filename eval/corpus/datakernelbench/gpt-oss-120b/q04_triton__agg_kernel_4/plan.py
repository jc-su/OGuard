import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for triton.cdiv in launch helpers


# ----------------------------------------------------------------------
# Triton kernels (saved to a temporary file and loaded at import time)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel 1: set bitmap[orderkey] = 1 for lineitem rows where
#           l_commitdate < l_receiptdate
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["bitmap_ptr"],
)
@triton.jit
def _bitmap_kernel(
    l_orderkey_ptr,          # int64 *
    l_commit_ptr,            # int32 *
    l_receipt_ptr,           # int32 *
    bitmap_ptr,              # int32 *
    n_elements,              # total rows in lineitem
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    commit = tl.load(l_commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(l_receipt_ptr + offs, mask=mask, other=0)

    keep = mask & (commit < receipt)

    orderkey = tl.load(l_orderkey_ptr + offs, mask=keep, other=0)

    # write 1 (as int32) where predicate holds; atomic_max works as set‑to‑1
    tl.atomic_max(bitmap_ptr + orderkey, tl.full((1,), 1, tl.int32))


# ------------------------------------------------------------------
# Kernel 2: filter orders by date and bitmap, then count per priority
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_counts_ptr"],
)
@triton.jit
def _agg_kernel(
    o_orderkey_ptr,          # int64 *
    o_orderdate_ptr,         # int32 *
    o_priority_ptr,          # int32 *
    date_lo,                 # int32 scalar
    date_hi,                 # int32 scalar
    bitmap_ptr,              # int32 *
    out_counts_ptr,          # int64 * (size = num_priorities)
    n_elements,              # total rows in orders
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=0)
    keep_date = mask & (orderdate >= date_lo) & (orderdate < date_hi)

    orderkey = tl.load(o_orderkey_ptr + offs, mask=keep_date, other=0)

    bitmap_val = tl.load(bitmap_ptr + orderkey, mask=keep_date, other=0)
    keep = keep_date & (bitmap_val > 0)

    priority = tl.load(o_priority_ptr + offs, mask=keep, other=0)

    inc = keep.to(tl.int64)   # 1 where matched, 0 otherwise
    tl.atomic_add(out_counts_ptr + priority, inc)
'''

def _load_kernels() -> object:
    """Write the Triton source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


# Load kernels once at module import
_K = _load_kernels()


# ----------------------------------------------------------------------
# Launch helpers (torch tensors are required for Triton's reset logic)
# ----------------------------------------------------------------------
def _launch_bitmap(
    l_orderkey: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    bitmap: torch.Tensor,
) -> None:
    n = l_orderkey.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._bitmap_kernel[grid](
        l_orderkey,
        l_commitdate,
        l_receiptdate,
        bitmap,
        n,
    )


def _launch_agg(
    o_orderkey: torch.Tensor,
    o_orderdate: torch.Tensor,
    o_priority: torch.Tensor,
    date_lo: int,
    date_hi: int,
    bitmap: torch.Tensor,
    out_counts: torch.Tensor,
) -> None:
    n = o_orderkey.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._agg_kernel[grid](
        o_orderkey,
        o_orderdate,
        o_priority,
        date_lo,
        date_hi,
        bitmap,
        out_counts,
        n,
    )


# ----------------------------------------------------------------------
# Core fused GPU pipeline
# ----------------------------------------------------------------------
def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_priority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes the fused TPC‑H Q4 logic completely on the GPU.

    Returns:
        priority_ids (torch.int64), counts (torch.int64)
    """
    device = o_orderkey.device

    # Bitmap size = max orderkey + 1 (fits in int32)
    max_key = torch.max(torch.max(o_orderkey), torch.max(l_orderkey)).item()
    bitmap = torch.zeros(max_key + 1, dtype=torch.int32, device=device)

    # 1) Build bitmap of orderkeys that have a qualifying lineitem
    _launch_bitmap(l_orderkey, l_commitdate_days, l_receiptdate_days, bitmap)

    # 2) Aggregate per priority
    out_counts = torch.zeros(num_priorities, dtype=torch.int64, device=device)
    _launch_agg(
        o_orderkey,
        o_orderdate_days,
        o_priority_codes,
        date_lo_days,
        date_hi_days,
        bitmap,
        out_counts,
    )

    # 3) Keep only non‑zero groups
    nz = out_counts > 0
    priority_ids = torch.arange(num_priorities, device=device, dtype=torch.int64)[nz]
    counts = out_counts[nz]
    return priority_ids, counts


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count
         FROM orders
         WHERE o_orderdate >= CAST('1993-07-01' AS date)
           AND o_orderdate < CAST('1993-10-01' AS date)
           AND EXISTS (SELECT *
                       FROM lineitem
                       WHERE l_orderkey = o_orderkey
                         AND l_commitdate < l_receiptdate)
         GROUP BY o_orderpriority
         ORDER BY o_orderpriority;

    Description:
        * The heavy computation (date filtering, semi‑join, GROUP BY) is performed
          in two fused Triton kernels, drastically reducing HBM traffic compared
          to the reference eager/Torch‑compile pipelines.
        * cuDF is used only for lightweight metadata handling (categorical
          encoding of priority) and for constructing the final result frame.

    Params:
        date_lo (str): inclusive lower bound for o_orderdate (YYYY‑MM‑DD).
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # ------------------------------------------------------------------
    # Convert date bounds to int32 epoch‑days (no timezone information)
    # ------------------------------------------------------------------
    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # Encode order priority as categorical integer codes (small fixed group)
    # ------------------------------------------------------------------
    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int32
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    # ------------------------------------------------------------------
    # Convert columns to torch tensors (GPU)
    # ------------------------------------------------------------------
    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )

    # ------------------------------------------------------------------
    # Execute fused GPU pipeline
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build final cuDF DataFrame (small result, OK on CPU)
    # ------------------------------------------------------------------
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame(
        {
            "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
            "order_count": counts_t.cpu(),
        }
    )
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
