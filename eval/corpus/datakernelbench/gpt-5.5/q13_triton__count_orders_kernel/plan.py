import os
import re
import tempfile
import importlib.util
import linecache
from typing import Any

import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _count_orders_kernel(
    o_custkey_ptr,
    excluded_ptr,
    counts_ptr,
    n_orders: tl.constexpr,
    counts_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    cust = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    excluded = tl.load(excluded_ptr + offs, mask=mask, other=1)

    keep = mask & (excluded == 0) & (cust >= 0) & (cust < counts_len)
    tl.atomic_add(counts_ptr + cust, 1, sem="relaxed", mask=keep)


@triton.jit
def _hist_partial_kernel(
    c_custkey_ptr,
    counts_ptr,
    partial_hist_ptr,
    overflow_ptr,
    n_customers: tl.constexpr,
    counts_len: tl.constexpr,
    SMALL_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cust = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    valid_cust = mask & (cust >= 0) & (cust < counts_len)

    cnt = tl.load(counts_ptr + cust, mask=valid_cust, other=SMALL_BINS)
    in_small = valid_cust & (cnt >= 0) & (cnt < SMALL_BINS)

    hist_idx = pid * SMALL_BINS + cnt
    tl.atomic_add(partial_hist_ptr + hist_idx, 1, sem="relaxed", mask=in_small)

    overflow_lanes = valid_cust & ((cnt < 0) | (cnt >= SMALL_BINS))
    overflow_sum = tl.sum(tl.where(overflow_lanes, 1, 0))
    tl.atomic_add(overflow_ptr, overflow_sum, sem="relaxed")


@triton.jit
def _reduce_partial_hist_kernel(
    partial_hist_ptr,
    hist_ptr,
    num_blocks: tl.constexpr,
    SMALL_BINS: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    bin_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    offs = chunk_id * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    mask = offs < num_blocks

    vals = tl.load(partial_hist_ptr + offs * SMALL_BINS + bin_id, mask=mask, other=0)
    total = tl.sum(vals)
    tl.atomic_add(hist_ptr + bin_id, total, sem="relaxed")


@triton.jit
def _hist_full_kernel(
    c_custkey_ptr,
    counts_ptr,
    hist_ptr,
    n_customers: tl.constexpr,
    counts_len: tl.constexpr,
    hist_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cust = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    valid_cust = mask & (cust >= 0) & (cust < counts_len)

    cnt = tl.load(counts_ptr + cust, mask=valid_cust, other=0)
    valid_cnt = valid_cust & (cnt >= 0) & (cnt < hist_len)

    tl.atomic_add(hist_ptr + cnt, 1, sem="relaxed", mask=valid_cnt)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q13_triton_")
    path = os.path.join(tmpdir, "_q13_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q13_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _sql_like_to_regex(pattern: str) -> str:
    """Convert a simple SQL LIKE pattern to a regex for the cuDF string pre-filter."""
    out: list[str] = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return "^" + "".join(out) + "$"


def _sort_distribution(c_count: torch.Tensor, custdist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort the small final distribution on GPU with PyTorch tensor primitives.

    The expensive scans/aggregations have already run in Triton.  This step only sorts the
    usually tiny set of non-empty count buckets to match ORDER BY custdist DESC, c_count DESC.
    """
    order1 = torch.argsort(c_count, descending=True, stable=True)
    c_count = c_count[order1]
    custdist = custdist[order1]
    order2 = torch.argsort(custdist, descending=True, stable=True)
    return c_count[order2].contiguous(), custdist[order2].contiguous()


def _query_core(
    c_custkey: torch.Tensor,
    o_custkey: torch.Tensor,
    order_is_excluded: torch.Tensor,
    num_customers: int,
    small_bins: int = 256,
    count_block_size: int = 1024,
    customer_block_size: int = 1024,
    reduce_block_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the numeric hot path on GPU with Triton kernels.

    Triton first fuses the NOT-LIKE boolean result with the order customer-key scan and
    atomically builds per-customer order counts, avoiding a materialized filtered orders
    table.  A second Triton stage builds the customer-count distribution using a fast
    small-bin partial histogram for normal TPC-H Q13 cardinalities; if an unusually large
    per-customer count appears, it falls back to a full exact histogram.
    """
    n_orders = int(o_custkey.numel())
    n_customers = int(c_custkey.numel())
    counts_len = int(num_customers) + 1

    if n_customers == 0:
        empty = torch.empty(0, dtype=torch.int64, device=c_custkey.device)
        return empty, empty

    counts = torch.zeros(counts_len, dtype=torch.int32, device=c_custkey.device)

    if n_orders > 0:
        order_grid = (triton.cdiv(n_orders, count_block_size),)
        _K._count_orders_kernel[order_grid](
            o_custkey,
            order_is_excluded,
            counts,
            n_orders,
            counts_len,
            BLOCK_SIZE=count_block_size,
        )

    num_customer_blocks = triton.cdiv(n_customers, customer_block_size)
    partial_hist = torch.zeros(
        (num_customer_blocks, small_bins),
        dtype=torch.int64,
        device=c_custkey.device,
    )
    hist_small = torch.zeros(small_bins, dtype=torch.int64, device=c_custkey.device)
    overflow = torch.zeros(1, dtype=torch.int64, device=c_custkey.device)

    hist_grid = (num_customer_blocks,)
    _K._hist_partial_kernel[hist_grid](
        c_custkey,
        counts,
        partial_hist,
        overflow,
        n_customers,
        counts_len,
        SMALL_BINS=small_bins,
        BLOCK_SIZE=customer_block_size,
    )

    reduce_grid = (small_bins, triton.cdiv(num_customer_blocks, reduce_block_size))
    _K._reduce_partial_hist_kernel[reduce_grid](
        partial_hist,
        hist_small,
        num_customer_blocks,
        SMALL_BINS=small_bins,
        REDUCE_BLOCK=reduce_block_size,
    )

    if int(overflow.item()) == 0:
        c_count = torch.nonzero(hist_small, as_tuple=False).flatten().to(torch.int64)
        custdist = hist_small[c_count]
        return _sort_distribution(c_count, custdist)

    hist_len = n_orders + 1
    hist_full = torch.zeros(hist_len, dtype=torch.int64, device=c_custkey.device)
    full_grid = (num_customer_blocks,)
    _K._hist_full_kernel[full_grid](
        c_custkey,
        counts,
        hist_full,
        n_customers,
        counts_len,
        hist_len,
        BLOCK_SIZE=customer_block_size,
    )
    c_count = torch.nonzero(hist_full, as_tuple=False).flatten().to(torch.int64)
    custdist = hist_full[c_count]
    return _sort_distribution(c_count, custdist)


def run_query(
    tables: dict[str, cudf.DataFrame],
    comment_pattern: str = "%special%requests%",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey
             AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;

    Description:
        TPC-H Q13 customer distribution by number of orders whose comments do not match
        the special-requests pattern.  cuDF performs the string regex predicate because
        comments are variable-length strings.  The numeric hot path then runs in Triton:
        one fused scan consumes the cuDF boolean predicate and o_custkey to build
        per-customer counts without materializing filtered_orders, followed by Triton
        histogram kernels to compute custdist.  This reduces HBM traffic and avoids the
        reference PyTorch scatter/gather/unique pipeline over large temporary tensors.

    Params:
        tables: Dictionary containing cuDF DataFrames "customer" and "orders".
        comment_pattern: SQL LIKE pattern to exclude from orders; the default matches
            the query predicate '%special%requests%'.

    Returns:
        cuDF DataFrame with columns ["c_count", "custdist"] and int64 values.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    if comment_pattern == "%special%requests%":
        regex = "special.*requests"
    else:
        regex = _sql_like_to_regex(comment_pattern)

    order_is_excluded_s = df_orders["o_comment"].str.contains(regex, regex=True).fillna(True)

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(df_orders["o_custkey"].to_cupy(), dtype=torch.int64)
    order_is_excluded_t = torch.as_tensor(order_is_excluded_s.to_cupy(), dtype=torch.bool)

    if len(df_customer) == 0:
        empty = torch.empty(0, dtype=torch.int64, device=o_custkey_t.device)
        return cudf.DataFrame({"c_count": empty, "custdist": empty})

    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(
        c_custkey_t,
        o_custkey_t,
        order_is_excluded_t,
        num_customers,
    )

    return cudf.DataFrame(
        {
            "c_count": c_count_t,
            "custdist": custdist_t,
        }
    )
