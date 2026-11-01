import os
import tempfile
import importlib.util
import linecache
from typing import Any

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _fill_zero_f64_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.zeros((BLOCK_SIZE,), dtype=tl.float64), mask=mask)


@triton.jit
def _sum_quantity_by_orderkey_kernel(
    orderkey_ptr,
    quantity_ptr,
    sum_by_key_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    tl.atomic_add(sum_by_key_ptr + keys, qty, sem="relaxed", mask=mask)


@triton.jit
def _mark_orders_over_threshold_kernel(
    orderkey_ptr,
    sum_by_key_ptr,
    threshold_ptr,
    out_sum_ptr,
    out_keep_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    threshold = tl.load(threshold_ptr)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    sums = tl.load(sum_by_key_ptr + keys, mask=mask, other=0.0)
    keep = sums > threshold

    tl.store(out_sum_ptr + offs, sums, mask=mask)
    tl.store(out_keep_ptr + offs, keep, mask=mask)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q18_triton_")
    path = os.path.join(tmpdir, "_q18_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q18_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _triton_zero_f64(out: torch.Tensor, block_size: int = 1024) -> None:
    """
    GPU/Triton helper: fills a float64 tensor with zeros.

    This keeps the direct-address aggregation table initialization in the Triton
    hot path rather than relying on a framework fallback memset-like operation.
    """
    n = out.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, block_size),)
    _K._fill_zero_f64_kernel[grid](out, n, BLOCK_SIZE=block_size, num_warps=4)


def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    o_orderkey: torch.Tensor,
    qty_threshold: float,
    max_orderkey: int,
    sum_block_size: int = 256,
    scan_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU/Triton core for TPC-H Q18's large numeric work.

    It replaces the reference torch.unique + scatter_add groupby with a
    direct-address GPU aggregation table indexed by orderkey:
      1. Triton zeroes sum_by_key.
      2. Triton scans lineitem once and atomically accumulates l_quantity by key.
      3. Triton scans orders once and emits per-order sum plus a boolean HAVING mask.

    This is faster for TPC-H order keys because it avoids sorting/unique over about
    60M lineitem rows and avoids materializing a separate big-order join table in
    the hot path. The work is mostly memory-bound; kernels use coalesced scans and
    a compact direct-address table that fits comfortably in H100 HBM.
    """
    n_slots = int(max_orderkey) + 1
    device = l_quantity.device

    sum_by_key = torch.empty(n_slots, dtype=torch.float64, device=device)
    _triton_zero_f64(sum_by_key, block_size=scan_block_size)

    n_lineitem = l_orderkey.numel()
    if n_lineitem:
        grid_sum = (triton.cdiv(n_lineitem, sum_block_size),)
        _K._sum_quantity_by_orderkey_kernel[grid_sum](
            l_orderkey,
            l_quantity,
            sum_by_key,
            n_lineitem,
            BLOCK_SIZE=sum_block_size,
            num_warps=4,
        )

    n_orders = o_orderkey.numel()
    order_sums = torch.empty(n_orders, dtype=torch.float64, device=device)
    keep = torch.empty(n_orders, dtype=torch.bool, device=device)
    if n_orders:
        threshold = torch.tensor([qty_threshold], dtype=torch.float64, device=device)
        grid_mark = (triton.cdiv(n_orders, scan_block_size),)
        _K._mark_orders_over_threshold_kernel[grid_mark](
            o_orderkey,
            sum_by_key,
            threshold,
            order_sums,
            keep,
            n_orders,
            BLOCK_SIZE=scan_block_size,
            num_warps=4,
        )

    return keep, order_sums


def run_query(
    tables: dict[str, cudf.DataFrame],
    qty_threshold: float = 300.0,
    limit: int = 100,
    sum_block_size: int = 256,
    scan_block_size: int = 1024,
) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey
          FROM lineitem
          GROUP BY l_orderkey
          HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;

    Description:
      TPC-H Q18 — finds customers with the largest-quantity orders.

      The bulk numeric hot path runs on GPU in Triton: a direct-address fp64
      aggregation table computes sum(l_quantity) per orderkey, then a second
      Triton pass marks orders whose sum exceeds qty_threshold. cuDF handles the
      remaining relational/string-heavy work: boolean filtering of orders, joining
      to customer names, sorting, limiting, and formatting the result.

      Compared with the reference eager PyTorch torch.unique/scatter_add pipeline,
      this avoids a large unique/sort-style grouping step and avoids a hash semi-
      join between orders and a materialized big-order table. For the memory-bound
      scan over lineitem, the Triton path performs simple coalesced reads plus
      direct atomic accumulation, which is well suited to H100 bandwidth.

    Params:
      tables (dict[str, cudf.DataFrame]): input TPC-H tables containing customer,
        orders, and lineitem.
      qty_threshold (float): HAVING threshold for sum(l_quantity).
      limit (int): number of rows to return after ordering.
      sum_block_size (int): Triton tile size for the atomic lineitem aggregation.
      scan_block_size (int): Triton tile size for table initialization and order
        threshold marking.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)

    if len(df_orders) == 0:
        joined = cudf.DataFrame(
            {
                "c_name": cudf.Series([], dtype=df_customer["c_name"].dtype),
                "c_custkey": cudf.Series([], dtype=df_customer["c_custkey"].dtype),
                "o_orderkey": cudf.Series([], dtype=df_orders["o_orderkey"].dtype),
                "o_orderdate": cudf.Series([], dtype=df_orders["o_orderdate"].dtype),
                "o_totalprice": cudf.Series([], dtype=df_orders["o_totalprice"].dtype),
                "sum_qty": cudf.Series([], dtype="float64"),
            }
        )
    else:
        max_orderkey = max(
            int(df_orders["o_orderkey"].max()),
            int(df_lineitem["l_orderkey"].max()) if len(df_lineitem) else 0,
        )

        keep_t, order_sums_t = _query_core(
            l_orderkey_t,
            l_quantity_t,
            o_orderkey_t,
            float(qty_threshold),
            max_orderkey,
            sum_block_size=sum_block_size,
            scan_block_size=scan_block_size,
        )

        keep_s = cudf.Series(keep_t)
        orders_aug = cudf.DataFrame(
            {
                "o_orderkey": df_orders["o_orderkey"].reset_index(drop=True),
                "o_custkey": df_orders["o_custkey"].reset_index(drop=True),
                "o_orderdate": df_orders["o_orderdate"].reset_index(drop=True),
                "o_totalprice": df_orders["o_totalprice"].reset_index(drop=True),
                "sum_qty": cudf.Series(order_sums_t),
            }
        )

        orders_filtered = orders_aug[keep_s]
        joined = orders_filtered.merge(
            df_customer[["c_custkey", "c_name"]],
            left_on="o_custkey",
            right_on="c_custkey",
            how="inner",
        )

    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"],
        ascending=[False, True],
    ).head(limit)

    result = cudf.DataFrame(
        {
            "c_name": joined["c_name"].reset_index(drop=True),
            "c_custkey": joined["c_custkey"].reset_index(drop=True),
            "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
            "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
            "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
            "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
        }
    )
    return result
