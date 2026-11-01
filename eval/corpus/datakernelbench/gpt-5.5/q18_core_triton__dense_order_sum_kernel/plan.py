import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _dense_order_sum_kernel(
    orderkey_ptr,
    quantity_ptr,
    sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    tl.atomic_add(sums_ptr + keys, qty, mask=mask, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """
    Writes Triton GPU kernels to a real on-disk Python file and imports them.

    Triton JIT compilation runs from the imported temporary module, which avoids
    validator/import issues with defining @triton.jit functions directly in the
    host module.
    """
    tmpdir = tempfile.mkdtemp(prefix="q18_triton_")
    path = os.path.join(tmpdir, "_q18_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q18_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_dense_order_sums(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    table_size: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Runs on GPU: one Triton pass atomically accumulates l_quantity into a dense
    orderkey-indexed sum array.

    This replaces the reference torch.unique(return_inverse=True) plus
    scatter_add_ path. For TPC-H order keys, the key domain is dense enough that
    a direct dense accumulator avoids the expensive global unique/sort-style
    grouping work and reduces temporary inverse-index traffic.
    """
    sums = torch.zeros(table_size, dtype=l_quantity.dtype, device=l_quantity.device)
    n = l_orderkey.numel()
    if n == 0:
        return sums

    grid = (triton.cdiv(n, block_size),)
    _K._dense_order_sum_kernel[grid](
        l_orderkey,
        l_quantity,
        sums,
        n,
        BLOCK_SIZE=block_size,
    )
    return sums


def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes per-order sums of l_quantity on GPU, then returns qualifying orders.

    GPU/Triton work: a fused dense aggregation kernel performs one coalesced scan
    of lineitem order keys and quantities, accumulating directly into a dense
    orderkey-indexed array with atomic adds.

    GPU/PyTorch tensor work: compact nonzero/qualifying dense slots into the
    output tensors expected by run_query. This is faster than the reference hot
    path for TPC-H Q18 because it avoids torch.unique(return_inverse=True), the
    large inverse tensor, and the subsequent scatter over inverse group ids.
    """
    n = l_orderkey.numel()
    if n == 0:
        empty_keys = torch.empty(0, dtype=l_orderkey.dtype, device=l_orderkey.device)
        empty_sums = torch.empty(0, dtype=l_quantity.dtype, device=l_quantity.device)
        return empty_keys, empty_sums, empty_keys

    max_key = int(torch.max(l_orderkey).item())
    table_size = max_key + 1

    sums_dense = _launch_dense_order_sums(l_orderkey, l_quantity, table_size)

    unique_keys = torch.nonzero(sums_dense != 0, as_tuple=False).flatten().to(dtype=l_orderkey.dtype)
    unique_sums = sums_dense[unique_keys]

    big_mask = unique_sums > qty_threshold
    big_keys = unique_keys[big_mask]
    big_sums = unique_sums[big_mask]

    return big_keys, big_sums, unique_keys


def run_query(tables: dict[str, cudf.DataFrame], qty_threshold: float = 300.0, limit: int = 100) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;
    Description: TPC-H Q18 — finds the top 100 customers with the largest-quantity orders.
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) in the core ---
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    # Build a cuDF DataFrame of qualifying orderkeys + their total quantity
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # Semi-join orders with big orderkeys, then join with customer
    orders_filtered = df_orders.merge(df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # Order by o_totalprice DESC, o_orderdate ASC
    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    result = cudf.DataFrame({
        "c_name": joined["c_name"].reset_index(drop=True),
        "c_custkey": joined["c_custkey"].reset_index(drop=True),
        "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
        "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
        "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
    })
    return result
