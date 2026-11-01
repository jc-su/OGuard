import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import triton

# ----------------------------------------------------------------------
# Triton kernel source (written to a temporary file at import time)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configurations suitable for H100
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["sums_ptr"],
)
@triton.jit
def _agg_kernel(
    orderkey_ptr,          # int64 *
    qty_ptr,               # fp64 *
    sums_ptr,              # fp64 * (size = max_key+1)
    n_elements,            # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load only valid lanes; OOB lanes get dummy values (won't affect result)
    order = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Atomic accumulation per orderkey
    # `order` is int64, valid range is [0, max_key]
    tl.atomic_add(sums_ptr + order, qty)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q18_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load the compiled kernels module (available as `_K`).
_K = _load_kernels()

# ----------------------------------------------------------------------
# Core GPU aggregation (fused histogram‑style aggregation)
# ----------------------------------------------------------------------
def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused aggregation of ``l_quantity`` per ``l_orderkey`` using a
    Triton kernel. The kernel performs a single pass over the two columns,
    atomically adding each quantity to a dense per‑order accumulator.
    This eliminates the two‑step ``torch.unique`` + ``scatter_add`` pipeline
    in the reference implementation and reduces memory traffic.

    After the kernel finishes, we:
      * identify ``big_keys`` where the summed quantity exceeds ``qty_threshold``;
      * collect the corresponding ``big_sums``;
      * produce ``unique_keys`` – the set of orderkeys that actually appear
        (non‑zero accumulators).

    Returns
    -------
    big_keys : torch.Tensor[int64]
        Order keys whose total quantity is > ``qty_threshold``.
    big_sums : torch.Tensor[float64]
        Corresponding summed quantities.
    unique_keys : torch.Tensor[int64]
        All distinct order keys present in ``l_orderkey``.
    """
    device = l_quantity.device

    # Determine the dense size needed for the per‑order accumulator.
    max_key = int(l_orderkey.max().item())
    sums = torch.zeros(max_key + 1, dtype=l_quantity.dtype, device=device)

    n = l_orderkey.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # Launch the fused aggregation kernel.
    _K._agg_kernel[grid](
        l_orderkey,
        l_quantity,
        sums,
        n,
    )

    # Post‑process the dense accumulator.
    # 1) Orders exceeding the threshold.
    big_mask = sums > qty_threshold
    big_keys = torch.nonzero(big_mask, as_tuple=False).squeeze(1)
    big_sums = sums[big_mask]

    # 2) All distinct order keys (non‑zero sums).
    unique_mask = sums != 0.0
    unique_keys = torch.nonzero(unique_mask, as_tuple=False).squeeze(1)

    return big_keys, big_sums, unique_keys

# ----------------------------------------------------------------------
# End‑to‑end query (unchanged from reference; only the hot path is new)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    qty_threshold: float = 300.0,
    limit: int = 100,
) -> cudf.DataFrame:
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
