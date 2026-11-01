"""
TPC-H Q17 optimized with Triton fusion.

SQL:
    SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
    FROM lineitem, part
    WHERE p_partkey = l_partkey
      AND p_brand = 'Brand#23'
      AND p_container = 'MED BOX'
      AND l_quantity < (
          SELECT 0.2 * avg(l_quantity)
          FROM lineitem
          WHERE l_partkey = p_partkey);

Description:
    * `run_query` uses cuDF to filter the `part` table on the string predicates.
    * Required columns are transferred to torch tensors on the GPU.
    * Per‑part average quantity is computed with a single scatter‑add pass.
    * The final filter (`l_quantity < 0.2 * avg(l_quantity)`) and the global sum are
      performed in one fused Triton kernel, eliminating intermediate mask tensors
      and extra torch kernels. This reduces HBM traffic and kernel launch overhead,
      providing a faster memory‑bound execution on an H100.
Params:
    tables (dict[str, cudf.DataFrame]): Input tables.
    brand_val (str): Literal filter for `p_brand` (default "Brand#23").
    container_val (str): Literal filter for `p_container` (default "MED BOX").
Returns:
    float: Query result (sum(l_extendedprice) / 7.0).
"""

import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import triton


# ----------------------------------------------------------------------------- #
# Triton kernel source (written to a temporary file at import time)
# ----------------------------------------------------------------------------- #
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune a range of block sizes for Hopper (H100)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _final_sum_kernel(
    price_ptr,          # float64 *
    quantity_ptr,       # float64 *
    idx_ptr,            # int64 *
    part_keep_ptr,      # uint8 *
    thresh_ptr,         # float64 *
    out_ptr,            # float64 *
    n_elements,         # i64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    # Load part membership flag and part index
    part_keep = tl.load(part_keep_ptr + offs, mask=mask, other=0)
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)

    # Load per‑part threshold only for rows that belong to a filtered part
    thresh = tl.load(thresh_ptr + idx, mask=(part_keep == 1) & mask, other=0.0)

    # Row passes if it belongs to a filtered part and quantity < threshold
    cond = (part_keep == 1) & (quantity < thresh)

    contribution = tl.where(cond, price, 0.0)

    # Block‑wise reduction followed by atomic add to global result
    tl.atomic_add(out_ptr, tl.sum(contribution))
'''
# ----------------------------------------------------------------------------- #
# Load the kernel source into a temporary Python module
# ----------------------------------------------------------------------------- #
def _load_kernels() -> object:
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_q17_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q17_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_K = _load_kernels()


# ----------------------------------------------------------------------------- #
# GPU launch helper: runs the fused final‑sum kernel
# ----------------------------------------------------------------------------- #
def _launch_final_sum(
    price: torch.Tensor,
    quantity: torch.Tensor,
    idx: torch.Tensor,
    part_keep: torch.Tensor,
    thresh: torch.Tensor,
) -> torch.Tensor:
    """
    Executes the fused Triton kernel that applies the per‑part quantity filter
    and accumulates the sum of `price`. Returns a scalar tensor on the same device.
    """
    n = price.numel()
    out = torch.zeros(1, dtype=torch.float64, device=price.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # No explicit BLOCK_SIZE argument; autotuner decides it.
    _K._final_sum_kernel[grid](
        price,
        quantity,
        idx,
        part_keep,
        thresh,
        out,
        n,
    )
    return out.squeeze(0)


# ----------------------------------------------------------------------------- #
# Core computation (hot path)
# ----------------------------------------------------------------------------- #
def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Q17 using a fused final‑sum Triton kernel.
    Steps:
        1. Sort filtered part keys.
        2. Map each lineitem row to a part index via searchsorted.
        3. Scatter‑add per‑part sum and count of quantity.
        4. Derive per‑part threshold = 0.2 * avg_quantity.
        5. Fuse the final filter and aggregation in a single Triton launch.
    Returns:
        Tensor containing the final sum (float64) before division by 7.
    """
    # 1. Sort part keys for binary search
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()

    # 2. Searchsorted gives candidate index; validate exact membership
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)
    in_part = sorted_keys[idx_clamped] == l_partkey  # bool tensor

    # 3. Compute per‑part quantity sum and count using scatter_add
    sel_idx = idx_clamped[in_part]
    sel_qty = l_quantity[in_part]

    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_quantity.device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_quantity.device)

    sum_qty.scatter_add_(0, sel_idx, sel_qty)
    cnt_qty.scatter_add_(0, sel_idx, torch.ones_like(sel_qty, dtype=torch.float64))

    # 4. Per‑part average and threshold (0.2 * avg)
    avg_qty = torch.where(cnt_qty > 0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))
    thresh = 0.2 * avg_qty  # float64 tensor of shape (n_parts,)

    # 5. Prepare tensors for the fused kernel
    part_keep_uint8 = in_part.to(torch.uint8)  # 0/1 flag for each row
    total = _launch_final_sum(
        price=l_extendedprice,
        quantity=l_quantity,
        idx=idx_clamped,
        part_keep=part_keep_uint8,
        thresh=thresh,
    )
    return total / 7.0


# ----------------------------------------------------------------------------- #
# Public API (signature must match the reference)
# ----------------------------------------------------------------------------- #
def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (
              SELECT 0.2 * avg(l_quantity)
              FROM lineitem
              WHERE l_partkey = p_partkey);
    Description:
        * Filters `part` using cuDF (fast string predicates).
        * Transfers required columns of `lineitem` and the filtered part keys to
          torch tensors on the GPU.
        * Calls `_query_core`, which performs a fused GPU computation with Triton.
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables.
        brand_val (str): Filter literal for `p_brand`.
        container_val (str): Filter literal for `p_container`.
    Returns:
        float: Query result.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # ----- Part filter (cuDF) -----
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    # ----- Transfer columns to GPU tensors -----
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)

    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    # ----- Execute core computation -----
    result_tensor = _query_core(
        l_partkey_t,
        l_quantity_t,
        l_extendedprice_t,
        part_keys_t,
    )
    return float(result_tensor.item())
