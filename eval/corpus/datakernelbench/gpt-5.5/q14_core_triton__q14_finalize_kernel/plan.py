import os
import tempfile
import importlib.util
import linecache
from typing import Any

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q14_scan_kernel(
    price_ptr,
    discount_ptr,
    shipdate_ptr,
    promo_flag_ptr,
    partials_ptr,
    partial_stride,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=in_bounds, other=date_hi)
    keep = in_bounds & (shipdate >= date_lo) & (shipdate < date_hi)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo_flag = tl.load(promo_flag_ptr + offs, mask=keep, other=0)

    revenue = price * (1.0 - discount)
    promo_revenue = tl.where(promo_flag, revenue, 0.0)

    tl.store(partials_ptr + pid, tl.sum(revenue))
    tl.store(partials_ptr + partial_stride + pid, tl.sum(promo_revenue))


@triton.jit
def _q14_finalize_kernel(
    partials_ptr,
    out_ptr,
    partial_stride,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_blocks

    total_vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
    promo_vals = tl.load(partials_ptr + partial_stride + offs, mask=mask, other=0.0)

    total = tl.sum(total_vals)
    promo = tl.sum(promo_vals)
    tl.store(out_ptr, (100.0 * promo) / total)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to an on-disk module and import it for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q14_triton_")
    path = os.path.join(tmpdir, "_q14_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q14_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _next_power_of_2(x: int) -> int:
    """Return a positive power-of-two block size for the Triton final reduction."""
    return 1 << max(0, x - 1).bit_length()


def _launch_q14(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 8192,
) -> torch.Tensor:
    """
    Run Q14's numeric hot path on GPU with Triton.

    cuDF performs the join and tensor extraction in run_query. This launcher uses one
    fused Triton scan for the date predicate, revenue projection, promo CASE expression,
    and per-block FP64 reductions, followed by a tiny Triton final reduction. The scan
    conditionally loads price/discount/promo only for rows in the month, avoiding the
    reference PyTorch mask, gather, revenue temporary, and multiple full-column passes.
    A single flat partial buffer also avoids extra view construction on the host path.
    """
    n = l_extendedprice.numel()
    n_blocks = triton.cdiv(n, block_size) if n > 0 else 0
    partial_stride = max(n_blocks, 1)

    partials = torch.empty((partial_stride * 2,), dtype=torch.float64, device=l_extendedprice.device)
    out = torch.empty((), dtype=torch.float64, device=l_extendedprice.device)

    if n_blocks > 0:
        _K._q14_scan_kernel[(n_blocks,)](
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            p_type_is_promo,
            partials,
            partial_stride,
            shipdate_lo_days,
            shipdate_hi_days,
            n,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

    final_block = _next_power_of_2(max(n_blocks, 1))
    _K._q14_finalize_kernel[(1,)](
        partials,
        out,
        partial_stride,
        n_blocks,
        BLOCK_SIZE=final_block,
        num_warps=4,
    )
    return out


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    GPU tensor core for TPC-H Q14.

    Inputs are aligned 1D GPU tensors prepared by run_query after the cuDF join. This
    function runs only Triton kernels for the memory-bound filter/project/aggregate
    work, fusing the eager PyTorch operations and reducing temporary HBM traffic.
    """
    return _launch_q14(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
