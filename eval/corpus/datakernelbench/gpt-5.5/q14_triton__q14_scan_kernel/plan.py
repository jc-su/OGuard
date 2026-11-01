import os
import tempfile
import importlib.util
import linecache
from types import ModuleType
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
    shipdate_days_ptr,
    promo_ptr,
    date_lo,
    date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_days_ptr + offs, mask=in_bounds, other=date_hi)
    keep = in_bounds & (shipdate >= date_lo) & (shipdate < date_hi)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep, other=0)

    revenue = price * (1.0 - discount)
    total_revenue = tl.sum(tl.where(keep, revenue, 0.0))
    promo_revenue = tl.sum(tl.where(keep & promo, revenue, 0.0))

    tl.atomic_add(out_ptr, total_revenue, sem="relaxed")
    tl.atomic_add(out_ptr + 1, promo_revenue, sem="relaxed")
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q14_triton_")
    path = os.path.join(tmpdir, "_q14_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q14_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q14_scan(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 4096,
) -> torch.Tensor:
    """
    GPU/Triton: performs one fused pass over joined columns, applying the date
    predicate, computing revenue, and reducing total/promo revenue into a
    two-element fp64 tensor. This avoids PyTorch mask materialization, filtered
    temporaries, and multiple eager reduction kernels.
    """
    n = int(l_extendedprice.numel())
    out = torch.zeros(2, dtype=torch.float64, device=l_extendedprice.device)
    if n == 0:
        return out

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q14_scan_kernel[grid](
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        out,
        n,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 4096,
) -> torch.Tensor:
    """
    GPU/Triton core for Q14. The kernel fuses filter, projection, conditional
    promo projection, and both fp64 reductions in one memory-coalesced scan.
    """
    sums = _launch_q14_scan(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        block_size=block_size,
    )
    return 100.0 * sums[1] / sums[0]


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
    block_size: int = 4096,
) -> float:
    """
    SQL:
        SELECT
            100.00 * sum(
                CASE WHEN p_type LIKE 'PROMO%' THEN
                    l_extendedprice * (1 - l_discount)
                ELSE
                    0
                END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM
            lineitem,
            part
        WHERE
            l_partkey = p_partkey
            AND l_shipdate >= date '1995-09-01'
            AND l_shipdate < CAST('1995-10-01' AS date);

    Description:
        Computes the percentage of revenue from promotional parts shipped in the
        requested date range. cuDF handles string prefix evaluation on the small
        part table, epoch-day conversion, an early lineitem date reduction, and
        the relational join. The final arithmetic hot path runs in one fused
        Triton GPU kernel over the joined columns, reducing total revenue and
        promotional revenue together. This is faster than the reference because
        it filters lineitem before the join and avoids PyTorch boolean masks,
        temporary filtered revenue arrays, torch.where temporaries, and multiple
        reduction launches.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables containing
            "lineitem" and "part".
        shipdate_lo (str): Inclusive lower bound on l_shipdate.
        shipdate_hi (str): Exclusive upper bound on l_shipdate.
        block_size (int): Triton block size for the fused scan reduction.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    date_mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)

    line_proj = cudf.DataFrame(
        {
            "l_partkey": df_lineitem["l_partkey"][date_mask],
            "l_extendedprice": df_lineitem["l_extendedprice"][date_mask],
            "l_discount": df_lineitem["l_discount"][date_mask],
            "l_shipdate_days": l_shipdate_days[date_mask],
        }
    )

    part_proj = cudf.DataFrame(
        {
            "p_partkey": df_part["p_partkey"],
            "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
        }
    )

    joined = line_proj.merge(
        part_proj,
        left_on="l_partkey",
        right_on="p_partkey",
        how="inner",
        sort=False,
    )

    l_extendedprice_t = torch.as_tensor(
        joined["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        joined["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(
        joined["l_shipdate_days"].to_cupy(), dtype=torch.int32
    )
    p_type_is_promo_t = torch.as_tensor(
        joined["p_type_is_promo"].astype("bool").to_cupy(), dtype=torch.bool
    )

    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
        block_size=block_size,
    )
    return float(res.item())
