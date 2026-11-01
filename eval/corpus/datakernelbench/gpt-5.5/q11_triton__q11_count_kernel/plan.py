import os
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
def _q11_scan_kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    supplier_mask_ptr,
    group_values_ptr,
    total_ptr,
    supplier_mask_n: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_elements

    suppkey = tl.load(ps_suppkey_ptr + offs, mask=row_mask, other=-1)
    supp_in_range = row_mask & (suppkey >= 0) & (suppkey < supplier_mask_n)
    supp_keep = tl.load(supplier_mask_ptr + suppkey, mask=supp_in_range, other=0) != 0
    keep = row_mask & supp_keep

    partkey = tl.load(ps_partkey_ptr + offs, mask=keep, other=0)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=keep, other=0.0)
    availqty_i64 = tl.load(ps_availqty_ptr + offs, mask=keep, other=0)
    value = supplycost * availqty_i64.to(tl.float64)

    block_total = tl.sum(tl.where(keep, value, 0.0))
    tl.atomic_add(total_ptr, block_total, sem="relaxed")

    tl.atomic_add(group_values_ptr + partkey, value, sem="relaxed", mask=keep)


@triton.jit
def _q11_count_kernel(
    group_values_ptr,
    total_ptr,
    fraction_ptr,
    count_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups

    threshold = tl.load(total_ptr) * tl.load(fraction_ptr)
    vals = tl.load(group_values_ptr + offs, mask=mask, other=0.0)
    keep = mask & (vals > threshold)

    cnt = tl.sum(tl.where(keep, 1, 0))
    tl.atomic_add(count_ptr, cnt, sem="relaxed")


@triton.jit
def _q11_compact_kernel(
    group_values_ptr,
    total_ptr,
    fraction_ptr,
    write_count_ptr,
    out_keys_ptr,
    out_values_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups

    threshold = tl.load(total_ptr) * tl.load(fraction_ptr)
    vals = tl.load(group_values_ptr + offs, mask=mask, other=0.0)
    keep = mask & (vals > threshold)

    one = tl.full((BLOCK_SIZE,), 1, tl.int32)
    pos = tl.atomic_add(write_count_ptr, one, sem="relaxed", mask=keep)

    tl.store(out_keys_ptr + pos, offs.to(tl.int64), mask=keep)
    tl.store(out_values_ptr + pos, vals, mask=keep)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q11_triton_")
    path = os.path.join(tmpdir, "_q11_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)

    spec = importlib.util.spec_from_file_location("_q11_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result() -> cudf.DataFrame:
    """Return an empty cuDF result with the exact SQL output schema and dtypes."""
    return cudf.DataFrame(
        {
            "ps_partkey": cudf.Series([], dtype="int64"),
            "value": cudf.Series([], dtype="float64"),
        }
    )


def _launch_scan(
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    supplier_mask: torch.Tensor,
    max_partkey: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU/Triton path: scan `partsupp` once, test supplier membership, compute
    `ps_supplycost * ps_availqty`, and atomically accumulate both dense
    per-part sums and the global total.

    This is faster than materializing the joined rows plus separate PyTorch
    projection/grouping because non-matching rows only load `ps_suppkey` and a
    tiny supplier bitmap; expensive columns are loaded conditionally for the
    selected nation's suppliers.
    """
    device = ps_partkey.device
    group_values = torch.zeros(max_partkey + 1, dtype=torch.float64, device=device)
    total = torch.zeros(1, dtype=torch.float64, device=device)

    n = ps_suppkey.numel()
    if n == 0:
        return group_values, total

    grid = (triton.cdiv(n, block_size),)
    _K._q11_scan_kernel[grid](
        ps_partkey,
        ps_suppkey,
        ps_supplycost,
        ps_availqty,
        supplier_mask,
        group_values,
        total,
        supplier_mask.numel(),
        n,
        BLOCK_SIZE=block_size,
    )
    return group_values, total


def _launch_extract(
    group_values: torch.Tensor,
    total: torch.Tensor,
    threshold_fraction: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU/Triton path: scan the dense aggregate array, apply the HAVING threshold,
    compact surviving groups, then use PyTorch only for the final small ORDER BY.

    The threshold is carried in a float64 device tensor so Triton compares with
    the same aggregate precision as the reference.
    """
    device = group_values.device
    n_groups = group_values.numel()
    fraction = torch.tensor([threshold_fraction], dtype=torch.float64, device=device)

    count = torch.zeros(1, dtype=torch.int32, device=device)
    grid = (triton.cdiv(n_groups, block_size),)
    _K._q11_count_kernel[grid](
        group_values,
        total,
        fraction,
        count,
        n_groups,
        BLOCK_SIZE=block_size,
    )

    out_n = int(count.item())
    if out_n == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float64, device=device),
        )

    out_keys = torch.empty((out_n,), dtype=torch.int64, device=device)
    out_values = torch.empty((out_n,), dtype=torch.float64, device=device)
    write_count = torch.zeros(1, dtype=torch.int32, device=device)

    _K._q11_compact_kernel[grid](
        group_values,
        total,
        fraction,
        write_count,
        out_keys,
        out_values,
        n_groups,
        BLOCK_SIZE=block_size,
    )

    if out_n > 1:
        order = torch.argsort(out_values, descending=True)
        out_keys = out_keys[order]
        out_values = out_values[order]

    return out_keys, out_values


def _query_core(
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    supplier_mask: torch.Tensor,
    max_partkey: int,
    threshold_fraction: float,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute the numeric hot path on GPU with Triton kernels.

    The first fused Triton kernel performs filter/project/aggregate over
    `partsupp`; the second and third Triton kernels apply HAVING and compact the
    dense group array. This avoids cuDF join materialization for the large fact
    table and avoids PyTorch `unique`/`scatter_add_` over joined rows.
    """
    group_values, total = _launch_scan(
        ps_partkey=ps_partkey,
        ps_suppkey=ps_suppkey,
        ps_supplycost=ps_supplycost,
        ps_availqty=ps_availqty,
        supplier_mask=supplier_mask,
        max_partkey=max_partkey,
        block_size=block_size,
    )
    return _launch_extract(
        group_values=group_values,
        total=total,
        threshold_fraction=threshold_fraction,
        block_size=block_size,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            ps_partkey,
            sum(ps_supplycost * ps_availqty) AS value
        FROM
            partsupp,
            supplier,
            nation
        WHERE
            ps_suppkey = s_suppkey
            AND s_nationkey = n_nationkey
            AND n_name = 'GERMANY'
        GROUP BY
            ps_partkey
        HAVING
            sum(ps_supplycost * ps_availqty) > (
                SELECT
                    sum(ps_supplycost * ps_availqty) * 0.0001000000
                FROM
                    partsupp,
                    supplier,
                    nation
                WHERE
                    ps_suppkey = s_suppkey
                    AND s_nationkey = n_nationkey
                    AND n_name = 'GERMANY')
        ORDER BY
            value DESC;

    Description:
        TPC-H Q11. cuDF handles the tiny string/dimension filtering needed to
        identify supplier keys for `nation_name`. The bulk fact-table work runs
        in Triton: a fused scan filters `partsupp` by a GPU supplier bitmap,
        conditionally loads value columns, computes `ps_supplycost * ps_availqty`
        in float64, and accumulates dense per-part and global totals. Triton then
        applies HAVING and compacts results. This should be faster than the
        reference because it avoids large cuDF join materialization and avoids
        PyTorch `unique`/inverse grouping on the joined rows, while reducing HBM
        traffic for non-selected suppliers.

    Params:
        tables (dict[str, cudf.DataFrame]): input TPCH tables; uses
            `partsupp`, `supplier`, and `nation`.
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.

    Returns:
        cudf.DataFrame: columns `ps_partkey` int64 and `value` float64, ordered
        by `value` descending.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    if len(df_partsupp) == 0:
        return _empty_result()

    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    if len(filtered_nation) == 0:
        return _empty_result()

    nation_keys = filtered_nation["n_nationkey"]
    filtered_supplier = df_supplier[df_supplier["s_nationkey"].isin(nation_keys)]
    if len(filtered_supplier) == 0:
        return _empty_result()

    selected_suppkeys = torch.as_tensor(
        filtered_supplier["s_suppkey"].to_cupy(),
        dtype=torch.int64,
    )

    max_suppkey = int(df_supplier["s_suppkey"].max())
    supplier_mask = torch.zeros(
        max_suppkey + 1,
        dtype=torch.int8,
        device=selected_suppkeys.device,
    )
    supplier_mask[selected_suppkeys] = 1

    max_partkey = int(df_partsupp["ps_partkey"].max())

    ps_partkey = torch.as_tensor(df_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey = torch.as_tensor(df_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost = torch.as_tensor(
        df_partsupp["ps_supplycost"].to_cupy(),
        dtype=torch.float64,
    )
    ps_availqty = torch.as_tensor(
        df_partsupp["ps_availqty"].to_cupy(),
        dtype=torch.int64,
    )

    keys, values = _query_core(
        ps_partkey=ps_partkey,
        ps_suppkey=ps_suppkey,
        ps_supplycost=ps_supplycost,
        ps_availqty=ps_availqty,
        supplier_mask=supplier_mask,
        max_partkey=max_partkey,
        threshold_fraction=threshold_fraction,
    )

    return cudf.DataFrame(
        {
            "ps_partkey": keys,
            "value": values,
        }
    )
