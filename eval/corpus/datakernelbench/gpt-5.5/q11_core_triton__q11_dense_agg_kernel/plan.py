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
def _q11_dense_agg_kernel(
    ps_partkey_ptr,
    ps_supplycost_ptr,
    ps_availqty_ptr,
    group_values_ptr,
    group_counts_ptr,
    total_ptr,
    n_elements,
    base_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    partkeys = tl.load(ps_partkey_ptr + offs, mask=mask, other=base_key)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=mask, other=0.0)
    availqty = tl.load(ps_availqty_ptr + offs, mask=mask, other=0.0)

    values = supplycost * availqty
    group_offs = partkeys - base_key

    tl.atomic_add(group_values_ptr + group_offs, values, mask=mask, sem="relaxed")
    tl.atomic_add(
        group_counts_ptr + group_offs,
        tl.full((BLOCK_SIZE,), 1, tl.int32),
        mask=mask,
        sem="relaxed",
    )

    block_sum = tl.sum(tl.where(mask, values, 0.0))
    tl.atomic_add(total_ptr, block_sum, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir: str = tempfile.mkdtemp(prefix="q11_triton_")
    path: str = os.path.join(tmpdir, "_q11_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q11_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _ensure_contiguous_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a 1D tensor and make it contiguous for coalesced Triton pointer arithmetic."""
    if x.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    return x if x.is_contiguous() else x.contiguous()


def _launch_q11_dense_aggregate(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    block_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Run the GPU hot aggregation path.

    The Triton kernel performs one fused pass over joined partsupp rows: it computes
    ps_supplycost * ps_availqty in registers, atomically accumulates a dense
    per-partkey sum, records key presence, and accumulates the global total. This
    replaces the reference torch.unique + inverse + scatter_add hot path with a
    bandwidth-friendly GPU pass and avoids materializing the per-row value tensor.
    """
    n: int = ps_partkey.numel()
    if n == 0:
        empty_values = torch.empty(0, dtype=ps_supplycost.dtype, device=ps_supplycost.device)
        empty_counts = torch.empty(0, dtype=torch.int32, device=ps_partkey.device)
        total = torch.zeros(1, dtype=ps_supplycost.dtype, device=ps_supplycost.device)
        return empty_values, empty_counts, total, 0

    min_key_t, max_key_t = torch.aminmax(ps_partkey)
    min_key: int = int(min_key_t.item())
    max_key: int = int(max_key_t.item())
    dense_size: int = max_key - min_key + 1

    group_values = torch.zeros(dense_size, dtype=ps_supplycost.dtype, device=ps_supplycost.device)
    group_counts = torch.zeros(dense_size, dtype=torch.int32, device=ps_partkey.device)
    total = torch.zeros(1, dtype=ps_supplycost.dtype, device=ps_supplycost.device)

    grid = (triton.cdiv(n, block_size),)
    _K._q11_dense_agg_kernel[grid](
        ps_partkey,
        ps_supplycost,
        ps_availqty,
        group_values,
        group_counts,
        total,
        n,
        min_key,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )
    return group_values, group_counts, total, min_key


def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU tensor core for TPC-H Q11 after cuDF has performed filtering and joins.

    Runs the expensive numeric GROUP BY on the GPU with a Triton fused dense
    aggregation kernel: one pass reads partkey/supplycost/availqty, computes the
    value in registers, accumulates per-part sums, and accumulates the global
    threshold total. PyTorch tensor ops are then used only for the comparatively
    small presence compaction and ORDER BY. This is faster than the reference hot
    path because it avoids torch.unique/inverse construction and avoids a separate
    materialized per-row value tensor.
    """
    ps_partkey = _ensure_contiguous_1d(ps_partkey, "ps_partkey")
    ps_supplycost = _ensure_contiguous_1d(ps_supplycost, "ps_supplycost")
    ps_availqty = _ensure_contiguous_1d(ps_availqty, "ps_availqty")

    if ps_partkey.numel() == 0:
        return (
            torch.empty(0, dtype=ps_partkey.dtype, device=ps_partkey.device),
            torch.empty(0, dtype=ps_supplycost.dtype, device=ps_supplycost.device),
        )

    grouped_values, group_counts, total, min_key = _launch_q11_dense_aggregate(
        ps_partkey, ps_supplycost, ps_availqty
    )

    present_offsets = torch.nonzero(group_counts, as_tuple=False).flatten()
    present_values = grouped_values[present_offsets]
    threshold = total[0] * threshold_fraction

    mask = present_values > threshold
    filtered_offsets = present_offsets[mask]
    filtered_values = present_values[mask]
    filtered_keys = filtered_offsets.to(torch.int64) + min_key

    sort_idx = torch.argsort(filtered_values, descending=True)
    return filtered_keys[sort_idx], filtered_values[sort_idx]


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        )
        ORDER BY value DESC;
    Description: TPC-H Q11 - Identifies parts contributing significantly to the total
                 supply cost value for suppliers in a given nation.
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # Filter nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # Join supplier <-> nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Join partsupp <-> (supplier <-> nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
