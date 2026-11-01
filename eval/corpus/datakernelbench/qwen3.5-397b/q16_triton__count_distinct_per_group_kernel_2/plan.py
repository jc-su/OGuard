import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton
from typing import Dict

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _count_distinct_per_group_kernel(
    group_ptr,
    supp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count distinct (group, supp) pairs per group.
    Input must be sorted by (group_id, supp_id).
    Memory-bound optimization: single fused pass over sorted data,
    detecting unique pairs via comparison with predecessor.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    gid = tl.load(group_ptr + offs, mask=mask, other=-1)
    sid = tl.load(supp_ptr + offs, mask=mask, other=-1)
    
    # Load previous element to detect pair boundaries
    prev_offs = offs - 1
    prev_mask = (offs > 0) & mask
    prev_gid = tl.load(group_ptr + prev_offs, mask=prev_mask, other=-1)
    prev_sid = tl.load(supp_ptr + prev_offs, mask=prev_mask, other=-1)
    
    # Mark first occurrence of each (group, supp) pair
    is_first = (offs == 0) | (gid != prev_gid) | (sid != prev_sid)
    is_first = is_first & mask
    
    # Atomically accumulate count per group
    tl.atomic_add(out_ptr + gid, is_first.to(tl.int64), mask=mask)
'''

def _load_kernels():
    """Load Triton kernels from temp file (required for JIT compilation)."""
    tmpdir = tempfile.mkdtemp(prefix="q16_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q16_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_count_distinct(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Launch Triton kernel for distinct counting per group."""
    n = group_ids.numel()
    if n == 0:
        return torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)
    
    out = torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._count_distinct_per_group_kernel[grid](
        group_ids, supp_ids, out, n,
    )
    return out

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using fused Triton kernel.
    
    GPU: Single Triton kernel pass over sorted (group_id, supp_id) pairs.
    Why faster: Fused distinct detection + counting in one kernel launch,
    avoiding torch.unique overhead and multiple tensor operations.
    Input must be sorted by (group_id, supp_id).
    
    Args:
        group_ids: int64 tensor of group identifiers [0, num_groups)
        supp_ids: int64 tensor of supplier keys (ps_suppkey)
        num_groups: number of unique groups
    
    Returns:
        int64 tensor of distinct supplier counts per group
    """
    return _launch_count_distinct(group_ids, supp_ids, num_groups)

def run_query(
    tables: dict[str, cudf.DataFrame],
    excluded_brand: str = "Brand#45",
    excluded_type_prefix: str = "MEDIUM POLISHED",
    sizes: tuple[int, ...] = (49, 14, 23, 45, 19, 3, 36, 9),
    bad_supplier_comment_pattern: str = "%Customer%Complaints%",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
        FROM partsupp, part
        WHERE p_partkey = ps_partkey
          AND p_brand <> 'Brand#45'
          AND p_type NOT LIKE 'MEDIUM POLISHED%'
          AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
          AND ps_suppkey NOT IN (
              SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
        GROUP BY p_brand, p_type, p_size
        ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
    
    Description: TPC-H Q16 — count distinct suppliers per (brand, type, size) with
    filters on part attributes and excluding suppliers with complaint patterns.
    
    cuDF: String filtering (brand, type, comment), join on partkey, group factorization.
    Triton: Fused kernel for distinct counting on sorted (group_id, supp_id) pairs.
    
    Why faster: Single Triton kernel replaces torch.unique + scatter_add,
    reducing HBM traffic and kernel launch overhead for the aggregation hot path.
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (part, partsupp, supplier).
        excluded_brand (str): Brand to exclude (default: 'Brand#45').
        excluded_type_prefix (str): Type prefix to exclude (default: 'MEDIUM POLISHED').
        sizes (tuple[int, ...]): Allowed p_size values (default: 8 specific sizes).
        bad_supplier_comment_pattern (str): SQL LIKE pattern for bad supplier comments.
    
    Returns:
        cudf.DataFrame: Result with columns p_brand, p_type, p_size, supplier_cnt.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # --- Filter part table (cuDF: string ops) ---
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]].reset_index(drop=True)

    # --- Bad suppliers subquery (cuDF: string matching) ---
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.contains(
            bad_supplier_comment_pattern.replace("%", ".*"), regex=True
        )
    ]["s_suppkey"]

    # --- Filter partsupp: exclude bad suppliers ---
    ps_filtered = df_partsupp[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers)
    ][["ps_partkey", "ps_suppkey"]].reset_index(drop=True)

    # --- Early exit if empty ---
    if len(part_filtered) == 0 or len(ps_filtered) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Join part ⨝ partsupp (cuDF) ---
    joined = part_filtered.merge(
        ps_filtered, left_on="p_partkey", right_on="ps_partkey"
    )[["p_brand", "p_type", "p_size", "ps_suppkey"]]

    if len(joined) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Factorize (brand, type, size) into group IDs ---
    group_df = joined[["p_brand", "p_type", "p_size"]].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    num_groups = len(group_df)

    # --- Transfer to GPU tensors ---
    group_ids_t = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64)
    supp_ids_t = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64)

    # --- Sort by (group_id, supp_id) for Triton distinct counting ---
    # Composite key for sorting: group_id * multiplier + supp_id
    multiplier = 1000000  # Safe for suppkey range
    composite = group_ids_t * multiplier + supp_ids_t
    sort_indices = torch.argsort(composite)
    
    group_ids_sorted = group_ids_t[sort_indices]
    supp_ids_sorted = supp_ids_t[sort_indices]

    # --- Triton: count distinct per group (fused kernel) ---
    counts_t = _query_core(group_ids_sorted, supp_ids_sorted, num_groups)

    # --- Assemble result ---
    result = cudf.DataFrame({
        "p_brand": group_df["p_brand"],
        "p_type": group_df["p_type"],
        "p_size": group_df["p_size"],
        "supplier_cnt": cudf.Series(counts_t.cpu().numpy(), dtype="int64"),
    })

    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
