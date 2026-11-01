import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
import cudf


# Simple Triton kernel: atomic add for each unique (group, supp) pair
# Uses reset_to_zero to avoid accumulation across autotune runs
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs in [256, 512, 1024]
        for w in [4, 8]
    ],
    key=["n_elements"],
    reset_to_zero=["output_ptr"],
)
@triton.jit
def _count_unique_kernel(
    unique_group_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one unique (group, supp) pair
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load groups for valid indices only
    groups = tl.load(unique_group_ptr + idx, mask=mask, other=-1)
    
    # For each valid element, add 1 to its group's count
    for i in range(BLOCK_SIZE):
        local_idx = pid * BLOCK_SIZE + i
        if local_idx < n_elements:
            g = tl.load(unique_group_ptr + local_idx)
            tl.atomic_add(output_ptr + g, 1)
'''


def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q16_triton_")
    path = os.path.join(tmpdir, "_q16_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q16_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _launch_unique_count(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Optimized distinct count using sort + vectorized unique detection.
    
    Steps:
    1. Sort by (group_id, supp_id) using GPU-optimized torch.sort
    2. Use vectorized torch operations to find unique transitions
    3. Use Triton atomic_add to count per group
    
    This is faster than reference because:
    - argsort + diff is faster than torch.unique on stacked Nx2 matrix
    - Parallel atomic_add in Triton for the final counting
    """
    n = group_ids.numel()
    
    # Sort by (group, supp) - composite key approach
    max_supp = supp_ids.max().item() + 1
    composite_key = group_ids * max_supp + supp_ids
    
    sorted_idx = torch.argsort(composite_key, stable=True)
    sorted_group = group_ids[sorted_idx]
    sorted_supp = supp_ids[sorted_idx]
    
    # Find unique transitions using vectorized operations
    # A pair is unique if group OR supp differs from previous
    # Pad with sentinel values that will never match
    prev_group = torch.roll(sorted_group, 1)
    prev_supp = torch.roll(sorted_supp, 1)
    prev_group[0] = -1
    prev_supp[0] = -1
    
    # is_unique = (group != prev_group) OR (supp != prev_supp)
    is_unique = (sorted_group != prev_group) | (sorted_supp != prev_supp)
    
    # Extract unique group IDs
    unique_groups = sorted_group[is_unique]
    
    # Output tensor for counts per group
    counts = torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)
    
    num_unique = unique_groups.numel()
    
    if num_unique > 0:
        # Launch Triton kernel to count unique groups
        grid = lambda meta: (triton.cdiv(num_unique, meta["BLOCK_SIZE"]),)
        
        _K._count_unique_kernel[grid](
            unique_groups, counts, num_unique,
        )
    
    return counts


def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using optimized Triton kernel.
    
    Optimized approach:
    - Sort by (group_id, supp_id) using GPU-optimized torch.sort
    - Vectorized torch operations for unique detection
    - Parallel Triton atomic_add for counting
    
    This replaces torch.unique + scatter_add with sort + vectorized diff + Triton.
    """
    return _launch_unique_count(group_ids, supp_ids, num_groups)


# run_query is copied verbatim from reference
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
                 filters on part attributes and excluding suppliers whose comment
                 matches a complaint pattern.
    Params:
        excluded_brand (str): brand to exclude.
        excluded_type_prefix (str): type prefix to exclude (LIKE 'prefix%').
        sizes (tuple[int, ...]): allowed p_size values.
        bad_supplier_comment_pattern (str): SQL LIKE pattern for supplier comments to exclude.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # --- Filter part table ---
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]]

    # --- Bad suppliers (subquery) ---
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        )
    ]["s_suppkey"]

    # --- Filter partsupp: exclude bad suppliers ---
    ps_filtered = df_partsupp[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers)
    ][["ps_partkey", "ps_suppkey"]]

    # --- Join part ⨝ partsupp ---
    joined = part_filtered.merge(
        ps_filtered, left_on="p_partkey", right_on="ps_partkey"
    )

    if len(joined) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Build group keys on host/cuDF side (strings can't go to torch) ---
    # Factorize composite (p_brand, p_type, p_size) into integer group ids.
    group_df = joined[["p_brand", "p_type", "p_size"]].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    num_groups = len(group_df)

    group_ids_t = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64)
    supp_ids_t = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64)

    counts_t = _query_core(group_ids_t, supp_ids_t, num_groups)

    # --- Assemble result ---
    result = cudf.DataFrame({
        "p_brand": group_df["p_brand"],
        "p_type": group_df["p_type"],
        "p_size": group_df["p_size"],
        "supplier_cnt": cudf.Series(counts_t, dtype="int64"),
    })

    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
