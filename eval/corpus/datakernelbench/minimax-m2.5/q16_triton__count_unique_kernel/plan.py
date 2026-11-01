import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source: fused kernel to count distinct supplier keys
# ----------------------------------------------------------------------
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
    reset_to_zero=["out_counts_ptr"],
)
@triton.jit
def _count_unique_kernel(sorted_keys_ptr, out_counts_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # FACTOR > max supplier id (supplier rows = 100k). 100001 is safe.
    GROUP_FACTOR = 100001
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load current composite key
    key = tl.load(sorted_keys_ptr + offs, mask=mask, other=0)

    # Load previous key (if any) – needed to detect a new unique pair
    prev_offs = offs - 1
    prev_mask = (offs > 0) & mask
    prev_key = tl.load(sorted_keys_ptr + prev_offs, mask=prev_mask, other=0)

    # Is this the very first element? If so, it's a new pair.
    is_first = offs == 0
    # Keep this element if it's the first or the key differs from the previous one.
    keep = is_first | (key != prev_key)

    # Decode group id from composite key (high bits)
    group_id = key // GROUP_FACTOR

    # Atomically increment the counter for this group if the pair is unique.
    tl.atomic_add(out_counts_ptr + group_id, tl.cast(keep, tl.int64))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q16_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


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

    Description:
        This implementation follows the reference cuDF pipeline for filtering and joining,
        then accelerates the hot‑path (distinct supplier counting) with a fused Triton
        kernel.  The steps are:
        1) Build a composite 64‑bit key = group_id * FACTOR + suppkey.
        2) Sort the keys with torch.sort – memory‑bound but highly optimized.
        3) A single‑pass Triton kernel that scans the sorted array, detects unique
           (group, supplier) pairs, and atomically increments the per‑group counter.
        This avoids the overhead of torch.unique + scatter_add used in the reference.

    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (part, partsupp, supplier).
        excluded_brand (str): Brand to exclude.
        excluded_type_prefix (str): Type prefix to exclude (LIKE 'prefix%').
        sizes (tuple[int, ...]): Allowed part sizes.
        bad_supplier_comment_pattern (str): SQL LIKE pattern for supplier comments to exclude.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # ------------------------------------------------------------------
    # 1) Filter part table
    # ------------------------------------------------------------------
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]]

    # ------------------------------------------------------------------
    # 2) Bad suppliers (sub‑query)
    # ------------------------------------------------------------------
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        )
    ]["s_suppkey"]

    # ------------------------------------------------------------------
    # 3) Filter partsupp, removing bad suppliers
    # ------------------------------------------------------------------
    ps_filtered = df_partsupp[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers)
    ][["ps_partkey", "ps_suppkey"]]

    # ------------------------------------------------------------------
    # 4) Join part ↔ partsupp
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5) Encode (brand,type,size) groups as integer ids
    # ------------------------------------------------------------------
    group_df = (
        joined[["p_brand", "p_type", "p_size"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")
    num_groups = len(group_df)

    # Attach group ids to the joined rows
    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    # ------------------------------------------------------------------
    # 6) GPU‑side distinct counting (the hot path)
    # ------------------------------------------------------------------
    # Factor larger than any supplier id (max supplier rows = 100 000)
    FACTOR = 100001

    # Transfer columns to GPU
    group_ids = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64)
    supp_ids = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64)

    # Composite key: high bits = group, low bits = supplier
    composite_keys = group_ids * FACTOR + supp_ids

    # Sort to bring duplicate (group, supplier) pairs together
    sorted_keys, _ = torch.sort(composite_keys)

    # Per‑group counter (zero‑initialized)
    counts = torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)

    # Launch fused Triton kernel
    n_rows = sorted_keys.shape[0]
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._count_unique_kernel[grid](sorted_keys, counts, n_rows)

    # ------------------------------------------------------------------
    # 7) Assemble result DataFrame
    # ------------------------------------------------------------------
    result = cudf.DataFrame({
        "p_brand": group_df["p_brand"],
        "p_type": group_df["p_type"],
        "p_size": group_df["p_size"],
        "supplier_cnt": cudf.Series(counts, dtype="int64"),
    })

    # Sort as required: supplier_cnt DESC, then brand/type/size ASC
    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
