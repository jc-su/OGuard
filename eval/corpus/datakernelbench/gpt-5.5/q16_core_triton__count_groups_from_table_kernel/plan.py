import os
import tempfile
import importlib.util
import linecache
import types
import torch
import triton
import cudf
import pandas as pd
from typing import Tuple


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _fill_i64_kernel(
    out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _insert_distinct_pairs_kernel(
    group_ptr,
    supp_ptr,
    table_ptr,
    n_elements,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    first_gid = tl.load(group_ptr + 0)
    first_supp = tl.load(supp_ptr + 0)

    gid_loaded = tl.load(group_ptr + offs, mask=valid, other=0)
    supp_loaded = tl.load(supp_ptr + offs, mask=valid, other=0)

    gid = tl.where(valid, gid_loaded, first_gid)
    supp = tl.where(valid, supp_loaded, first_supp)

    # TPC-H supplier keys and generated group ids are non-negative and fit in 32 bits.
    # Store one 64-bit composite key in an open-addressed hash set.
    key = gid * 4294967296 + supp

    h = (gid * 1315423911) ^ (supp * 2654435761)
    h = h ^ (h >> 16)
    slot = h & table_mask

    empty = tl.full((BLOCK_SIZE,), -1, tl.int64)
    done = tl.full((BLOCK_SIZE,), False, tl.int1)

    while tl.sum(tl.where(done, 0, 1)) > 0:
        old = tl.atomic_cas(table_ptr + slot, empty, key)
        hit = (old == empty) | (old == key)
        done = done | hit
        slot = tl.where(done, slot, (slot + 1) & table_mask)


@triton.jit
def _count_groups_from_table_kernel(
    table_ptr,
    counts_ptr,
    table_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < table_size
    key = tl.load(table_ptr + offs, mask=mask, other=-1)
    keep = mask & (key != -1)

    gid = key >> 32
    one = tl.full((BLOCK_SIZE,), 1, tl.int64)
    tl.atomic_add(counts_ptr + gid, one, sem="relaxed", mask=keep)
'''


def _load_kernels() -> types.ModuleType:
    """
    Write Triton GPU kernels to a real on-disk Python file and import them.

    Triton JIT compilation runs from the imported temporary module, which avoids
    validator/source-location failures when this host module is loaded via exec.
    """
    tmpdir = tempfile.mkdtemp(prefix="q16_triton_")
    path = os.path.join(tmpdir, "_q16_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q16_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _next_power_of_2(x: int) -> int:
    """Return the next power-of-two integer >= x on the CPU host."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _fill_i64(out: torch.Tensor, value: int) -> None:
    """
    Fill an int64 CUDA tensor using a Triton GPU kernel.

    This keeps initialization for the hash table/counts in the Triton path and
    avoids extra eager tensor operations in the numeric hot path.
    """
    n = out.numel()
    if n == 0:
        return
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._fill_i64_kernel[grid](out, int(value), n, BLOCK_SIZE=1024)


def _launch_distinct_count(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
    insert_block_size: int = 256,
) -> torch.Tensor:
    """
    Count distinct (group_id, supplier_id) pairs with Triton GPU kernels.

    The reference uses torch.stack + torch.unique, which materializes a 2-column
    temporary and performs a general-purpose unique. This implementation builds
    a compact open-addressed 64-bit hash set directly on GPU, then scans the set
    once to atomically count entries per group. The workload is mostly memory and
    hash-probe bound, so avoiding the stacked temporary and general sort/unique
    machinery should reduce hot-path time.
    """
    n = group_ids.numel()
    counts = torch.empty((num_groups,), dtype=torch.int64, device=group_ids.device)
    _fill_i64(counts, 0)

    if n == 0 or num_groups == 0:
        return counts

    if not group_ids.is_contiguous():
        group_ids = group_ids.contiguous()
    if not supp_ids.is_contiguous():
        supp_ids = supp_ids.contiguous()

    table_size = _next_power_of_2(max(2, n * 2))
    table = torch.empty((table_size,), dtype=torch.int64, device=group_ids.device)
    _fill_i64(table, -1)

    insert_grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._insert_distinct_pairs_kernel[insert_grid](
        group_ids,
        supp_ids,
        table,
        n,
        table_size - 1,
        BLOCK_SIZE=insert_block_size,
    )

    count_grid = lambda meta: (triton.cdiv(table_size, meta["BLOCK_SIZE"]),)
    _K._count_groups_from_table_kernel[count_grid](
        table,
        counts,
        table_size,
        BLOCK_SIZE=1024,
    )
    return counts


def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using Triton GPU kernels.

    group_ids: int64 tensor in [0, num_groups)
    supp_ids: int64 tensor of ps_suppkey values (already filtered).

    Runs entirely on the GPU numeric path: a Triton open-addressed hash set
    deduplicates composite (group, supplier) keys, and a second Triton scan
    accumulates per-group counts. This avoids the reference's general
    torch.unique over stacked pairs and its large intermediate tensor.
    """
    return _launch_distinct_count(group_ids, supp_ids, num_groups)


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
