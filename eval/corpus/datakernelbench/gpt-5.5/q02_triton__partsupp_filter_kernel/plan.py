import os
import tempfile
import importlib.util
import linecache
from typing import Any

import cupy as cp
import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _mark_keys_kernel(
    keys_ptr,
    flags_ptr,
    n_keys,
    n_flags,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_keys
    key = tl.load(keys_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key < n_flags)
    safe_key = tl.where(valid, key, 0)
    tl.store(flags_ptr + safe_key, 1, mask=valid)


@triton.jit
def _partsupp_filter_kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    part_flags_ptr,
    supp_flags_ptr,
    out_mask_ptr,
    n_rows,
    n_part_flags,
    n_supp_flags,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    partkey = tl.load(ps_partkey_ptr + offs, mask=mask, other=-1)
    suppkey = tl.load(ps_suppkey_ptr + offs, mask=mask, other=-1)

    valid_part = mask & (partkey >= 0) & (partkey < n_part_flags)
    valid_supp = mask & (suppkey >= 0) & (suppkey < n_supp_flags)

    safe_part = tl.where(valid_part, partkey, 0)
    safe_supp = tl.where(valid_supp, suppkey, 0)

    part_ok = tl.load(part_flags_ptr + safe_part, mask=valid_part, other=0)
    supp_ok = tl.load(supp_flags_ptr + safe_supp, mask=valid_supp, other=0)

    keep = mask & (part_ok != 0) & (supp_ok != 0)
    tl.store(out_mask_ptr + offs, keep, mask=mask)
'''


def _load_kernels() -> Any:
    """Write Triton JIT kernels to a real Python file and import them for validator-safe compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q2_triton_")
    path = os.path.join(tmpdir, "_q2_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q2_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_mark_keys(keys: torch.Tensor, n_flags: int, block_size: int) -> torch.Tensor:
    """
    GPU/Triton: builds a dense int8 membership bitmap from integer keys.

    This replaces expensive Python-list isin paths for the large partsupp scan; duplicate writes are benign
    because every lane stores the same value.
    """
    flags = torch.zeros((n_flags,), dtype=torch.int8, device=keys.device)
    n_keys = int(keys.numel())
    if n_keys > 0 and n_flags > 0:
        grid = (triton.cdiv(n_keys, block_size),)
        _K._mark_keys_kernel[grid](keys, flags, n_keys, n_flags, BLOCK_SIZE=block_size, num_warps=4)
    return flags


def _launch_partsupp_filter(
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    part_flags: torch.Tensor,
    supp_flags: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    GPU/Triton: one fused coalesced scan over partsupp keys.

    The kernel tests both dense membership predicates in registers and writes a boolean mask, avoiding
    multiple cuDF isin passes and reducing the rows that reach cuDF joins/groupby.
    """
    n_rows = int(ps_partkey.numel())
    out_mask = torch.empty((n_rows,), dtype=torch.bool, device=ps_partkey.device)
    if n_rows > 0:
        grid = (triton.cdiv(n_rows, block_size),)
        _K._partsupp_filter_kernel[grid](
            ps_partkey,
            ps_suppkey,
            part_flags,
            supp_flags,
            out_mask,
            n_rows,
            int(part_flags.numel()),
            int(supp_flags.numel()),
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return out_mask


def _empty_result(df_supplier: cudf.DataFrame, df_nation: cudf.DataFrame, df_part: cudf.DataFrame) -> cudf.DataFrame:
    """cuDF: construct an empty result with the exact SELECT column order and source dtypes."""
    out = cudf.DataFrame()
    out["s_acctbal"] = df_supplier["s_acctbal"].head(0).reset_index(drop=True)
    out["s_name"] = df_supplier["s_name"].head(0).reset_index(drop=True)
    out["n_name"] = df_nation["n_name"].head(0).reset_index(drop=True)
    out["p_partkey"] = df_part["p_partkey"].head(0).reset_index(drop=True)
    out["p_mfgr"] = df_part["p_mfgr"].head(0).reset_index(drop=True)
    out["s_address"] = df_supplier["s_address"].head(0).reset_index(drop=True)
    out["s_phone"] = df_supplier["s_phone"].head(0).reset_index(drop=True)
    out["s_comment"] = df_supplier["s_comment"].head(0).reset_index(drop=True)
    return out


def _filter_partsupp_triton(
    df_partsupp: cudf.DataFrame,
    part_keys: cudf.Series,
    supp_keys: cudf.Series,
    block_size: int,
) -> cudf.DataFrame:
    """
    GPU/Triton + cuDF: filter partsupp to rows whose part and supplier are both eligible.

    Triton builds dense membership bitmaps and performs one memory-bound fused scan over ps_partkey and
    ps_suppkey. cuDF then applies the resulting device boolean mask and projects the three needed columns.
    """
    if len(part_keys) == 0 or len(supp_keys) == 0 or len(df_partsupp) == 0:
        out = cudf.DataFrame()
        out["ps_partkey"] = df_partsupp["ps_partkey"].head(0).reset_index(drop=True)
        out["ps_suppkey"] = df_partsupp["ps_suppkey"].head(0).reset_index(drop=True)
        out["ps_supplycost"] = df_partsupp["ps_supplycost"].head(0).reset_index(drop=True)
        return out

    part_keys_t = torch.as_tensor(part_keys.to_cupy(), dtype=torch.int64)
    supp_keys_t = torch.as_tensor(supp_keys.to_cupy(), dtype=torch.int64)

    part_flag_len = int(part_keys.max()) + 1
    supp_flag_len = int(supp_keys.max()) + 1

    part_flags = _launch_mark_keys(part_keys_t, part_flag_len, block_size)
    supp_flags = _launch_mark_keys(supp_keys_t, supp_flag_len, block_size)

    ps_partkey_t = torch.as_tensor(df_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(df_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    keep_t = _launch_partsupp_filter(ps_partkey_t, ps_suppkey_t, part_flags, supp_flags, block_size)

    keep_cp = cp.from_dlpack(keep_t)
    keep_s = cudf.Series(keep_cp)

    filtered_raw = df_partsupp[keep_s].reset_index(drop=True)
    filtered = cudf.DataFrame()
    filtered["ps_partkey"] = filtered_raw["ps_partkey"].reset_index(drop=True)
    filtered["ps_suppkey"] = filtered_raw["ps_suppkey"].reset_index(drop=True)
    filtered["ps_supplycost"] = filtered_raw["ps_supplycost"].reset_index(drop=True)
    return filtered


def run_query(
    tables: dict[str, cudf.DataFrame],
    size_val: int = 15,
    type_suffix: str = "BRASS",
    region_name: str = "EUROPE",
    limit: int = 100,
    block_size: int = 256,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
        FROM part, supplier, partsupp, nation, region
        WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15
          AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
          AND ps_supplycost = (
              SELECT min(ps_supplycost)
              FROM partsupp, supplier, nation, region
              WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
                AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
                AND r_name = 'EUROPE')
        ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
        LIMIT 100;

    Description:
        TPC-H Q2. cuDF handles strings, small dimension joins, hash groupby, final joins, and ordering.
        Triton handles the bulk memory-bound partsupp pruning: it builds dense membership maps for
        qualifying parts and European suppliers, then performs one fused GPU scan over ps_partkey/ps_suppkey.
        This avoids Python to_pylist/isin round-trips and reduces the rows sent to cuDF groupby/join while
        preserving the exact SQL semantics.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables keyed by table name.
        size_val (int): Required p_size value.
        type_suffix (str): Suffix for p_type LIKE '%suffix'.
        region_name (str): Region name filter.
        limit (int): Number of sorted rows to return.
        block_size (int): Triton tile size for bitmap marking and partsupp scan.
    """
    df_part = tables["part"].reset_index(drop=True)
    df_supplier = tables["supplier"].reset_index(drop=True)
    df_partsupp = tables["partsupp"].reset_index(drop=True)
    df_nation = tables["nation"].reset_index(drop=True)
    df_region = tables["region"].reset_index(drop=True)

    region_mask = (df_region["r_name"] == region_name).fillna(False)
    region_keys = df_region[region_mask][["r_regionkey"]].reset_index(drop=True)
    if len(region_keys) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    nation_eu_raw = df_nation.merge(
        region_keys,
        left_on="n_regionkey",
        right_on="r_regionkey",
        how="inner",
    ).reset_index(drop=True)
    nation_eu = cudf.DataFrame()
    nation_eu["n_nationkey"] = nation_eu_raw["n_nationkey"].reset_index(drop=True)
    nation_eu["n_name"] = nation_eu_raw["n_name"].reset_index(drop=True)
    if len(nation_eu) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    supp_eu_raw = df_supplier.merge(
        nation_eu,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner",
    ).reset_index(drop=True)
    supp_eu = cudf.DataFrame()
    supp_eu["s_suppkey"] = supp_eu_raw["s_suppkey"].reset_index(drop=True)
    supp_eu["s_name"] = supp_eu_raw["s_name"].reset_index(drop=True)
    supp_eu["s_address"] = supp_eu_raw["s_address"].reset_index(drop=True)
    supp_eu["s_phone"] = supp_eu_raw["s_phone"].reset_index(drop=True)
    supp_eu["s_acctbal"] = supp_eu_raw["s_acctbal"].reset_index(drop=True)
    supp_eu["s_comment"] = supp_eu_raw["s_comment"].reset_index(drop=True)
    supp_eu["s_nationkey"] = supp_eu_raw["s_nationkey"].reset_index(drop=True)
    supp_eu["n_nationkey"] = supp_eu_raw["n_nationkey"].reset_index(drop=True)
    supp_eu["n_name"] = supp_eu_raw["n_name"].reset_index(drop=True)
    if len(supp_eu) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    part_mask = ((df_part["p_size"] == size_val).fillna(False) & df_part["p_type"].str.endswith(type_suffix).fillna(False))
    parts_raw = df_part[part_mask].reset_index(drop=True)
    parts_clean = cudf.DataFrame()
    parts_clean["p_partkey"] = parts_raw["p_partkey"].reset_index(drop=True)
    parts_clean["p_mfgr"] = parts_raw["p_mfgr"].reset_index(drop=True)
    if len(parts_clean) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    ps_filtered = _filter_partsupp_triton(
        df_partsupp,
        parts_clean["p_partkey"],
        supp_eu["s_suppkey"],
        block_size,
    )
    if len(ps_filtered) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    min_cost = ps_filtered.groupby("ps_partkey", as_index=False)["ps_supplycost"].min().reset_index(drop=True)
    min_cost_clean = cudf.DataFrame()
    min_cost_clean["ps_partkey"] = min_cost["ps_partkey"].reset_index(drop=True)
    min_cost_clean["min_supplycost"] = min_cost["ps_supplycost"].reset_index(drop=True)

    cand = ps_filtered.merge(min_cost_clean, on="ps_partkey", how="inner").reset_index(drop=True)
    cand = cand[cand["ps_supplycost"] == cand["min_supplycost"]].reset_index(drop=True)
    if len(cand) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    cand_parts = cand.merge(
        parts_clean,
        left_on="ps_partkey",
        right_on="p_partkey",
        how="inner",
    ).reset_index(drop=True)

    result = cand_parts.merge(
        supp_eu,
        left_on="ps_suppkey",
        right_on="s_suppkey",
        how="inner",
    ).reset_index(drop=True)

    final = cudf.DataFrame()
    final["s_acctbal"] = result["s_acctbal"].reset_index(drop=True)
    final["s_name"] = result["s_name"].reset_index(drop=True)
    final["n_name"] = result["n_name"].reset_index(drop=True)
    final["p_partkey"] = result["p_partkey"].reset_index(drop=True)
    final["p_mfgr"] = result["p_mfgr"].reset_index(drop=True)
    final["s_address"] = result["s_address"].reset_index(drop=True)
    final["s_phone"] = result["s_phone"].reset_index(drop=True)
    final["s_comment"] = result["s_comment"].reset_index(drop=True)

    final = final.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    k = min(int(limit), len(final))
    final = final.iloc[:k].reset_index(drop=True)
    return final
