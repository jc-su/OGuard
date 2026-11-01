import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
import cudf
from typing import Dict

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _min_cost_per_partkey(
    partkey_ptr,
    cost_ptr,
    min_cost_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    cost = tl.load(cost_ptr + offs, mask=mask, other=1e30)
    tl.atomic_min(min_cost_ptr + pk, cost)

@triton.jit
def _filter_min_cost(
    partkey_ptr,
    cost_ptr,
    min_cost_ptr,
    keep_mask_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    cost = tl.load(cost_ptr + offs, mask=mask, other=-1.0)
    min_c = tl.load(min_cost_ptr + pk)
    diff = tl.abs(cost - min_c)
    keep = mask & (diff < 1e-9)
    tl.store(keep_mask_ptr + offs, keep.to(tl.int32))
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q2_triton_")
    path = os.path.join(tmpdir, "_q2_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q2_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: Dict[str, cudf.DataFrame],
    size_val: int = 15,
    type_suffix: str = "BRASS",
    region_name: str = "EUROPE",
    limit: int = 100,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
        FROM part, supplier, partsupp, nation, region
        WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15
          AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
          AND ps_supplycost = (SELECT min(ps_supplycost) FROM partsupp, supplier, nation, region
                               WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
                                 AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
                                 AND r_name = 'EUROPE')
        ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
        LIMIT 100;
    Description: TPC-H Q2. Finds European suppliers with minimum supply cost per part.
                 Uses Triton for min-cost computation and filtering to reduce HBM passes.
                 cuDF handles dimension filtering and joins; Triton handles numeric aggregation.
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables.
        size_val (int): Part size filter.
        type_suffix (str): Part type suffix.
        region_name (str): Region name.
        limit (int): Result limit.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_partsupp = tables["partsupp"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # Filter dimensions
    eu_rkeys = df_region.loc[df_region["r_name"] == region_name, "r_regionkey"]
    eu_nkeys = df_nation.loc[df_nation["n_regionkey"].isin(eu_rkeys), "n_nationkey"]
    df_n_eu = df_nation.loc[df_nation["n_regionkey"].isin(eu_rkeys), ["n_nationkey", "n_name"]].reset_index(drop=True)
    
    df_s_eu = df_supplier.loc[
        df_supplier["s_nationkey"].isin(eu_nkeys),
        ["s_suppkey", "s_name", "s_address", "s_phone", "s_acctbal", "s_comment", "s_nationkey"]
    ].reset_index(drop=True)
    
    df_p_f = df_part.loc[
        (df_part["p_size"] == size_val) & df_part["p_type"].str.endswith(type_suffix),
        ["p_partkey", "p_mfgr"]
    ].reset_index(drop=True)

    # Join fact tables
    df_ps = df_partsupp.merge(df_s_eu, left_on="ps_suppkey", right_on="s_suppkey", how="inner")
    df_w = df_ps.merge(df_p_f, left_on="ps_partkey", right_on="p_partkey", how="inner")
    df_w = df_w.merge(df_n_eu, left_on="s_nationkey", right_on="n_nationkey", how="inner").reset_index(drop=True)
    
    if len(df_w) == 0:
        return cudf.DataFrame(columns=["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", 
                                        "s_address", "s_phone", "s_comment"])

    # Triton: compute min cost per partkey
    pk_t = torch.as_tensor(df_w["p_partkey"].to_cupy(), dtype=torch.int64)
    cost_t = torch.as_tensor(df_w["ps_supplycost"].to_cupy(), dtype=torch.float64)
    n = len(df_w)
    max_pk = int(pk_t.max().item()) + 2
    
    # Initialize with large value for atomic_min
    min_cost_buf = torch.ones((max_pk,), dtype=torch.float64, device=pk_t.device) * 1e30
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._min_cost_per_partkey[grid](pk_t, cost_t, min_cost_buf, n, BLOCK_SIZE=BLOCK_SIZE)

    # Triton: filter rows where cost == min (with tolerance)
    keep_mask = torch.zeros(n, dtype=torch.int32, device=pk_t.device)
    _K._filter_min_cost[grid](pk_t, cost_t, min_cost_buf, keep_mask, n, BLOCK_SIZE=BLOCK_SIZE)
    
    keep_idx = torch.where(keep_mask == 1)[0].cpu().numpy()
    df_cand = df_w.iloc[keep_idx].reset_index(drop=True)

    # Sort and limit
    if len(df_cand) > 0:
        df_cand = df_cand.sort_values(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            ascending=[False, True, True, True]
        ).reset_index(drop=True)
        df_cand = df_cand.head(limit)

    # Project output columns
    result = cudf.DataFrame()
    result["s_acctbal"] = df_cand["s_acctbal"].reset_index(drop=True)
    result["s_name"] = df_cand["s_name"].reset_index(drop=True)
    result["n_name"] = df_cand["n_name"].reset_index(drop=True)
    result["p_partkey"] = df_cand["p_partkey"].reset_index(drop=True)
    result["p_mfgr"] = df_cand["p_mfgr"].reset_index(drop=True)
    result["s_address"] = df_cand["s_address"].reset_index(drop=True)
    result["s_phone"] = df_cand["s_phone"].reset_index(drop=True)
    result["s_comment"] = df_cand["s_comment"].reset_index(drop=True)
    
    return result
