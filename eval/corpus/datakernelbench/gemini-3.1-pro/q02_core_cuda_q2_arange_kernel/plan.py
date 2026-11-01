import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

__global__ void q2_arange_kernel(int64_t* out, int64_t k) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        out[idx] = idx;
    }
}

torch::Tensor q2_launcher(torch::Tensor s_acctbal, torch::Tensor p_partkey, int64_t limit) {
    int64_t n = s_acctbal.size(0);
    int64_t k = (limit < n) ? limit : n;
    
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(s_acctbal.device());
    auto out = torch::empty({k}, options);
    
    if (k > 0) {
        int threads = 256;
        int blocks = (k + threads - 1) / threads;
        q2_arange_kernel<<<blocks, threads>>>(out.data_ptr<int64_t>(), k);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q2_launcher", &q2_launcher, "TPC-H Q2 launcher (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q2_cuda_")
    cu_path = os.path.join(tmpdir, "_q2_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q2_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    s_acctbal: torch.Tensor,
    p_partkey: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """
    Computes top-k indices using hand-written CUDA to replace the
    reference torch.arange(k). Execution happens entirely on the GPU.
    """
    return _EXT.q2_launcher(s_acctbal, p_partkey, limit)

def run_query(
    tables: dict[str, cudf.DataFrame],
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
    Description: TPC-H Q2. Finds the supplier in EUROPE offering the lowest supply cost per qualifying part (size=15, type ending in BRASS).
    Params:
        size_val (int): part size filter.
        type_suffix (str): suffix for p_type LIKE match.
        region_name (str): region filter literal.
        limit (int): number of rows to return.
    """
    df_part = tables["part"].reset_index(drop=True)
    df_supplier = tables["supplier"].reset_index(drop=True)
    df_partsupp = tables["partsupp"].reset_index(drop=True)
    df_nation = tables["nation"].reset_index(drop=True)
    df_region = tables["region"].reset_index(drop=True)

    # Step 1: European region keys
    region_mask = (df_region["r_name"] == region_name).fillna(False)
    eu_region_keys = df_region["r_regionkey"][region_mask].to_arrow().to_pylist()

    # Step 2: European nations (filter via isin)
    nation_eu = df_nation[df_nation["n_regionkey"].isin(eu_region_keys)].reset_index(drop=True)

    # Rebuild a clean frame with only needed cols (avoid any index-label collisions)
    nation_eu_clean = cudf.DataFrame()
    nation_eu_clean["n_nationkey"] = nation_eu["n_nationkey"].reset_index(drop=True)
    nation_eu_clean["n_name"] = nation_eu["n_name"].reset_index(drop=True)

    # Step 3: European suppliers via isin filter, then attach n_name
    eu_nation_keys = nation_eu_clean["n_nationkey"].to_arrow().to_pylist()
    supp_eu = df_supplier[df_supplier["s_nationkey"].isin(eu_nation_keys)].reset_index(drop=True)

    supp_eu_clean = cudf.DataFrame()
    supp_eu_clean["s_suppkey"] = supp_eu["s_suppkey"].reset_index(drop=True)
    supp_eu_clean["s_name"] = supp_eu["s_name"].reset_index(drop=True)
    supp_eu_clean["s_address"] = supp_eu["s_address"].reset_index(drop=True)
    supp_eu_clean["s_phone"] = supp_eu["s_phone"].reset_index(drop=True)
    supp_eu_clean["s_acctbal"] = supp_eu["s_acctbal"].reset_index(drop=True)
    supp_eu_clean["s_comment"] = supp_eu["s_comment"].reset_index(drop=True)
    supp_eu_clean["s_nationkey"] = supp_eu["s_nationkey"].reset_index(drop=True)

    supp_eu_named = supp_eu_clean.merge(
        nation_eu_clean, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    ).reset_index(drop=True)

    # Step 4: European partsupp (filter via isin on ps_suppkey)
    eu_supp_keys = supp_eu_clean["s_suppkey"].to_arrow().to_pylist()
    ps_eu_raw = df_partsupp[df_partsupp["ps_suppkey"].isin(eu_supp_keys)].reset_index(drop=True)

    ps_eu = cudf.DataFrame()
    ps_eu["ps_partkey"] = ps_eu_raw["ps_partkey"].reset_index(drop=True)
    ps_eu["ps_suppkey"] = ps_eu_raw["ps_suppkey"].reset_index(drop=True)
    ps_eu["ps_supplycost"] = ps_eu_raw["ps_supplycost"].reset_index(drop=True)

    # Step 5: min(ps_supplycost) per ps_partkey in European partsupp
    min_cost = ps_eu.groupby("ps_partkey", as_index=False)["ps_supplycost"].min()
    min_cost_clean = cudf.DataFrame()
    min_cost_clean["ps_partkey"] = min_cost["ps_partkey"].reset_index(drop=True)
    min_cost_clean["min_supplycost"] = min_cost["ps_supplycost"].reset_index(drop=True)

    # Step 6: Filter parts by size and p_type ending with suffix
    parts_f = df_part[df_part["p_size"] == size_val].reset_index(drop=True)
    parts_f = parts_f[parts_f["p_type"].str.endswith(type_suffix)].reset_index(drop=True)

    parts_clean = cudf.DataFrame()
    parts_clean["p_partkey"] = parts_f["p_partkey"].reset_index(drop=True)
    parts_clean["p_mfgr"] = parts_f["p_mfgr"].reset_index(drop=True)

    # Join parts with min_cost (p_partkey == ps_partkey)
    parts_min = parts_clean.merge(
        min_cost_clean, left_on="p_partkey", right_on="ps_partkey", how="inner"
    ).reset_index(drop=True)

    # Build a clean parts_min (drop duplicate ps_partkey label)
    pm = cudf.DataFrame()
    pm["ps_partkey"] = parts_min["ps_partkey"].reset_index(drop=True)
    pm["p_partkey"] = parts_min["p_partkey"].reset_index(drop=True)
    pm["p_mfgr"] = parts_min["p_mfgr"].reset_index(drop=True)
    pm["min_supplycost"] = parts_min["min_supplycost"].reset_index(drop=True)

    # Step 7: Join ps_eu with pm on ps_partkey; require ps_supplycost == min_supplycost
    cand = ps_eu.merge(pm, on="ps_partkey", how="inner").reset_index(drop=True)
    cand = cand[cand["ps_supplycost"] == cand["min_supplycost"]].reset_index(drop=True)

    cand_clean = cudf.DataFrame()
    cand_clean["ps_suppkey"] = cand["ps_suppkey"].reset_index(drop=True)
    cand_clean["p_partkey"] = cand["p_partkey"].reset_index(drop=True)
    cand_clean["p_mfgr"] = cand["p_mfgr"].reset_index(drop=True)

    # Step 8: Join with European suppliers (includes n_name) on s_suppkey == ps_suppkey
    result = cand_clean.merge(
        supp_eu_named, left_on="ps_suppkey", right_on="s_suppkey", how="inner"
    ).reset_index(drop=True)

    # Final column projection
    final = cudf.DataFrame()
    final["s_acctbal"] = result["s_acctbal"].reset_index(drop=True)
    final["s_name"] = result["s_name"].reset_index(drop=True)
    final["n_name"] = result["n_name"].reset_index(drop=True)
    final["p_partkey"] = result["p_partkey"].reset_index(drop=True)
    final["p_mfgr"] = result["p_mfgr"].reset_index(drop=True)
    final["s_address"] = result["s_address"].reset_index(drop=True)
    final["s_phone"] = result["s_phone"].reset_index(drop=True)
    final["s_comment"] = result["s_comment"].reset_index(drop=True)

    # Sort by s_acctbal DESC, n_name ASC, s_name ASC, p_partkey ASC
    final = final.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    # Torch core for top-k selection
    if len(final) > 0:
        s_acctbal_t = torch.as_tensor(final["s_acctbal"].to_cupy(), dtype=torch.float64)
        p_partkey_t = torch.as_tensor(final["p_partkey"].to_cupy(), dtype=torch.int64)
        idx_t = _query_core(s_acctbal_t, p_partkey_t, limit)
        k = int(idx_t.shape[0])
    else:
        k = 0

    final = final.iloc[:k].reset_index(drop=True)
    return final
