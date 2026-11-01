import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <cmath>

#define BLOCK_SIZE 1024

// Kernel to filter rows where supplycost equals min_supplycost (per-row comparison)
__global__ void filter_min_cost_kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ min_supplycost_ptr,
    int64_t* __restrict__ keep_indices,
    int64_t* __restrict__ keep_count,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double cost = supplycost_ptr[idx];
        double min_c = min_supplycost_ptr[idx];
        bool keep = (cost == min_c);
        if (keep) {
            unsigned long long int* count_ptr = (unsigned long long int*)keep_count;
            unsigned long long int pos = atomicAdd(count_ptr, (unsigned long long int)1);
            keep_indices[pos] = idx;
        }
    }
}

// Launcher for filter
torch::Tensor filter_launcher(
    torch::Tensor supplycost,
    torch::Tensor min_supplycost,
    int64_t n_elements
) {
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(min_supplycost.scalar_type() == torch::kFloat64, "min_supplycost must be float64");
    
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!min_supplycost.is_cuda()) min_supplycost = min_supplycost.cuda();
    
    supplycost = supplycost.contiguous();
    min_supplycost = min_supplycost.contiguous();
    
    auto keep_indices = torch::empty({n_elements}, torch::dtype(torch::kInt64).device(supplycost.device()));
    auto keep_count = torch::zeros({1}, torch::dtype(torch::kInt64).device(supplycost.device()));
    
    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    filter_min_cost_kernel<<<blocks, BLOCK_SIZE>>>(
        supplycost.data_ptr<double>(),
        min_supplycost.data_ptr<double>(),
        keep_indices.data_ptr<int64_t>(),
        keep_count.data_ptr<int64_t>(),
        n_elements
    );
    
    cudaDeviceSynchronize();
    
    auto count_cpu = keep_count.cpu();
    int64_t count = count_cpu.data_ptr<int64_t>()[0];
    
    if (count > 0) {
        keep_indices = keep_indices.slice(0, 0, count).contiguous();
    } else {
        keep_indices = torch::empty({0}, torch::dtype(torch::kInt64).device(supplycost.device()));
    }
    
    return keep_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_launcher", &filter_launcher, "Filter rows where supplycost == min_supplycost (CUDA)");
}
'''

def _load_cuda():
    # Use a fixed build directory for caching across imports
    tmpdir = tempfile.gettempdir()
    build_dir = os.path.join(tmpdir, "q2_cuda_build")
    os.makedirs(build_dir, exist_ok=True)
    cu_path = os.path.join(build_dir, "_q2_kernels.cu")
    
    # Only write if not exists or content changed
    write_file = True
    if os.path.exists(cu_path):
        with open(cu_path, "r") as f:
            if f.read() == _CUDA_SOURCE:
                write_file = False
    
    if write_file:
        with open(cu_path, "w") as f:
            f.write(_CUDA_SOURCE)
    
    return load(
        name="q2_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

_EXT = _load_cuda()

def _query_core(
    ps_supplycost: torch.Tensor,
    min_supplycost: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated filter for TPC-H Q2 hot path.
    
    Runs on GPU via CUDA kernel to filter rows where supplycost == min_supplycost.
    Uses coalesced memory access and atomic operations for efficient parallel filtering.
    
    Args:
        ps_supplycost: supplycost column from candidate rows (float64)
        min_supplycost: min_supplycost column from candidate rows (float64)
    
    Returns:
        keep_indices: Indices of rows where supplycost == min_supplycost (int64)
    """
    keep_indices = _EXT.filter_launcher(
        ps_supplycost, min_supplycost, ps_supplycost.shape[0]
    )
    return keep_indices


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
    
    Description: TPC-H Q2. Finds the supplier in EUROPE offering the lowest supply cost 
    per qualifying part (size=15, type ending in BRASS).
    
    Optimization strategy:
    - cuDF: Complex joins (region→nation→supplier), string operations (p_type LIKE), 
      groupby aggregation (min per partkey), final sort - all known-correct operations
    - CUDA: Hot path filter operation - identifies rows where supplycost == min_supplycost 
      using fused kernel with coalesced memory access and atomic index gathering
    
    Params:
        size_val (int): part size filter (default 15).
        type_suffix (str): suffix for p_type LIKE match (default "BRASS").
        region_name (str): region filter literal (default "EUROPE").
        limit (int): number of rows to return (default 100).
    
    Returns:
        cudf.DataFrame: Query results with columns [s_acctbal, s_name, n_name, p_partkey, 
                      p_mfgr, s_address, s_phone, s_comment]
    """
    df_part = tables["part"].reset_index(drop=True)
    df_supplier = tables["supplier"].reset_index(drop=True)
    df_partsupp = tables["partsupp"].reset_index(drop=True)
    df_nation = tables["nation"].reset_index(drop=True)
    df_region = tables["region"].reset_index(drop=True)

    # Step 1: European region keys
    region_mask = (df_region["r_name"] == region_name).fillna(False)
    eu_region_keys = df_region["r_regionkey"][region_mask].to_arrow().to_pylist()

    # Step 2: European nations
    nation_eu = df_nation[df_nation["n_regionkey"].isin(eu_region_keys)].reset_index(drop=True)
    nation_eu_clean = cudf.DataFrame()
    nation_eu_clean["n_nationkey"] = nation_eu["n_nationkey"].reset_index(drop=True)
    nation_eu_clean["n_name"] = nation_eu["n_name"].reset_index(drop=True)

    # Step 3: European suppliers
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

    # Step 4: European partsupp
    eu_supp_keys = supp_eu_clean["s_suppkey"].to_arrow().to_pylist()
    ps_eu_raw = df_partsupp[df_partsupp["ps_suppkey"].isin(eu_supp_keys)].reset_index(drop=True)
    ps_eu = cudf.DataFrame()
    ps_eu["ps_partkey"] = ps_eu_raw["ps_partkey"].reset_index(drop=True)
    ps_eu["ps_suppkey"] = ps_eu_raw["ps_suppkey"].reset_index(drop=True)
    ps_eu["ps_supplycost"] = ps_eu_raw["ps_supplycost"].reset_index(drop=True)

    # Step 5: min(ps_supplycost) per ps_partkey
    if len(ps_eu) > 0:
        min_cost_gdf = ps_eu.groupby("ps_partkey", as_index=False)["ps_supplycost"].min()
        min_cost_gdf = min_cost_gdf.rename(columns={"ps_supplycost": "min_supplycost"})
        
        # Step 6: Filter parts
        parts_f = df_part[df_part["p_size"] == size_val].reset_index(drop=True)
        parts_f = parts_f[parts_f["p_type"].str.endswith(type_suffix)].reset_index(drop=True)
        parts_clean = cudf.DataFrame()
        parts_clean["p_partkey"] = parts_f["p_partkey"].reset_index(drop=True)
        parts_clean["p_mfgr"] = parts_f["p_mfgr"].reset_index(drop=True)

        if len(parts_clean) > 0 and len(min_cost_gdf) > 0:
            parts_min = parts_clean.merge(
                min_cost_gdf, left_on="p_partkey", right_on="ps_partkey", how="inner"
            ).reset_index(drop=True)
            
            pm = cudf.DataFrame()
            pm["ps_partkey"] = parts_min["ps_partkey"].reset_index(drop=True)
            pm["p_partkey"] = parts_min["p_partkey"].reset_index(drop=True)
            pm["p_mfgr"] = parts_min["p_mfgr"].reset_index(drop=True)
            pm["min_supplycost"] = parts_min["min_supplycost"].reset_index(drop=True)

            # Step 7: Join and filter
            cand = ps_eu.merge(pm, on="ps_partkey", how="inner").reset_index(drop=True)
            
            if len(cand) > 0:
                ps_supplycost_t = torch.as_tensor(cand["ps_supplycost"].to_cupy(), dtype=torch.float64)
                min_supplycost_t = torch.as_tensor(cand["min_supplycost"].to_cupy(), dtype=torch.float64)
                
                keep_indices_t = _query_core(ps_supplycost_t, min_supplycost_t)
                
                if len(keep_indices_t) > 0:
                    keep_idx_np = keep_indices_t.cpu().numpy()
                    cand_filtered = cudf.DataFrame()
                    cand_filtered["ps_suppkey"] = cand["ps_suppkey"].iloc[keep_idx_np].reset_index(drop=True)
                    cand_filtered["p_partkey"] = cand["p_partkey"].iloc[keep_idx_np].reset_index(drop=True)
                    cand_filtered["p_mfgr"] = cand["p_mfgr"].iloc[keep_idx_np].reset_index(drop=True)
                else:
                    cand_filtered = cudf.DataFrame()
                    cand_filtered["ps_suppkey"] = cudf.Series([], dtype='int64')
                    cand_filtered["p_partkey"] = cudf.Series([], dtype='int64')
                    cand_filtered["p_mfgr"] = cudf.Series([], dtype='object')
            else:
                cand_filtered = cudf.DataFrame()
                cand_filtered["ps_suppkey"] = cudf.Series([], dtype='int64')
                cand_filtered["p_partkey"] = cudf.Series([], dtype='int64')
                cand_filtered["p_mfgr"] = cudf.Series([], dtype='object')
        else:
            cand_filtered = cudf.DataFrame()
            cand_filtered["ps_suppkey"] = cudf.Series([], dtype='int64')
            cand_filtered["p_partkey"] = cudf.Series([], dtype='int64')
            cand_filtered["p_mfgr"] = cudf.Series([], dtype='object')
    else:
        cand_filtered = cudf.DataFrame()
        cand_filtered["ps_suppkey"] = cudf.Series([], dtype='int64')
        cand_filtered["p_partkey"] = cudf.Series([], dtype='int64')
        cand_filtered["p_mfgr"] = cudf.Series([], dtype='object')

    # Step 8: Join with suppliers
    if len(cand_filtered) > 0 and len(supp_eu_named) > 0:
        result = cand_filtered.merge(
            supp_eu_named, left_on="ps_suppkey", right_on="s_suppkey", how="inner"
        ).reset_index(drop=True)
    else:
        result = cudf.DataFrame()

    # Final projection
    if len(result) > 0:
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
        final = final.iloc[:limit].reset_index(drop=True)
    else:
        final = cudf.DataFrame()
        final["s_acctbal"] = cudf.Series([], dtype='float64')
        final["s_name"] = cudf.Series([], dtype='object')
        final["n_name"] = cudf.Series([], dtype='object')
        final["p_partkey"] = cudf.Series([], dtype='int64')
        final["p_mfgr"] = cudf.Series([], dtype='object')
        final["s_address"] = cudf.Series([], dtype='object')
        final["s_phone"] = cudf.Series([], dtype='object')
        final["s_comment"] = cudf.Series([], dtype='object')

    return final
