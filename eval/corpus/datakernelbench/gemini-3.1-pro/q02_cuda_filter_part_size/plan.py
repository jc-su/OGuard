import os
import tempfile
import torch
import cudf
import cupy as cp
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

__global__ void filter_part_size(
    const int64_t* __restrict__ p_size,
    int64_t size_val,
    bool* __restrict__ out_mask,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_mask[idx] = (p_size[idx] == size_val);
    }
}

torch::Tensor filter_part_size_launcher(torch::Tensor p_size, int64_t size_val) {
    TORCH_CHECK(p_size.scalar_type() == torch::kInt64);
    if (!p_size.is_cuda()) p_size = p_size.cuda();
    p_size = p_size.contiguous();
    
    int64_t n = p_size.numel();
    auto options = torch::TensorOptions().dtype(torch::kBool).device(p_size.device());
    auto out = torch::empty({n}, options);
    
    if (n > 0) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        filter_part_size<<<blocks, threads>>>(p_size.data_ptr<int64_t>(), size_val, out.data_ptr<bool>(), n);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_part_size", &filter_part_size_launcher, "Filter part size");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q2_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
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
    Description: TPC-H Q2. Finds the supplier in EUROPE offering the lowest supply cost per qualifying part.
                 Uses a fused CUDA kernel for part size filtering and efficient cuDF joins to bypass costly list lookups.
    Params:
        size_val (int): part size filter.
        type_suffix (str): suffix for p_type LIKE match.
        region_name (str): region filter literal.
        limit (int): number of rows to return.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_partsupp = tables["partsupp"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # 1. European region keys
    region_eu = df_region[df_region["r_name"] == region_name][["r_regionkey"]]

    # 2. European nations
    nation_eu = df_nation.merge(region_eu, left_on="n_regionkey", right_on="r_regionkey", how="inner")
    nation_eu = nation_eu[["n_nationkey", "n_name"]]

    # 3. European suppliers
    supp_eu = df_supplier.merge(nation_eu, left_on="s_nationkey", right_on="n_nationkey", how="inner")
    supp_eu = supp_eu[["s_suppkey", "s_name", "s_address", "s_phone", "s_acctbal", "s_comment", "n_name"]]

    # 4. Filter parts (Using compiled CUDA kernel for integer equality filter)
    p_size_t = torch.as_tensor(df_part["p_size"].to_cupy(), dtype=torch.int64)
    size_mask_t = _EXT.filter_part_size(p_size_t, size_val)
    size_mask = cp.asarray(size_mask_t)
    
    parts_f = df_part[size_mask].reset_index(drop=True)
    parts_f = parts_f[parts_f["p_type"].str.endswith(type_suffix)][["p_partkey", "p_mfgr"]]

    # 5. Filter partsupp to matching parts
    ps_matching_parts = df_partsupp.merge(parts_f, left_on="ps_partkey", right_on="p_partkey", how="inner")

    # 6. Filter partsupp to EU suppliers
    ps_eu_parts = ps_matching_parts.merge(supp_eu, left_on="ps_suppkey", right_on="s_suppkey", how="inner")

    # 7. Find min supply cost per part
    min_cost = ps_eu_parts.groupby("ps_partkey", as_index=False)["ps_supplycost"].min()
    min_cost = min_cost.rename(columns={"ps_supplycost": "min_supplycost"})

    # 8. Filter to min supply cost
    final = ps_eu_parts.merge(min_cost, on="ps_partkey", how="inner")
    final = final[final["ps_supplycost"] == final["min_supplycost"]]

    # 9. Format, Sort and Limit
    final = final[["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment"]]
    final = final.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True]
    ).head(limit).reset_index(drop=True)

    return final
