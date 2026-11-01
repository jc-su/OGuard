import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q11_kernel(
    const int64_t* __restrict__ partkey,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    double* __restrict__ grouped_values,
    double* __restrict__ total_sum,
    int64_t n_elements
) {
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double val = 0.0;
    if (idx < n_elements) {
        val = supplycost[idx] * availqty[idx];
        // Accumulate into the group-by array directly since keys are dense
        atomicAdd(&grouped_values[partkey[idx]], val);
    }
    
    // Block-level reduction for the total sum
    block_sum[threadIdx.x] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(total_sum, block_sum[0]);
    }
}

std::vector<torch::Tensor> q11_launcher(
    torch::Tensor partkey,
    torch::Tensor supplycost,
    torch::Tensor availqty,
    int64_t max_key
) {
    TORCH_CHECK(partkey.scalar_type() == torch::kInt64, "partkey must be int64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(availqty.scalar_type() == torch::kFloat64, "availqty must be float64");
    
    if (!partkey.is_cuda()) partkey = partkey.cuda();
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!availqty.is_cuda()) availqty = availqty.cuda();
    
    partkey = partkey.contiguous();
    supplycost = supplycost.contiguous();
    availqty = availqty.contiguous();
    
    int64_t n = partkey.numel();
    
    auto options = supplycost.options();
    auto grouped_values = torch::zeros({max_key + 1}, options);
    auto total_sum = torch::zeros({1}, options);
    
    if (n > 0) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q11_kernel<<<blocks, BLOCK_SIZE>>>(
            partkey.data_ptr<int64_t>(),
            supplycost.data_ptr<double>(),
            availqty.data_ptr<double>(),
            grouped_values.data_ptr<double>(),
            total_sum.data_ptr<double>(),
            n
        );
    }
    
    return {grouped_values, total_sum};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q11_launcher", &q11_launcher, "TPC-H Q11 launcher (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q11_cuda_")
    cu_path = os.path.join(tmpdir, "_q11_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q11_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Q11 using a single fused CUDA kernel for projection, group-by aggregation, 
    and total sum reduction. Replaces expensive `torch.unique` and `scatter_add_` passes 
    with a direct array allocated up to max(ps_partkey). Fast and memory efficient for 
    the dense 1-based keys in TPC-H.
    """
    if ps_partkey.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=ps_partkey.device), \
               torch.empty(0, dtype=torch.float64, device=ps_supplycost.device)
               
    # Evaluate max key to allocate our array correctly
    max_key = int(ps_partkey.max().item())
    
    # Run the fused projection and aggregation
    grouped_values, total_sum = _EXT.q11_launcher(
        ps_partkey, ps_supplycost, ps_availqty, max_key
    )
    
    # Calculate the global threshold
    threshold = total_sum[0] * threshold_fraction
    
    # Filter keys based on HAVING condition
    mask = grouped_values > threshold
    filtered_keys = torch.nonzero(mask, as_tuple=False).squeeze(1)
    filtered_values = grouped_values[filtered_keys]
    
    # Perform sorting by descending value
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
