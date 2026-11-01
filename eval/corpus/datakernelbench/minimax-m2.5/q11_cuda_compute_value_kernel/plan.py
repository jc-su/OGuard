import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// Kernel to compute value = supplycost * availqty (fused multiply)
__global__ void compute_value_kernel(
    const int64_t* __restrict__ partkey_in,
    const double* __restrict__ supplycost,
    const double* __restrict__ availqty,
    int64_t* __restrict__ partkey_out,
    double* __restrict__ value_out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        partkey_out[idx] = partkey_in[idx];
        value_out[idx] = supplycost[idx] * availqty[idx];
    }
}

// Launcher that computes value = supplycost * availqty
// Returns tuple of (partkey_copy, value)
std::vector<torch::Tensor> compute_value_launcher(
    torch::Tensor partkey_in,
    torch::Tensor supplycost,
    torch::Tensor availqty
) {
    TORCH_CHECK(partkey_in.scalar_type() == torch::kInt64);
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64);
    TORCH_CHECK(availqty.scalar_type() == torch::kFloat64);
    
    // Move to GPU if needed
    if (!partkey_in.is_cuda()) partkey_in = partkey_in.cuda();
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!availqty.is_cuda()) availqty = availqty.cuda();
    
    partkey_in = partkey_in.contiguous();
    supplycost = supplycost.contiguous();
    availqty = availqty.contiguous();
    
    int64_t n = partkey_in.numel();
    
    // Allocate output arrays
    auto partkey_out = torch::empty({n}, partkey_in.options());
    auto value_out = torch::empty({n}, supplycost.options());
    
    // Launch kernel to compute value
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_value_kernel<<<blocks, BLOCK_SIZE>>>(
        partkey_in.data_ptr<int64_t>(),
        supplycost.data_ptr<double>(),
        availqty.data_ptr<double>(),
        partkey_out.data_ptr<int64_t>(),
        value_out.data_ptr<double>(),
        n
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(err));
    }
    
    std::vector<torch::Tensor> results;
    results.push_back(partkey_out);
    results.push_back(value_out);
    
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_value_launcher", &compute_value_launcher, 
          "Compute value = supplycost * availqty for TPC-H Q11");
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
    Optimized CUDA path for TPC-H Q11 groupby aggregation.
    
    Uses CUDA kernel to compute value = supplycost * availqty (fused multiply),
    then leverages PyTorch's optimized primitives for unique/sort/reduce.
    
    This approach fuses the multiply operation in a single CUDA kernel,
    reducing memory traffic by keeping both source columns in registers
    and writing only the result.
    
    Pipeline:
    1. CUDA kernel computes value = supplycost * availqty (fused in one pass)
    2. torch.unique + scatter_add for groupby aggregation
    3. Apply having filter
    4. Sort by value DESC
    """
    # Ensure tensors are on GPU and contiguous
    ps_partkey = ps_partkey.cuda().contiguous()
    ps_supplycost = ps_supplycost.cuda().contiguous()
    ps_availqty = ps_availqty.cuda().contiguous()
    
    # Step 1: Compute value using CUDA kernel (fused multiply)
    partkey_copy, value = _EXT.compute_value_launcher(
        ps_partkey, ps_supplycost, ps_availqty
    )
    
    # Step 2: Group by partkey using unique + scatter_add (PyTorch optimized)
    unique_keys, inverse = torch.unique(partkey_copy, return_inverse=True)
    grouped_values = torch.zeros(unique_keys.shape[0], dtype=value.dtype, device=value.device)
    grouped_values.scatter_add_(0, inverse, value)
    
    # Step 3: Compute threshold and filter
    total = torch.sum(value)
    threshold = total * threshold_fraction
    
    # HAVING filter: value > threshold
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]
    
    # Step 4: ORDER BY value DESC
    if filtered_keys.numel() > 0:
        sort_idx = torch.argsort(filtered_values, descending=True)
        filtered_keys = filtered_keys[sort_idx]
        filtered_values = filtered_values[sort_idx]
    
    return filtered_keys, filtered_values


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
                 
                 Optimization: cuDF handles joins (strings, small dims), CUDA handles
                 the hot path: fused kernel computes value = supplycost * availqty,
                 then PyTorch's optimized primitives handle unique/sort/reduce.
                 
                 The fused CUDA kernel reduces memory bandwidth by reading both
                 source columns once and computing the product in registers before
                 writing the result.
    
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
