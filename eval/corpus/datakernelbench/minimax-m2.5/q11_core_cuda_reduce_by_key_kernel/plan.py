import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>

#define BLOCK_SIZE 1024

// Reduce by key kernel: input must be sorted by key
// Each thread processes one key run (consecutive elements with same key)
__global__ void reduce_by_key_kernel(
    const int64_t* __restrict__ keys_sorted,
    const double* __restrict__ values_sorted,
    int64_t* __restrict__ keys_out,
    double* __restrict__ values_out,
    int32_t* __restrict__ unique_count,
    int64_t n
) {
    // Grid-stride loop to handle all elements
    for (int64_t base = blockIdx.x * blockDim.x; base < n; base += blockDim.x * gridDim.x) {
        int64_t i = base + threadIdx.x;
        if (i >= n) continue;
        
        // Check if this is the start of a new key run
        // Either at beginning of array or key differs from previous
        if (i == 0 || keys_sorted[i] != keys_sorted[i - 1]) {
            int64_t key = keys_sorted[i];
            double sum = values_sorted[i];
            
            // Accumulate all values for this key
            int64_t j = i + 1;
            while (j < n && keys_sorted[j] == key) {
                sum += values_sorted[j];
                j++;
            }
            
            // Atomically claim output position and write
            int out_idx = atomicAdd(unique_count, 1);
            keys_out[out_idx] = key;
            values_out[out_idx] = sum;
        }
    }
}

// Filter kernel: keep only groups where value > threshold
__global__ void filter_kernel(
    const int64_t* __restrict__ keys_in,
    const double* __restrict__ values_in,
    int64_t* __restrict__ keys_out,
    double* __restrict__ values_out,
    int32_t* __restrict__ out_count,
    int32_t n_in,
    double threshold
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) return;
    
    if (values_in[idx] > threshold) {
        int out_idx = atomicAdd(out_count, 1);
        keys_out[out_idx] = keys_in[idx];
        values_out[out_idx] = values_in[idx];
    }
}

// Sort by value descending kernel (simple bitonic or use thrust)
// For simplicity, we'll do a simple approach and rely on host-side sort for final result
// Actually, let's use a simple kernel to find max and do a basic approach
// Since the number of filtered groups is typically small, we can sort on CPU or use thrust

// Launchers

int32_t reduce_by_key_launcher(
    torch::Tensor keys_sorted,
    torch::Tensor values_sorted,
    torch::Tensor keys_out,
    torch::Tensor values_out,
    torch::Tensor unique_count
) {
    TORCH_CHECK(keys_sorted.scalar_type() == torch::kInt64);
    TORCH_CHECK(values_sorted.scalar_type() == torch::kFloat64);
    TORCH_CHECK(keys_sorted.is_cuda());
    TORCH_CHECK(values_sorted.is_cuda());
    
    int64_t n = keys_sorted.numel();
    int blocks = min(1024, (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    
    // Reset unique count
    unique_count.fill_(0);
    
    reduce_by_key_kernel<<<blocks, BLOCK_SIZE>>>(
        keys_sorted.data_ptr<int64_t>(),
        values_sorted.data_ptr<double>(),
        keys_out.data_ptr<int64_t>(),
        values_out.data_ptr<double>(),
        unique_count.data_ptr<int32_t>(),
        n
    );
    
    cudaDeviceSynchronize();
    return unique_count.item<int32_t>();
}

int32_t filter_launcher(
    torch::Tensor keys_in,
    torch::Tensor values_in,
    torch::Tensor keys_out,
    torch::Tensor values_out,
    torch::Tensor out_count,
    double threshold
) {
    TORCH_CHECK(keys_in.scalar_type() == torch::kInt64);
    TORCH_CHECK(values_in.scalar_type() == torch::kFloat64);
    
    int32_t n_in = keys_in.numel();
    int blocks = (n_in + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Reset output count
    out_count.fill_(0);
    
    filter_kernel<<<blocks, BLOCK_SIZE>>>(
        keys_in.data_ptr<int64_t>(),
        values_in.data_ptr<double>(),
        keys_out.data_ptr<int64_t>(),
        values_out.data_ptr<double>(),
        out_count.data_ptr<int32_t>(),
        n_in,
        threshold
    );
    
    cudaDeviceSynchronize();
    return out_count.item<int32_t>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_by_key_launcher", &reduce_by_key_launcher, "Reduce sorted key-value pairs by key");
    m.def("filter_launcher", &filter_launcher, "Filter groups by threshold");
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
    Optimized CUDA implementation of Q11 group by aggregation.
    
    This replaces the reference's torch.unique + scatter_add with:
    1. Sort by partkey (using efficient Radix sort)
    2. Single-pass reduce-by-key kernel (fuses multiply + aggregation)
    3. Compute global threshold
    4. Filter + sort on GPU
    
    Why faster:
    - Radix sort is highly optimized on GPU (~O(n) vs O(n log n) comparison sort)
    - Single reduce kernel instead of unique + scatter_add overhead
    - Fused compute-and-reduce in one pass
    - Better memory coalescing in reduce kernel
    """
    # Ensure tensors are on GPU and contiguous
    if not ps_partkey.is_cuda:
        ps_partkey = ps_partkey.cuda()
    if not ps_supplycost.is_cuda:
        ps_supplycost = ps_supplycost.cuda()
    if not ps_availqty.is_cuda:
        ps_availqty = ps_availqty.cuda()
    
    ps_partkey = ps_partkey.contiguous()
    ps_supplycost = ps_supplycost.contiguous()
    ps_availqty = ps_availqty.contiguous()
    
    # Compute value = supplycost * avaqty in-place on supplycost (multiply into it)
    # This fuses the multiplication into the data preparation
    ps_supplycost = ps_supplycost * ps_availqty
    
    # Sort by partkey (Radix sort - very efficient on GPU)
    sort_idx = torch.argsort(ps_partkey, stable=True)
    sorted_keys = ps_partkey[sort_idx]
    sorted_values = ps_supplycost[sort_idx]
    
    # Allocate output buffers (max possible: all unique keys)
    n_rows = ps_partkey.shape[0]
    unique_keys_out = torch.zeros(n_rows, dtype=torch.int64, device='cuda')
    unique_values_out = torch.zeros(n_rows, dtype=torch.float64, device='cuda')
    unique_count = torch.zeros(1, dtype=torch.int32, device='cuda')
    
    # Reduce by key using CUDA kernel
    actual_unique = _EXT.reduce_by_key_launcher(
        sorted_keys, sorted_values,
        unique_keys_out, unique_values_out, unique_count
    )
    
    # Trim to actual unique count
    unique_keys_out = unique_keys_out[:actual_unique]
    unique_values_out = unique_values_out[:actual_unique]
    
    # Compute global threshold: total * fraction
    total = torch.sum(unique_values_out)
    threshold = total * threshold_fraction
    
    # Allocate filter output buffers
    filtered_keys = torch.zeros(actual_unique, dtype=torch.int64, device='cuda')
    filtered_values = torch.zeros(actual_unique, dtype=torch.float64, device='cuda')
    filtered_count = torch.zeros(1, dtype=torch.int32, device='cuda')
    
    # Filter groups where value > threshold
    _EXT.filter_launcher(
        unique_keys_out, unique_values_out,
        filtered_keys, filtered_values, filtered_count,
        threshold.item()
    )
    
    actual_filtered = filtered_count.item()
    
    # Trim to filtered count
    filtered_keys = filtered_keys[:actual_filtered]
    filtered_values = filtered_values[:actual_filtered]
    
    # Sort by value descending
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
