import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 1024

// Binary search for val in sorted array arr of size n.
// Returns index if found, -1 otherwise.
__device__ int find_part_index(const long long* arr, int n, long long val) {
    int left = 0;
    int right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        long long mid_val = arr[mid];
        if (mid_val == val) {
            return mid;
        } else if (mid_val < val) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

__global__ void q17_compute_aggs(
    const long long* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const long long* __restrict__ sorted_part_keys,
    int n_parts,
    double* __restrict__ part_sums,
    double* __restrict__ part_counts,
    int64_t n_rows
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n_rows; i += stride) {
        long long pk = l_partkey[i];
        int p_idx = find_part_index(sorted_part_keys, n_parts, pk);
        if (p_idx != -1) {
            double qty = l_quantity[i];
            atomicAdd(&part_sums[p_idx], qty);
            atomicAdd(&part_counts[p_idx], 1.0);
        }
    }
}

__global__ void q17_compute_final(
    const long long* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const long long* __restrict__ sorted_part_keys,
    int n_parts,
    const double* __restrict__ part_avgs,
    double* __restrict__ out_total,
    int64_t n_rows
) {
    __shared__ double s_block_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    double my_sum = 0.0;

    for (int64_t i = idx; i < n_rows; i += stride) {
        long long pk = l_partkey[i];
        int p_idx = find_part_index(sorted_part_keys, n_parts, pk);
        if (p_idx != -1) {
            double qty = l_quantity[i];
            double avg = part_avgs[p_idx];
            if (qty < 0.2 * avg) {
                my_sum += l_extendedprice[i];
            }
        }
    }

    s_block_sum[tid] = my_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_block_sum[tid] += s_block_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_total, s_block_sum[0]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> q17_launcher_aggs(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor sorted_part_keys,
    int n_parts
) {
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be CUDA");
    TORCH_CHECK(sorted_part_keys.is_cuda(), "sorted_part_keys must be CUDA");
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(sorted_part_keys.scalar_type() == torch::kInt64, "sorted_part_keys must be int64");

    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    sorted_part_keys = sorted_part_keys.contiguous();

    int64_t n_rows = l_partkey.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(l_partkey.device());
    
    torch::Tensor part_sums = torch::zeros({n_parts}, options);
    torch::Tensor part_counts = torch::zeros({n_parts}, options);

    int blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q17_compute_aggs<<<blocks, BLOCK_SIZE>>>(
        (const long long*)l_partkey.data_ptr(),
        (const double*)l_quantity.data_ptr(),
        (const long long*)sorted_part_keys.data_ptr(),
        n_parts,
        (double*)part_sums.data_ptr(),
        (double*)part_counts.data_ptr(),
        n_rows
    );

    return std::make_tuple(part_sums, part_counts);
}

double q17_launcher_final(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor sorted_part_keys,
    torch::Tensor part_avgs,
    int n_parts
) {
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be CUDA");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be CUDA");
    TORCH_CHECK(sorted_part_keys.is_cuda(), "sorted_part_keys must be CUDA");
    TORCH_CHECK(part_avgs.is_cuda(), "part_avgs must be CUDA");
    
    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    sorted_part_keys = sorted_part_keys.contiguous();
    part_avgs = part_avgs.contiguous();

    int64_t n_rows = l_partkey.numel();
    auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat64).device(l_partkey.device()));

    int blocks = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q17_compute_final<<<blocks, BLOCK_SIZE>>>(
        (const long long*)l_partkey.data_ptr(),
        (const double*)l_quantity.data_ptr(),
        (const double*)l_extendedprice.data_ptr(),
        (const long long*)sorted_part_keys.data_ptr(),
        n_parts,
        (const double*)part_avgs.data_ptr(),
        (double*)out.data_ptr(),
        n_rows
    );

    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher_aggs", &q17_launcher_aggs, "Compute part sums and counts");
    m.def("q17_launcher_final", &q17_launcher_final, "Compute final filtered sum");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17: sum(l_extendedprice)/7.0 where l_quantity < 0.2 * avg(l_quantity) over parts
    matching brand/container filters, grouped by partkey.
    Optimized with CUDA kernels for aggregation and filtering.
    """
    n_parts = part_keys.numel()
    if n_parts == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=l_partkey.device)

    # Sort part_keys for binary search in CUDA
    sorted_part_keys, _ = torch.sort(part_keys)
    
    # Pass 1: Compute per-part sum(quantity) and count(*)
    part_sums, part_counts = _EXT.q17_launcher_aggs(
        l_partkey, l_quantity, sorted_part_keys, n_parts
    )
    
    # Compute averages on host (torch)
    # Avoid division by zero
    mask = part_counts > 0
    part_avgs = torch.zeros_like(part_sums)
    part_avgs[mask] = part_sums[mask] / part_counts[mask]
    
    # Pass 2: Filter rows and sum extendedprice
    total = _EXT.q17_launcher_final(
        l_partkey, l_quantity, l_extendedprice, sorted_part_keys, part_avgs, n_parts
    )
    
    return torch.tensor(total / 7.0, dtype=torch.float64, device=l_partkey.device)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small-quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
