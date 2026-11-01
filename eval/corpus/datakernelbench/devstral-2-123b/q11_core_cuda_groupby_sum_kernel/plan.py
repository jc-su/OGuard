import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__device__ inline unsigned long long int atomicAdd_ull(unsigned long long int* address, unsigned long long int val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}

__global__ void groupby_sum_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ values,
    int64_t* __restrict__ unique_keys,
    double* __restrict__ out_values,
    size_t* unique_count,
    size_t max_unique,
    size_t n
) {
    __shared__ int64_t s_keys[BLOCK_SIZE];
    __shared__ double s_vals[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    if (idx < n) {
        s_keys[tid] = keys[idx];
        s_vals[tid] = values[idx];
    } else {
        s_keys[tid] = -1;
        s_vals[tid] = 0.0;
    }
    __syncthreads();

    if (tid < blockDim.x && idx < n) {
        int64_t key = s_keys[tid];
        if (key != -1) {
            bool is_leader = true;
            for (int i = 0; i < blockDim.x; i++) {
                if (i != tid && s_keys[i] == key) {
                    is_leader = false;
                    break;
                }
            }

            if (is_leader) {
                double sum_val = s_vals[tid];
                for (int i = 0; i < blockDim.x; i++) {
                    if (i != tid && s_keys[i] == key) {
                        sum_val += s_vals[i];
                        s_keys[i] = -1;
                    }
                }

                size_t pos = atomicAdd_ull(reinterpret_cast<unsigned long long int*>(unique_count), 1);
                if (pos < max_unique) {
                    unique_keys[pos] = key;
                    out_values[pos] = sum_val;
                }
            }
        }
    }
}

void groupby_sum_launcher(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor unique_keys,
    torch::Tensor out_values,
    size_t n,
    size_t max_unique
) {
    size_t unique_count = 0;
    size_t* d_unique_count;
    cudaMalloc(&d_unique_count, sizeof(size_t));
    cudaMemcpy(d_unique_count, &unique_count, sizeof(size_t), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    groupby_sum_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        values.data_ptr<double>(),
        unique_keys.data_ptr<int64_t>(),
        out_values.data_ptr<double>(),
        d_unique_count,
        max_unique,
        n
    );

    cudaMemcpy(&unique_count, d_unique_count, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(d_unique_count);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupby_sum_launcher", &groupby_sum_launcher, "groupby_sum_launcher");
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
    values = ps_supplycost * ps_availqty
    total = torch.sum(values)
    threshold = total * threshold_fraction

    max_unique = 8000000
    unique_keys = torch.zeros(max_unique, dtype=torch.int64, device=ps_partkey.device)
    out_values = torch.zeros(max_unique, dtype=torch.float64, device=ps_partkey.device)

    _EXT.groupby_sum_launcher(ps_partkey, values, unique_keys, out_values, ps_partkey.numel(), max_unique)

    valid_count = (unique_keys != 0).sum().item()
    unique_keys = unique_keys[:valid_count]
    out_values = out_values[:valid_count]

    mask = out_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = out_values[mask]

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

    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    sup_nat = df_supplier.merge(filtered_nation, left_on="s_nationkey", right_on="n_nationkey")
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    keys_t, values_t = _query_core(ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction)

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
