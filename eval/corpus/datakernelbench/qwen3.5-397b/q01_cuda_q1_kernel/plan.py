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

#define BLOCK_SIZE 256
#define MAX_GROUPS 30
#define NUM_AGGS 6

// Aggregates per group (6 values):
// 0: sum_qty, 1: sum_base_price, 2: sum_disc_price
// 3: sum_charge, 4: sum_disc, 5: count (as double)

__global__ void q1_kernel(
    const int32_t* __restrict__ rf_codes,
    const int32_t* __restrict__ ls_codes,
    const double* __restrict__ quantity,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    double* __restrict__ out_sums,
    int64_t n_elements
) {
    // Shared memory for block-level group totals
    __shared__ double s_final[MAX_GROUPS * NUM_AGGS];
    
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory (thread 0 only)
    if (tid == 0) {
        for (int i = 0; i < MAX_GROUPS * NUM_AGGS; i++) {
            s_final[i] = 0.0;
        }
    }
    __syncthreads();
    
    // Each thread processes its element and accumulates via shared memory atomics
    if (idx < n_elements) {
        if (shipdate_days[idx] <= shipdate_hi_days) {
            int32_t rf = rf_codes[idx];
            int32_t ls = ls_codes[idx];
            // Encode group: rf * 10 + ls (codes are small 0-based integers)
            int32_t g = rf * 10 + ls;
            
            if (g >= 0 && g < MAX_GROUPS) {
                double qty = quantity[idx];
                double p = price[idx];
                double d = discount[idx];
                double t = tax[idx];
                double disc_price = p * (1.0 - d);
                double charge = disc_price * (1.0 + t);
                
                int f_base = g * NUM_AGGS;
                atomicAdd(&s_final[f_base + 0], qty);
                atomicAdd(&s_final[f_base + 1], p);
                atomicAdd(&s_final[f_base + 2], disc_price);
                atomicAdd(&s_final[f_base + 3], charge);
                atomicAdd(&s_final[f_base + 4], d);
                atomicAdd(&s_final[f_base + 5], 1.0);
            }
        }
    }
    __syncthreads();
    
    // Thread 0 writes block totals to global memory
    if (tid == 0) {
        for (int i = 0; i < MAX_GROUPS * NUM_AGGS; i++) {
            if (s_final[i] != 0.0) {
                atomicAdd(&out_sums[i], s_final[i]);
            }
        }
    }
}

std::vector<torch::Tensor> q1_launcher(
    torch::Tensor rf_codes,
    torch::Tensor ls_codes,
    torch::Tensor quantity,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor tax,
    torch::Tensor shipdate_days,
    int32_t shipdate_hi_days
) {
    TORCH_CHECK(rf_codes.is_cuda(), "rf_codes must be CUDA");
    TORCH_CHECK(ls_codes.is_cuda(), "ls_codes must be CUDA");
    TORCH_CHECK(quantity.is_cuda(), "quantity must be CUDA");
    TORCH_CHECK(price.is_cuda(), "price must be CUDA");
    TORCH_CHECK(discount.is_cuda(), "discount must be CUDA");
    TORCH_CHECK(tax.is_cuda(), "tax must be CUDA");
    TORCH_CHECK(shipdate_days.is_cuda(), "shipdate_days must be CUDA");

    TORCH_CHECK(rf_codes.scalar_type() == torch::kInt32);
    TORCH_CHECK(ls_codes.scalar_type() == torch::kInt32);
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(tax.scalar_type() == torch::kFloat64);
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);

    int64_t n = rf_codes.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(rf_codes.device());
    auto out_sums = torch::zeros({MAX_GROUPS * NUM_AGGS}, options);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q1_kernel<<<blocks, BLOCK_SIZE>>>(
        rf_codes.data_ptr<int32_t>(),
        ls_codes.data_ptr<int32_t>(),
        quantity.data_ptr<double>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        tax.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        shipdate_hi_days,
        out_sums.data_ptr<double>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {out_sums};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC-H Q1 CUDA Kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q1_cuda_")
    cu_path = os.path.join(tmpdir, "_q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # H100 is sm_90
    return load(
        name="q1_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report. Uses a fused CUDA kernel for filter, project, and group-by aggregate.
                 Reduces HBM passes by computing all aggregates in one pass over the lineitem table.
                 Uses shared memory to accumulate block-level partials before global atomic update.
                 BLOCK_SIZE=256 keeps shared memory under 48KB limit while maintaining occupancy.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns to integer codes
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    
    # Preserve categories for decoding later
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    # Convert to torch tensors on GPU
    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy(), device='cuda')
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy(), device='cuda')

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64, device='cuda')
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64, device='cuda')
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64, device='cuda')
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64, device='cuda')
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(), device='cuda'
    )

    # Launch CUDA kernel
    # Returns [out_sums] where out_sums is [MAX_GROUPS * NUM_AGGS]
    out_sums_list = _EXT.q1_launcher(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )
    out_sums = out_sums_list[0]  # Shape [30 * 6]

    # Post-process on host/CPU
    # out_sums layout: for g in 0..29: [sum_qty, sum_base, sum_disc_p, sum_charge, sum_disc, count]
    out_sums_cpu = out_sums.cpu()
    
    results = []
    for g in range(30):
        base = g * 6
        count = out_sums_cpu[base + 5].item()
        if count > 0:
            rf_code = g // 10
            ls_code = g % 10
            
            # Decode strings
            rf_str = rf_categories[rf_code] if rf_code < len(rf_categories) else "?"
            ls_str = ls_categories[ls_code] if ls_code < len(ls_categories) else "?"
            
            sum_qty = out_sums_cpu[base + 0].item()
            sum_base_price = out_sums_cpu[base + 1].item()
            sum_disc_price = out_sums_cpu[base + 2].item()
            sum_charge = out_sums_cpu[base + 3].item()
            sum_disc = out_sums_cpu[base + 4].item()
            
            # Compute averages
            avg_qty = sum_qty / count
            avg_price = sum_base_price / count
            avg_disc = sum_disc / count
            
            results.append({
                "l_returnflag": rf_str,
                "l_linestatus": ls_str,
                "sum_qty": sum_qty,
                "sum_base_price": sum_base_price,
                "sum_disc_price": sum_disc_price,
                "sum_charge": sum_charge,
                "avg_qty": avg_qty,
                "avg_price": avg_price,
                "avg_disc": avg_disc,
                "count_order": int(count),
            })
    
    # Create DataFrame
    df_res = cudf.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(["l_returnflag", "l_linestatus"]).reset_index(drop=True)
        
    return df_res
