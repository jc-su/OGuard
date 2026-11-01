import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <vector>

__global__ void q1_kernel(
    const int32_t* __restrict__ rf_ptr,
    const int32_t* __restrict__ ls_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi,
    int32_t num_ls,
    int32_t num_groups,
    int32_t num_sets,
    double* __restrict__ sum_qty,
    double* __restrict__ sum_base_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    int64_t* __restrict__ count_order,
    int64_t n_elements
) {
    int warp_id = threadIdx.x / 32;
    int set_id = warp_id % num_sets;
    
    // Shared memory allocated dynamically
    extern __shared__ double smem[];
    int row_size = num_groups * 6;
    double* my_smem = smem + set_id * row_size;
    
    // Initialize shared memory for this block
    for (int i = threadIdx.x; i < num_sets * row_size; i += blockDim.x) {
        smem[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        if (shipdate_ptr[idx] <= shipdate_hi) {
            int group = rf_ptr[idx] * num_ls + ls_ptr[idx];
            double qty = qty_ptr[idx];
            double price = price_ptr[idx];
            double disc = disc_ptr[idx];
            double tax = tax_ptr[idx];
            
            atomicAdd(&my_smem[group * 6 + 0], qty);
            atomicAdd(&my_smem[group * 6 + 1], price);
            atomicAdd(&my_smem[group * 6 + 2], price * (1.0 - disc));
            atomicAdd(&my_smem[group * 6 + 3], price * (1.0 - disc) * (1.0 + tax));
            atomicAdd(&my_smem[group * 6 + 4], disc);
            
            // Re-use the 6th double as an unsigned long long counter
            unsigned long long* w_count = (unsigned long long*)&my_smem[group * 6 + 5];
            atomicAdd(w_count, 1ULL);
        }
    }
    __syncthreads();

    // Reduce local accumulators to global memory
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        double t_qty = 0, t_price = 0, t_disc_price = 0, t_charge = 0, t_disc = 0;
        unsigned long long t_count = 0;
        
        for (int s = 0; s < num_sets; ++s) {
            double* s_smem = smem + s * row_size;
            t_qty += s_smem[g * 6 + 0];
            t_price += s_smem[g * 6 + 1];
            t_disc_price += s_smem[g * 6 + 2];
            t_charge += s_smem[g * 6 + 3];
            t_disc += s_smem[g * 6 + 4];
            t_count += *(unsigned long long*)&s_smem[g * 6 + 5];
        }
        
        if (t_count > 0) {
            atomicAdd(&sum_qty[g], t_qty);
            atomicAdd(&sum_base_price[g], t_price);
            atomicAdd(&sum_disc_price[g], t_disc_price);
            atomicAdd(&sum_charge[g], t_charge);
            atomicAdd(&sum_disc[g], t_disc);
            atomicAdd((unsigned long long*)&count_order[g], t_count);
        }
    }
}

std::vector<torch::Tensor> q1_launcher(
    torch::Tensor rf, torch::Tensor ls, torch::Tensor qty,
    torch::Tensor price, torch::Tensor disc, torch::Tensor tax,
    torch::Tensor shipdate, int32_t shipdate_hi, int32_t num_ls, int32_t num_groups
) {
    int64_t n = qty.numel();
    auto opts_f64 = qty.options().dtype(torch::kFloat64);
    auto opts_i64 = qty.options().dtype(torch::kInt64);
    
    auto sum_qty = torch::zeros({num_groups}, opts_f64);
    auto sum_base_price = torch::zeros({num_groups}, opts_f64);
    auto sum_disc_price = torch::zeros({num_groups}, opts_f64);
    auto sum_charge = torch::zeros({num_groups}, opts_f64);
    auto sum_disc = torch::zeros({num_groups}, opts_f64);
    auto count_order = torch::zeros({num_groups}, opts_i64);

    int block_size = 512;
    int blocks = (n + block_size - 1) / block_size;
    int num_warps = block_size / 32;
    int row_size = num_groups * 6;
    int num_sets = num_warps;
    
    // Respect maximum shared memory constraints (e.g. 48 KB)
    int max_smem_bytes = 49152;
    if (num_sets * row_size * sizeof(double) > max_smem_bytes) {
        num_sets = max_smem_bytes / (row_size * sizeof(double));
        if (num_sets < 1) num_sets = 1;
    }
    int smem_size = num_sets * row_size * sizeof(double);

    q1_kernel<<<blocks, block_size, smem_size>>>(
        rf.data_ptr<int32_t>(), ls.data_ptr<int32_t>(),
        qty.data_ptr<double>(), price.data_ptr<double>(),
        disc.data_ptr<double>(), tax.data_ptr<double>(),
        shipdate.data_ptr<int32_t>(), shipdate_hi, num_ls, num_groups, num_sets,
        sum_qty.data_ptr<double>(), sum_base_price.data_ptr<double>(),
        sum_disc_price.data_ptr<double>(), sum_charge.data_ptr<double>(),
        sum_disc.data_ptr<double>(), count_order.data_ptr<int64_t>(), n
    );
    
    return {sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count_order};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC-H Q1 fused scan-agg kernel (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q1_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
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
    
    Description: TPC-H Q1 fully fused in CUDA. Filter, math, and group aggregation happens in one
    memory-bound kernel pass with warp-distributed fast shared-memory reductions, greatly accelerating 
    execution over multiple framework hops.
    
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode categories identically to reference
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_cats = rf_codes.cat.categories.to_pandas().tolist()
    ls_cats = ls_codes.cat.categories.to_pandas().tolist()
    
    num_ls = len(ls_cats)
    num_groups = len(rf_cats) * num_ls

    t_rf = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    t_ls = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())
    t_qty = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    t_price = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    t_disc = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    t_tax = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    t_shipdate = torch.as_tensor((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32").to_cupy())

    # Launch fused grouping kernel
    sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count_order = _EXT.q1_launcher(
        t_rf.contiguous(), t_ls.contiguous(), t_qty.contiguous(),
        t_price.contiguous(), t_disc.contiguous(), t_tax.contiguous(),
        t_shipdate.contiguous(), shipdate_hi_days, num_ls, num_groups
    )

    # Filter out empty groups 
    valid_mask = count_order > 0
    group_idx = torch.nonzero(valid_mask).squeeze(-1)
    
    rf_idx = group_idx // num_ls
    ls_idx = group_idx % num_ls

    # Extracted data arrays are inherently sorted since group_idx scales monotonically
    sum_qty = sum_qty[valid_mask]
    sum_base_price = sum_base_price[valid_mask]
    sum_disc_price = sum_disc_price[valid_mask]
    sum_charge = sum_charge[valid_mask]
    sum_disc = sum_disc[valid_mask]
    count_val = count_order[valid_mask]

    # Averages
    avg_qty = sum_qty / count_val.to(torch.float64)
    avg_price = sum_base_price / count_val.to(torch.float64)
    avg_disc = sum_disc / count_val.to(torch.float64)

    # Decode categories back to string objects
    rf_out = rf_idx.cpu().tolist()
    ls_out = ls_idx.cpu().tolist()
    rf_strs = [rf_cats[i] for i in rf_out]
    ls_strs = [ls_cats[i] for i in ls_out]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_val,
    })
