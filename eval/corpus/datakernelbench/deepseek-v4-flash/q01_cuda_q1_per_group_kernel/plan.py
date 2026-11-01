import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_GROUPS 256

__global__ void q1_per_group_kernel(
    const int32_t* __restrict__ rf_code_ptr,
    const int32_t* __restrict__ ls_code_ptr,
    const double* __restrict__ qty_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    const double* __restrict__ tax_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t shipdate_hi_days,
    double* __restrict__ out_sum_qty,
    double* __restrict__ out_sum_base_price,
    double* __restrict__ out_sum_disc_price,
    double* __restrict__ out_sum_charge,
    double* __restrict__ out_sum_disc,
    int64_t* __restrict__ out_count,
    int32_t* __restrict__ out_rf,
    int32_t* __restrict__ out_ls,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = rf_code_ptr[idx] + 1;  // shift by 1 to avoid key 0
            int32_t ls = ls_code_ptr[idx] + 1;
            int32_t gkey = rf * 256 + ls;
            
            double q = qty_ptr[idx];
            double p = price_ptr[idx];
            double d = disc_ptr[idx];
            double t = tax_ptr[idx];
            double dp = p * (1.0 - d);
            double ch = dp * (1.0 + t);
            
            atomicAdd(out_sum_qty + gkey, q);
            atomicAdd(out_sum_base_price + gkey, p);
            atomicAdd(out_sum_disc_price + gkey, dp);
            atomicAdd(out_sum_charge + gkey, ch);
            atomicAdd(out_sum_disc + gkey, d);
            atomicAdd((unsigned long long*)(out_count + gkey), 1ULL);
            if (atomicCAS(out_rf + gkey, 0, rf) == 0) {}
            if (atomicCAS(out_ls + gkey, 0, ls) == 0) {}
        }
    }
}

void launch_q1_fused(
    torch::Tensor rf_code,
    torch::Tensor ls_code,
    torch::Tensor qty,
    torch::Tensor price,
    torch::Tensor disc,
    torch::Tensor tax,
    torch::Tensor shipdate_days,
    int32_t shipdate_hi_days,
    torch::Tensor out_sum_qty,
    torch::Tensor out_sum_base_price,
    torch::Tensor out_sum_disc_price,
    torch::Tensor out_sum_charge,
    torch::Tensor out_sum_disc,
    torch::Tensor out_count,
    torch::Tensor out_rf,
    torch::Tensor out_ls
) {
    int64_t n = rf_code.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMemset(out_sum_qty.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(double));
    cudaMemset(out_sum_base_price.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(double));
    cudaMemset(out_sum_disc_price.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(double));
    cudaMemset(out_sum_charge.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(double));
    cudaMemset(out_sum_disc.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(double));
    cudaMemset(out_count.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(int64_t));
    cudaMemset(out_rf.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(int32_t));
    cudaMemset(out_ls.data_ptr(), 0, MAX_GROUPS * MAX_GROUPS * sizeof(int32_t));
    
    q1_per_group_kernel<<<blocks, BLOCK_SIZE>>>(
        rf_code.data_ptr<int32_t>(),
        ls_code.data_ptr<int32_t>(),
        qty.data_ptr<double>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        tax.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        shipdate_hi_days,
        out_sum_qty.data_ptr<double>(),
        out_sum_base_price.data_ptr<double>(),
        out_sum_disc_price.data_ptr<double>(),
        out_sum_charge.data_ptr<double>(),
        out_sum_disc.data_ptr<double>(),
        out_count.data_ptr<int64_t>(),
        out_rf.data_ptr<int32_t>(),
        out_ls.data_ptr<int32_t>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_q1_fused", &launch_q1_fused, "Fused per-group aggregate for TPC-H Q1");
}
'''

_EXT = None

def _load_cuda():
    global _EXT
    if _EXT is not None:
        return _EXT
    tmpdir = tempfile.mkdtemp(prefix="tpch_q1_cuda_")
    cu_path = os.path.join(tmpdir, "_q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    _EXT = load(
        name="tpch_q1_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )
    return _EXT

def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 with fully fused CUDA kernel doing per-group aggregation.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df = tables["lineitem"]
    n = len(df)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Convert string columns to categorical codes using cuDF
    rf_series = df["l_returnflag"].astype("category")
    ls_series = df["l_linestatus"].astype("category")
    
    rf_cats = rf_series.cat.categories.to_arrow().to_pylist()
    ls_cats = ls_series.cat.categories.to_arrow().to_pylist()
    
    rf_code = torch.as_tensor(rf_series.cat.codes.astype("int32").to_cupy(), dtype=torch.int32, device='cuda')
    ls_code = torch.as_tensor(ls_series.cat.codes.astype("int32").to_cupy(), dtype=torch.int32, device='cuda')
    qty = torch.as_tensor(df["l_quantity"].to_cupy(), dtype=torch.float64, device='cuda')
    price = torch.as_tensor(df["l_extendedprice"].to_cupy(), dtype=torch.float64, device='cuda')
    disc = torch.as_tensor(df["l_discount"].to_cupy(), dtype=torch.float64, device='cuda')
    tax = torch.as_tensor(df["l_tax"].to_cupy(), dtype=torch.float64, device='cuda')
    shipdate_days = torch.as_tensor(
        ((df["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32, device='cuda'
    )

    MAX_GROUPS = 256
    total_slots = MAX_GROUPS * MAX_GROUPS
    
    out_sum_qty = torch.zeros(total_slots, dtype=torch.float64, device='cuda')
    out_sum_base_price = torch.zeros(total_slots, dtype=torch.float64, device='cuda')
    out_sum_disc_price = torch.zeros(total_slots, dtype=torch.float64, device='cuda')
    out_sum_charge = torch.zeros(total_slots, dtype=torch.float64, device='cuda')
    out_sum_disc = torch.zeros(total_slots, dtype=torch.float64, device='cuda')
    out_count = torch.zeros(total_slots, dtype=torch.int64, device='cuda')
    out_rf = torch.zeros(total_slots, dtype=torch.int32, device='cuda')
    out_ls = torch.zeros(total_slots, dtype=torch.int32, device='cuda')

    ext = _load_cuda()
    ext.launch_q1_fused(
        rf_code, ls_code, qty, price, disc, tax, shipdate_days,
        shipdate_hi_days, out_sum_qty, out_sum_base_price, out_sum_disc_price,
        out_sum_charge, out_sum_disc, out_count, out_rf, out_ls
    )

    # Find active groups
    active_mask = out_count > 0
    active_indices = torch.nonzero(active_mask).squeeze(1)
    
    if active_indices.numel() == 0:
        return cudf.DataFrame({
            "l_returnflag": cudf.Series([], dtype='object'),
            "l_linestatus": cudf.Series([], dtype='object'),
            "sum_qty": cudf.Series([], dtype='float64'),
            "sum_base_price": cudf.Series([], dtype='float64'),
            "sum_disc_price": cudf.Series([], dtype='float64'),
            "sum_charge": cudf.Series([], dtype='float64'),
            "avg_qty": cudf.Series([], dtype='float64'),
            "avg_price": cudf.Series([], dtype='float64'),
            "avg_disc": cudf.Series([], dtype='float64'),
            "count_order": cudf.Series([], dtype='int64'),
        })
    
    # Gather results
    rf_raw = out_rf[active_indices] - 1  # undo shift
    ls_raw = out_ls[active_indices] - 1
    sum_qty = out_sum_qty[active_indices]
    sum_base_price = out_sum_base_price[active_indices]
    sum_disc_price = out_sum_disc_price[active_indices]
    sum_charge = out_sum_charge[active_indices]
    sum_disc = out_sum_disc[active_indices]
    count_order = out_count[active_indices]

    avg_qty = sum_qty / count_order.to(torch.float64)
    avg_price = sum_base_price / count_order.to(torch.float64)
    avg_disc = sum_disc / count_order.to(torch.float64)

    # Sort by (rf, ls)
    sort_key = rf_raw.to(torch.int64) * 256 + ls_raw.to(torch.int64)
    order = torch.argsort(sort_key)
    
    rf_out = rf_raw[order]
    ls_out = ls_raw[order]
    sum_qty = sum_qty[order]
    sum_base_price = sum_base_price[order]
    sum_disc_price = sum_disc_price[order]
    sum_charge = sum_charge[order]
    avg_qty = avg_qty[order]
    avg_price = avg_price[order]
    avg_disc = avg_disc[order]
    count_order = count_order[order]
    
    rf_vals = rf_out.cpu().tolist()
    ls_vals = ls_out.cpu().tolist()
    rf_strs = [rf_cats[i] for i in rf_vals]
    ls_strs = [ls_cats[i] for i in ls_vals]
    
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
        "count_order": count_order,
    })
