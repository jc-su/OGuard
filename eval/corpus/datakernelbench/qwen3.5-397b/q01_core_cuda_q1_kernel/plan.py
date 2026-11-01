import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define MAX_GROUPS 1024
#define NUM_AGGS 6

__global__ void q1_kernel(
    const int32_t* rf,
    const int32_t* ls,
    const double* qty,
    const double* price,
    const double* disc,
    const double* tax,
    const int32_t* shipdate,
    int32_t shipdate_hi,
    double* accum,
    int64_t n
) {
    extern __shared__ double shmem[];
    // Initialize shared memory
    for (int i = threadIdx.x; i < MAX_GROUPS * NUM_AGGS; i += blockDim.x) {
        shmem[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int64_t i = idx; i < n; i += stride) {
        if (shipdate[i] <= shipdate_hi) {
            int32_t r = rf[i];
            int32_t l = ls[i];
            int32_t key = r * 256 + l;
            if (key < MAX_GROUPS) {
                double q = qty[i];
                double p = price[i];
                double d = disc[i];
                double t = tax[i];
                double dp = p * (1.0 - d);
                double ch = dp * (1.0 + t);
                int base = key * NUM_AGGS;
                // Accumulate in shared memory
                atomicAdd(&shmem[base + 0], q);
                atomicAdd(&shmem[base + 1], p);
                atomicAdd(&shmem[base + 2], dp);
                atomicAdd(&shmem[base + 3], ch);
                atomicAdd(&shmem[base + 4], d);
                atomicAdd(&shmem[base + 5], 1.0);
            }
        }
    }
    __syncthreads();

    // Write back to global memory (only thread 0 per block)
    if (threadIdx.x == 0) {
        for (int k = 0; k < MAX_GROUPS; k++) {
            int base = k * NUM_AGGS;
            double count = shmem[base + 5];
            if (count > 0.0) {
                atomicAdd(&accum[base + 0], shmem[base + 0]);
                atomicAdd(&accum[base + 1], shmem[base + 1]);
                atomicAdd(&accum[base + 2], shmem[base + 2]);
                atomicAdd(&accum[base + 3], shmem[base + 3]);
                atomicAdd(&accum[base + 4], shmem[base + 4]);
                atomicAdd(&accum[base + 5], count);
            }
        }
    }
}

torch::Tensor q1_launcher(
    torch::Tensor rf,
    torch::Tensor ls,
    torch::Tensor qty,
    torch::Tensor price,
    torch::Tensor disc,
    torch::Tensor tax,
    torch::Tensor shipdate,
    int32_t shipdate_hi
) {
    // Ensure contiguous and CUDA
    rf = rf.contiguous().cuda();
    ls = ls.contiguous().cuda();
    qty = qty.contiguous().cuda();
    price = price.contiguous().cuda();
    disc = disc.contiguous().cuda();
    tax = tax.contiguous().cuda();
    shipdate = shipdate.contiguous().cuda();

    int64_t n = qty.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto accum = torch::zeros({MAX_GROUPS * NUM_AGGS}, options);

    // Launch enough blocks to saturate GPU, but not too many to cause overhead
    // H100 has 132 SMs. 2048 blocks is sufficient.
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 2048) blocks = 2048;
    if (blocks < 1) blocks = 1;
    
    int shared_mem_size = MAX_GROUPS * NUM_AGGS * sizeof(double);
    
    // Request max dynamic shared memory to ensure 48KB+ is available
    cudaFuncSetAttribute(q1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    q1_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        rf.data_ptr<int32_t>(),
        ls.data_ptr<int32_t>(),
        qty.data_ptr<double>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        tax.data_ptr<double>(),
        shipdate.data_ptr<int32_t>(),
        shipdate_hi,
        accum.data_ptr<double>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return accum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC-H Q1 CUDA Aggregation");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q1_cuda_")
    cu_path = os.path.join(tmpdir, "_q1_kernels.cu")
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

def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Optimized CUDA implementation of TPC-H Q1 aggregation.
    Fuses filter, projection, and group-by aggregation into a single kernel.
    Uses shared memory for block-level reduction to minimize global atomics.
    """
    # Launch CUDA kernel
    accum = _EXT.q1_launcher(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
    )
    
    # Copy to CPU for post-processing
    accum_cpu = accum.cpu()
    # Reshape to (MAX_GROUPS, NUM_AGGS)
    accum_cpu = accum_cpu.view(-1, 6)
    
    # Extract valid groups (count > 0)
    count_col = accum_cpu[:, 5]
    mask = count_col > 0.0
    valid = accum_cpu[mask]
    
    if valid.shape[0] == 0:
        # Return empty tensors with correct dtype
        device = l_quantity.device
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )
    
    # Get keys for valid groups
    # keys are row indices in accum_cpu (0..1023)
    keys = torch.nonzero(mask, as_tuple=False).squeeze(1)
    
    # Decode keys: rf = key // 256, ls = key % 256
    rf_out = (keys // 256).to(torch.int64)
    ls_out = (keys % 256).to(torch.int64)
    
    # Extract aggregates
    sum_qty = valid[:, 0]
    sum_base_price = valid[:, 1]
    sum_disc_price = valid[:, 2]
    sum_charge = valid[:, 3]
    sum_disc = valid[:, 4]
    count_order_f = valid[:, 5] # float64
    count_order = count_order_f.to(torch.int64)
    
    # Compute averages
    avg_qty = sum_qty / count_order_f
    avg_price = sum_base_price / count_order_f
    avg_disc = sum_disc / count_order_f
    
    # Sort by (rf, ls)
    # Create sort key
    sort_key = rf_out * 256 + ls_out
    order = torch.argsort(sort_key)
    
    return (
        rf_out[order],
        ls_out[order],
        sum_qty[order],
        sum_base_price[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        count_order[order],
    )


def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns l_returnflag, l_linestatus to integer codes (single-char strings).
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    # Use categorical codes for encoding
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    (
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode category codes back to strings
    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

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
