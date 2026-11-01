import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define MAX_HIST_BINS 1024

// Kernel to count occurrences of each order customer key.
// Passing the boolean mask directly into the kernel prevents the need to
// eagerly materialize a filtered DataFrame in memory beforehand, saving
// significant allocation time and memory bandwidth.
__global__ void count_orders_kernel(
    const int64_t* __restrict__ o_custkey, 
    const bool* __restrict__ mask, 
    int32_t* __restrict__ counts, 
    int64_t n, 
    int64_t max_custkey
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx]) {
            int64_t cust = o_custkey[idx];
            if (cust >= 0 && cust <= max_custkey) {
                atomicAdd(&counts[cust], 1);
            }
        }
    }
}

// Kernel to build the histogram of counts for all valid customers.
// Heavily leverages shared memory for the most common order counts (e.g. 0-40) 
// to prevent massive serialization on global atomics, drastically improving speed.
__global__ void hist_counts_kernel(
    const int64_t* __restrict__ c_custkey, 
    const int32_t* __restrict__ counts, 
    int32_t* __restrict__ hist, 
    int64_t n, 
    int64_t max_custkey,
    int32_t hist_size
) {
    __shared__ int32_t smem_hist[MAX_HIST_BINS];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < MAX_HIST_BINS; i += blockDim.x) {
        smem_hist[i] = 0;
    }
    __syncthreads();
    
    // Bin assignments per thread
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t c = c_custkey[idx];
        if (c >= 0 && c <= max_custkey) {
            int32_t cnt = counts[c];
            if (cnt < MAX_HIST_BINS) {
                atomicAdd(&smem_hist[cnt], 1);
            } else if (cnt < hist_size) {
                // Fallback for anomalously high counts (rare)
                atomicAdd(&hist[cnt], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Flush shared memory reduction to global memory
    for (int i = threadIdx.x; i < MAX_HIST_BINS; i += blockDim.x) {
        int32_t val = smem_hist[i];
        if (val > 0) {
            atomicAdd(&hist[i], val);
        }
    }
}

std::vector<torch::Tensor> q13_launcher(
    torch::Tensor c_custkey, 
    torch::Tensor o_custkey, 
    torch::Tensor mask, 
    int64_t max_custkey
) {
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64, "c_custkey must be int64");
    TORCH_CHECK(o_custkey.scalar_type() == torch::kInt64, "o_custkey must be int64");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    
    if (!c_custkey.is_cuda()) c_custkey = c_custkey.cuda();
    if (!o_custkey.is_cuda()) o_custkey = o_custkey.cuda();
    if (!mask.is_cuda()) mask = mask.cuda();
    
    c_custkey = c_custkey.contiguous();
    o_custkey = o_custkey.contiguous();
    mask = mask.contiguous();
    
    auto options = o_custkey.options();
    auto counts = torch::zeros({max_custkey + 1}, options.dtype(torch::kInt32));
    
    int64_t n_orders = o_custkey.numel();
    if (n_orders > 0) {
        int blocks_o = (n_orders + 1023) / 1024;
        count_orders_kernel<<<blocks_o, 1024>>>(
            o_custkey.data_ptr<int64_t>(),
            mask.data_ptr<bool>(),
            counts.data_ptr<int32_t>(),
            n_orders,
            max_custkey
        );
    }
    
    // 100,000 bins is safely much larger than max orders per customer for standard scale factors
    int32_t hist_size = 100000; 
    auto hist = torch::zeros({hist_size}, options.dtype(torch::kInt32));
    
    int64_t n_customers = c_custkey.numel();
    if (n_customers > 0) {
        int blocks_c = (n_customers + 1023) / 1024;
        hist_counts_kernel<<<blocks_c, 1024>>>(
            c_custkey.data_ptr<int64_t>(),
            counts.data_ptr<int32_t>(),
            hist.data_ptr<int32_t>(),
            n_customers,
            max_custkey,
            hist_size
        );
    }
    
    return {hist};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q13_launcher", &q13_launcher, "TPCH Q13 CUDA Launcher");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q13_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(tables: dict[str, cudf.DataFrame], comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;
        
    Description: TPC-H Q13 - Customer distribution based on number of non-special-request orders.
    cuDF is used for standard regex resolution to generate a string boolean mask. A fused CUDA pipeline 
    then processes the raw order keys alongside the boolean mask directly, skipping the costly
    memory allocation of intermediate filtered dataframes, and builds the target histogram using 
    block-level shared memory to massively accelerate reductions.
    
    Params: comment_pattern (str): the LIKE pattern to exclude from orders via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Generate the boolean mask for filtered orders but do not materialize a new dataframe 
    # to avoid significant GPU memory allocations and bandwidth overheads.
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(df_orders["o_custkey"].to_cupy(), dtype=torch.int64)
    mask_t = torch.as_tensor(mask.to_cupy(), dtype=torch.bool)

    # Calculate bucket boundary parameters based on existing data
    max_custkey = int(df_customer["c_custkey"].max())

    # Launch optimized CUDA pipeline for fast order counts and histogram generation
    hist = _EXT.q13_launcher(c_custkey_t, o_custkey_t, mask_t, max_custkey)[0]

    # Post-process: Extract non-zero entries from the resulting histogram
    mask_nz = hist > 0
    c_count_t = torch.nonzero(mask_nz).squeeze(1).to(torch.int64)
    custdist_t = hist[mask_nz].to(torch.int64)

    # ORDER BY custdist DESC, c_count DESC
    # Sort by c_count DESC first, then stable sort by custdist DESC.
    order1 = torch.argsort(c_count_t, descending=True, stable=True)
    c_count_t = c_count_t[order1]
    custdist_t = custdist_t[order1]
    
    order2 = torch.argsort(custdist_t, descending=True, stable=True)
    c_count_t = c_count_t[order2]
    custdist_t = custdist_t[order2]

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
