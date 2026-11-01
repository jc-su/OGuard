import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 1024

__global__ void q14_kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const bool* __restrict__ promo,
    double* __restrict__ out_promo_rev,
    double* __restrict__ out_total_rev,
    int64_t n
) {
    __shared__ double block_promo[BLOCK_SIZE];
    __shared__ double block_total[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    double p_rev = 0.0;
    double t_rev = 0.0;

    // Grid stride loop ensures kernel works securely with arbitrary n elements 
    for (int64_t i = idx; i < n; i += stride) {
        double p = price[i];
        double d = discount[i];
        bool pr = promo[i];
        double rev = p * (1.0 - d);
        t_rev += rev;
        if (pr) {
            p_rev += rev;
        }
    }

    block_promo[threadIdx.x] = p_rev;
    block_total[threadIdx.x] = t_rev;
    __syncthreads();

    // Shared memory tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Single global atomic addition per block
    if (threadIdx.x == 0) {
        atomicAdd(out_promo_rev, block_promo[0]);
        atomicAdd(out_total_rev, block_total[0]);
    }
}

std::vector<torch::Tensor> q14_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor promo
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(promo.scalar_type() == torch::kBool);
    
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!promo.is_cuda()) promo = promo.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    promo = promo.contiguous();
    
    int64_t n = price.numel();
    auto options = price.options();
    auto out_promo_rev = torch::zeros({1}, options);
    auto out_total_rev = torch::zeros({1}, options);

    if (n > 0) {
        // Cap grid size; grid-stride loop handles excess elements
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65536);
        q14_kernel<<<blocks, BLOCK_SIZE>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            promo.data_ptr<bool>(),
            out_promo_rev.data_ptr<double>(),
            out_total_rev.data_ptr<double>(),
            n
        );
    }
    
    return {out_promo_rev, out_total_rev};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q14_launcher", &q14_launcher, "TPC-H Q14 fused calculation");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q14_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q14_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
                 This highly optimized path performs early pushdown filtering on `l_shipdate` in cuDF to drastically
                 shrink the join size. The projection and aggregation mathematically fuse inside a single CUDA kernel
                 bypassing enormous intermediate materialization of `(1 - discount) * price`.
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Pushdown filter on lineitem to drastically reduce dataframe sizes prior to join
    l_shipdate_days = df_lineitem["l_shipdate"].astype("int64") // 86400
    mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)
    f_lineitem = df_lineitem[mask]

    if len(f_lineitem) == 0:
        return 0.0

    # Precompute promo flag on part side before join for efficient transfer
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Join
    joined = f_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey", how="inner")

    if len(joined) == 0:
        return 0.0

    # Extract device arrays and pass to PyTorch explicitly keeping required types
    price = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    promo = torch.as_tensor(joined["p_promo"].to_cupy(), dtype=torch.bool)

    # Launch fused calculation kernel
    out_promo, out_total = _EXT.q14_launcher(price, discount, promo)

    total_val = float(out_total.item())
    if total_val == 0.0:
        return 0.0
    
    return 100.0 * float(out_promo.item()) / total_val
