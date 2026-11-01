import os
import tempfile
from typing import Any

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

#define BLOCK_SIZE 1024
#define MAX_BLOCKS_PER_SM 8

__global__ void q17_build_map_kernel(
    const int64_t* __restrict__ part_keys,
    int32_t* __restrict__ part_map,
    int64_t n_parts,
    int64_t map_size
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_parts;
         i += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = part_keys[i];
        if (key >= 0 && key < map_size) {
            part_map[key] = (int32_t)i;
        }
    }
}

__global__ void q17_first_pass_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ part_map,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_lineitem,
    int64_t map_size
) {
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_lineitem;
         row += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = l_partkey[row];
        if (key >= 0 && key < map_size) {
            int32_t pidx = part_map[key];
            if (pidx >= 0) {
                atomicAdd(&sum_qty[pidx], l_quantity[row]);
                atomicAdd(&cnt_qty[pidx], 1.0);
            }
        }
    }
}

__global__ void q17_second_pass_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int32_t* __restrict__ part_map,
    const double* __restrict__ sum_qty,
    const double* __restrict__ cnt_qty,
    double* __restrict__ out,
    int64_t n_lineitem,
    int64_t map_size
) {
    __shared__ double block_sum[BLOCK_SIZE];

    double local = 0.0;
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_lineitem;
         row += (int64_t)blockDim.x * gridDim.x) {
        int64_t key = l_partkey[row];
        if (key >= 0 && key < map_size) {
            int32_t pidx = part_map[key];
            if (pidx >= 0) {
                double cnt = cnt_qty[pidx];
                if (cnt > 0.0) {
                    double qty = l_quantity[row];
                    double threshold = 0.2 * (sum_qty[pidx] / cnt);
                    if (qty < threshold) {
                        local += l_extendedprice[row];
                    }
                }
            }
        }
    }

    block_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}

static int q17_blocks_for(int64_t work) {
    if (work <= 0) {
        return 1;
    }
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int64_t natural = (work + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int64_t cap = (int64_t)prop.multiProcessorCount * MAX_BLOCKS_PER_SM;
    int64_t blocks = std::min<int64_t>(natural, std::max<int64_t>(1, cap));
    return (int)blocks;
}

double q17_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_keys,
    int64_t map_size
) {
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(part_keys.scalar_type() == torch::kInt64, "part_keys must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_partkey.dim() == 1, "l_partkey must be 1-D");
    TORCH_CHECK(l_quantity.dim() == 1, "l_quantity must be 1-D");
    TORCH_CHECK(l_extendedprice.dim() == 1, "l_extendedprice must be 1-D");
    TORCH_CHECK(part_keys.dim() == 1, "part_keys must be 1-D");
    TORCH_CHECK(l_partkey.numel() == l_quantity.numel(), "l_partkey/l_quantity size mismatch");
    TORCH_CHECK(l_partkey.numel() == l_extendedprice.numel(), "l_partkey/l_extendedprice size mismatch");

    if (!l_partkey.is_cuda()) {
        l_partkey = l_partkey.to(torch::kCUDA);
    }
    c10::Device dev = l_partkey.device();

    if (!l_quantity.is_cuda() || l_quantity.device() != dev) {
        l_quantity = l_quantity.to(dev);
    }
    if (!l_extendedprice.is_cuda() || l_extendedprice.device() != dev) {
        l_extendedprice = l_extendedprice.to(dev);
    }
    if (!part_keys.is_cuda() || part_keys.device() != dev) {
        part_keys = part_keys.to(dev);
    }

    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    part_keys = part_keys.contiguous();

    int64_t n_lineitem = l_partkey.numel();
    int64_t n_parts = part_keys.numel();

    if (n_lineitem == 0 || n_parts == 0 || map_size <= 0) {
        return 0.0;
    }
    TORCH_CHECK(n_parts <= INT32_MAX, "too many filtered parts for int32 map indices");

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto dbl_opts = torch::TensorOptions().dtype(torch::kFloat64).device(dev);

    torch::Tensor part_map = torch::empty({map_size}, int_opts);
    torch::Tensor sum_qty = torch::zeros({n_parts}, dbl_opts);
    torch::Tensor cnt_qty = torch::zeros({n_parts}, dbl_opts);
    torch::Tensor out = torch::zeros({1}, dbl_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaMemsetAsync(
        part_map.data_ptr<int32_t>(),
        0xff,
        (size_t)map_size * sizeof(int32_t),
        stream
    ));

    int part_blocks = q17_blocks_for(n_parts);
    q17_build_map_kernel<<<part_blocks, BLOCK_SIZE, 0, stream>>>(
        part_keys.data_ptr<int64_t>(),
        part_map.data_ptr<int32_t>(),
        n_parts,
        map_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int line_blocks = q17_blocks_for(n_lineitem);
    q17_first_pass_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        part_map.data_ptr<int32_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_lineitem,
        map_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    q17_second_pass_kernel<<<line_blocks, BLOCK_SIZE, 0, stream>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        part_map.data_ptr<int32_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        out.data_ptr<double>(),
        n_lineitem,
        map_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    double total = out.item<double>();
    return total / 7.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 fused CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """
    Compile the hand-written CUDA extension from the in-module source string.

    The CUDA code runs the hot path on GPU: a dense part-key map build, one
    lineitem pass for per-part quantity sums/counts, and one fused pass for the
    final revenue reduction. Keeping these scans in CUDA avoids PyTorch eager
    searchsorted/scatter temporaries and replaces them with coalesced scans plus
    low-contention atomics.
    """
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _launch_q17(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
    map_size: int,
) -> torch.Tensor:
    """
    Launch the CUDA Q17 implementation and wrap its scalar result as a CUDA tensor.

    The extension performs the memory-bound lineitem scans in CUDA using a dense
    int32 lookup table for filtered part keys, reducing membership tests from
    per-row binary searches to one cached map load.
    """
    result = _EXT.q17_launcher(
        l_partkey,
        l_quantity,
        l_extendedprice,
        part_keys,
        int(map_size),
    )
    device = l_quantity.device if l_quantity.is_cuda else torch.device("cuda")
    return torch.tensor(result, dtype=torch.float64, device=device)


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17 core with CUDA kernels.

    GPU/CUDA work:
      1. Build a dense partkey -> filtered-part-index map.
      2. Scan lineitem once to compute sum/count of l_quantity per filtered part.
      3. Scan lineitem again to sum l_extendedprice for rows below 20% of the
         per-part average quantity, with block-level reduction before atomics.

    This should be faster than the reference eager PyTorch plan because it avoids
    sort/searchsorted masks and scatter temporaries over the full lineitem table.
    """
    if int(part_keys.numel()) == 0 or int(l_partkey.numel()) == 0:
        device = l_quantity.device if l_quantity.is_cuda else torch.device("cuda")
        return torch.tensor(0.0, dtype=torch.float64, device=device)
    map_size = int(torch.max(part_keys).item()) + 1
    return _launch_q17(l_partkey, l_quantity, l_extendedprice, part_keys, map_size)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT
            sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM
            lineitem,
            part
        WHERE
            p_partkey = l_partkey
            AND p_brand = 'Brand#23'
            AND p_container = 'MED BOX'
            AND l_quantity < (
                SELECT
                    0.2 * avg(l_quantity)
                FROM
                    lineitem
                WHERE
                    l_partkey = p_partkey);

    Description:
        TPC-H Q17. cuDF performs the string filter on the small part dimension
        table to obtain qualifying p_partkey values. The bulk lineitem work runs
        in hand-written CUDA: a dense key map is built for the filtered parts,
        then two coalesced lineitem scans compute per-part quantity averages and
        the final extended-price sum. This reduces the memory-bound hot path by
        avoiding PyTorch eager sort/searchsorted, boolean compaction, and scatter
        temporaries while loading l_extendedprice only for qualifying rows.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables containing
            "lineitem" and "part".
        brand_val (str): Brand filter literal for p_brand.
        container_val (str): Container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    filtered_part_keys = df_part.loc[part_mask, "p_partkey"]

    if len(filtered_part_keys) == 0 or len(df_lineitem) == 0:
        return 0.0

    max_part_key = int(filtered_part_keys.max())
    map_size = max_part_key + 1
    if map_size <= 0:
        return 0.0

    part_keys_t = torch.as_tensor(filtered_part_keys.to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(),
        dtype=torch.float64,
    )

    res = _launch_q17(
        l_partkey_t,
        l_quantity_t,
        l_extendedprice_t,
        part_keys_t,
        map_size,
    )
    return float(res.item())
