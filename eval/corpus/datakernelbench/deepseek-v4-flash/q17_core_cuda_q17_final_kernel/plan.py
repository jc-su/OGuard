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

__global__ void q17_final_kernel(
    const int64_t* __restrict__ sel_idx,
    const double* __restrict__ sel_qty,
    const double* __restrict__ sel_price,
    const double* __restrict__ avg_qty_ptr,
    int64_t n_sel,
    double* __restrict__ out_ptr
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_sel) {
        double qty = sel_qty[idx];
        double avg_val = avg_qty_ptr[sel_idx[idx]];
        double threshold = 0.2 * avg_val;
        if (qty < threshold) {
            my_rev = sel_price[idx];
        }
    }

    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}

double q17_final_launcher(
    torch::Tensor sel_idx,
    torch::Tensor sel_qty,
    torch::Tensor sel_price,
    torch::Tensor avg_qty
) {
    auto contiguify = [](torch::Tensor& t) {
        if (!t.is_cuda()) t = t.cuda();
        t = t.contiguous();
    };
    contiguify(sel_idx);
    contiguify(sel_qty);
    contiguify(sel_price);
    contiguify(avg_qty);

    int64_t n_sel = sel_idx.numel();
    auto out = torch::zeros({1}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    int n_blocks = (int)((n_sel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    q17_final_kernel<<<n_blocks, BLOCK_SIZE>>>(
        sel_idx.data_ptr<int64_t>(),
        sel_qty.data_ptr<double>(),
        sel_price.data_ptr<double>(),
        avg_qty.data_ptr<double>(),
        n_sel,
        out.data_ptr<double>()
    );

    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_final_launcher", &q17_final_launcher, "TPC-H Q17 final select+sum (CUDA)");
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
    Compute TPC-H Q17:
    - Filter lineitems to those matching filtered partkeys (using searchsorted)
    - Compute per-part averages (PyTorch scatter_add)
    - Fused CUDA kernel for final filter (qty < 0.2*avg) and sum of extendedprice
    Returns total / 7.0.
    """
    # Sort part keys for binary search
    sorted_keys, _ = torch.sort(part_keys)
    n_parts = sorted_keys.numel()

    # For each lineitem row, find index in sorted_keys; mark membership
    idx = torch.searchsorted(sorted_keys, l_partkey)
    idx_clamped = torch.clamp(idx, max=n_parts - 1)
    in_part = sorted_keys[idx_clamped] == l_partkey

    # Restrict to lineitems whose partkey is in the filtered parts
    sel_idx = idx_clamped[in_part]
    sel_qty = l_quantity[in_part]
    sel_price = l_extendedprice[in_part]

    # Compute per-part average quantity via scatter reductions (on GPU)
    sum_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_quantity.device)
    cnt_qty = torch.zeros(n_parts, dtype=torch.float64, device=l_quantity.device)
    sum_qty.scatter_add_(0, sel_idx, sel_qty)
    cnt_qty.scatter_add_(0, sel_idx, torch.ones_like(sel_qty))

    avg_qty = torch.where(cnt_qty > 0, sum_qty / cnt_qty, torch.zeros_like(sum_qty))

    # Fused CUDA kernel: apply threshold and sum
    total = _EXT.q17_final_launcher(sel_idx, sel_qty, sel_price, avg_qty)
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
