import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 512

__global__ void mark_valid_keys_kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commit,
    const int32_t* __restrict__ l_receipt,
    uint32_t* __restrict__ bitmap,
    int64_t n_lineitem
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_lineitem) {
        if (l_commit[idx] < l_receipt[idx]) {
            int64_t key = l_orderkey[idx];
            if (key > 0) atomicOr(&bitmap[key], 1u);
        }
    }
}

__global__ void count_priorities_kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_date,
    const int32_t* __restrict__ o_priority,
    const uint32_t* __restrict__ bitmap,
    unsigned long long* __restrict__ counts,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_orderkey[idx];
        if (key > 0 && bitmap[key]) {
            int32_t d = o_date[idx];
            if (d >= date_lo && d < date_hi) {
                int32_t p = o_priority[idx];
                atomicAdd(&counts[p], (unsigned long long)1);
            }
        }
    }
}

std::vector<torch::Tensor> q4_launcher(
    torch::Tensor l_orderkey,
    torch::Tensor l_commit,
    torch::Tensor l_receipt,
    torch::Tensor o_orderkey,
    torch::Tensor o_date,
    torch::Tensor o_priority,
    int32_t date_lo,
    int32_t date_hi,
    int64_t max_key,
    int32_t num_priorities
) {
    auto device = torch::kCUDA;
    l_orderkey = l_orderkey.to(device).contiguous();
    l_commit = l_commit.to(device).contiguous();
    l_receipt = l_receipt.to(device).contiguous();
    o_orderkey = o_orderkey.to(device).contiguous();
    o_date = o_date.to(device).contiguous();
    o_priority = o_priority.to(device).contiguous();

    int64_t n_li = l_orderkey.numel();
    int64_t n_ord = o_orderkey.numel();

    auto bitmap_opts = torch::TensorOptions().dtype(torch::kUInt32).device(device);
    torch::Tensor bitmap = torch::zeros({max_key + 1}, bitmap_opts);

    int blocks_li = (n_li + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mark_valid_keys_kernel<<<blocks_li, BLOCK_SIZE>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_commit.data_ptr<int32_t>(),
        l_receipt.data_ptr<int32_t>(),
        bitmap.data_ptr<uint32_t>(),
        n_li
    );

    auto counts_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor counts = torch::zeros({num_priorities}, counts_opts);

    int blocks_ord = (n_ord + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_priorities_kernel<<<blocks_ord, BLOCK_SIZE>>>(
        o_orderkey.data_ptr<int64_t>(),
        o_date.data_ptr<int32_t>(),
        o_priority.data_ptr<int32_t>(),
        bitmap.data_ptr<uint32_t>(),
        (unsigned long long*)counts.data_ptr<int64_t>(),
        date_lo,
        date_hi,
        n_ord
    );

    cudaDeviceSynchronize();
    return {counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_launcher", &q4_launcher, "TPC-H Q4 CUDA");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q4_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q4_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late. Uses fused CUDA kernels with a bitmap for EXISTS check to minimize memory passes and avoid sorting.
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    max_o = int(df_orders["o_orderkey"].max())
    max_l = int(df_lineitem["l_orderkey"].max())
    max_key = max(max_o, max_l)

    counts_t = _EXT.q4_launcher(
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        date_lo_days,
        date_hi_days,
        max_key,
        num_priorities,
    )[0]

    counts_host = counts_t.cpu()
    nz_mask = counts_host > 0
    priority_ids = torch.nonzero(nz_mask, as_tuple=False).squeeze(1).tolist()
    priority_strs = [priority_categories[i] for i in priority_ids]
    counts_vals = counts_host[nz_mask].tolist()

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_vals,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
