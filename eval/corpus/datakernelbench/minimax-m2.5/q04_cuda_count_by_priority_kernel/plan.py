import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

// Count orders by priority using binary search for EXISTS check
__global__ void count_by_priority_kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_priority,
    const int32_t* __restrict__ o_orderdate,
    int64_t n_orders,
    int64_t date_lo,
    int64_t date_hi,
    const int64_t* __restrict__ late_keys,
    int64_t n_late_keys,
    int64_t* __restrict__ counts,
    int32_t num_priorities
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements
    for (int64_t i = idx; i < n_orders; i += gridDim.x * blockDim.x) {
        int32_t od = o_orderdate[i];
        
        // Filter by date range
        if (od >= (int32_t)date_lo && od < (int32_t)date_hi) {
            int64_t ok = o_orderkey[i];
            
            // Binary search for EXISTS check
            int64_t lo = 0, hi = n_late_keys;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if (late_keys[mid] < ok) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            bool found = (lo < n_late_keys && late_keys[lo] == ok);
            
            if (found) {
                int32_t pc = o_priority[i];
                if (pc >= 0 && pc < num_priorities) {
                    atomicAdd((unsigned long long*)&counts[pc], (unsigned long long)1);
                }
            }
        }
    }
}

torch::Tensor run_q4(
    torch::Tensor o_orderkey,
    torch::Tensor o_orderdate,
    torch::Tensor o_priority,
    torch::Tensor late_keys_sorted,
    int64_t date_lo,
    int64_t date_hi,
    int32_t num_priorities
) {
    TORCH_CHECK(o_orderkey.is_cuda(), "o_orderkey must be CUDA");
    TORCH_CHECK(o_orderkey.scalar_type() == torch::kInt64, "o_orderkey int64");
    TORCH_CHECK(o_orderdate.scalar_type() == torch::kInt32, "o_orderdate int32");
    TORCH_CHECK(o_priority.scalar_type() == torch::kInt32, "o_priority int32");
    TORCH_CHECK(late_keys_sorted.is_cuda(), "late_keys_sorted must be CUDA");
    TORCH_CHECK(late_keys_sorted.scalar_type() == torch::kInt64, "late_keys_sorted int64");

    o_orderkey = o_orderkey.contiguous();
    o_orderdate = o_orderdate.contiguous();
    o_priority = o_priority.contiguous();
    late_keys_sorted = late_keys_sorted.contiguous();

    int64_t n_orders = o_orderkey.size(0);
    int64_t n_late = late_keys_sorted.size(0);
    
    // Allocate counts array
    auto counts = torch::zeros({num_priorities}, o_orderkey.options().dtype(torch::kInt64)).to(o_orderkey.device());
    
    // Launch kernel for filtering and aggregation
    int blocks = 256;
    int threads = BLOCK_SIZE;
    
    count_by_priority_kernel<<<blocks, threads>>>(
        o_orderkey.data_ptr<int64_t>(),
        o_priority.data_ptr<int32_t>(),
        o_orderdate.data_ptr<int32_t>(),
        n_orders,
        date_lo,
        date_hi,
        late_keys_sorted.data_ptr<int64_t>(),
        n_late,
        counts.data_ptr<int64_t>(),
        num_priorities
    );
    
    cudaDeviceSynchronize();
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_q4", &run_q4, "TPC-H Q4: count orders per priority with late lineitems (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q4_cuda_")
    cu_path = os.path.join(tmpdir, "_q4_kernels.cu")
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
    SQL:
        SELECT o_orderpriority, count(*) AS order_count
        FROM orders
        WHERE o_orderdate >= CAST('1993-07-01' AS date)
          AND o_orderdate < CAST('1993-10-01' AS date)
          AND EXISTS (SELECT * FROM lineitem
                      WHERE l_orderkey = o_orderkey
                        AND l_commitdate < l_receiptdate)
        GROUP BY o_orderpriority
        ORDER BY o_orderpriority;
    Description:
        TPC-H Q4: Counts orders per priority within a date range where at least one
        lineitem shipped late (commit date earlier than receipt date). Uses torch
        on Python side for efficient filter+sort+unique to get unique order keys
        with late shipments, then a fused CUDA kernel for the final filter+aggregate
        with binary search for the EXISTS check. Hot path stays on GPU.
    Params:
        tables: Dictionary mapping table names to cuDF DataFrames.
        date_lo: Inclusive lower bound for o_orderdate (default '1993-07-01').
        date_hi: Exclusive upper bound for o_orderdate (default '1993-10-01').
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    # Prepare order tensors on GPU
    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64).cuda()
    o_orderdate_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32
    ).cuda()
    o_priority_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32).cuda()

    o_orderkey_t = o_orderkey_t.contiguous()
    o_orderdate_t = o_orderdate_t.contiguous()
    o_priority_t = o_priority_t.contiguous()

    # Prepare lineitem tensors on GPU
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64).cuda()
    l_commitdate_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32
    ).cuda()
    l_receiptdate_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32
    ).cuda()

    l_orderkey_t = l_orderkey_t.contiguous()
    l_commitdate_t = l_commitdate_t.contiguous()
    l_receiptdate_t = l_receiptdate_t.contiguous()

    # Step 1: Filter late shipments (commit < receipt) - using torch operations on GPU
    late_mask = (l_commitdate_t < l_receiptdate_t)
    late_keys = l_orderkey_t[late_mask]  # Compact: only late shipments
    
    # Step 2: Sort and get unique keys using torch (efficient GPU implementation)
    late_keys_sorted, _ = torch.sort(late_keys, dim=0)
    late_keys_unique = torch.unique(late_keys_sorted)
    
    # Step 3: Launch CUDA kernel for filter by date range + aggregate by priority
    counts_tensor = _EXT.run_q4(
        o_orderkey_t,
        o_orderdate_t,
        o_priority_t,
        late_keys_unique,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    counts = counts_tensor.cpu().tolist()
    result_rows = [(priority_categories[i], cnt) for i, cnt in enumerate(counts) if cnt > 0]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series([r[0] for r in result_rows], dtype="object"),
        "order_count": cudf.Series([r[1] for r in result_rows], dtype="int64"),
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
