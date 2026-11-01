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
#define MAX_PRIORITIES 10

__global__ void q4_exists_bincount_kernel(
    const int64_t* __restrict__ filt_orderkey,
    const int32_t* __restrict__ filt_priority,
    const int64_t* __restrict__ unique_l_orderkey,
    int64_t n_filt_orders,
    int64_t n_unique_lineitems,
    int64_t num_priorities,
    unsigned long long int* __restrict__ priority_counts
) {
    __shared__ unsigned long long int block_counts[MAX_PRIORITIES];
    
    for (int i = threadIdx.x; i < num_priorities; i += blockDim.x) {
        block_counts[i] = 0;
    }
    __syncthreads();
    
    int64_t order_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (order_idx < n_filt_orders) {
        int64_t order_key = filt_orderkey[order_idx];
        int32_t priority = filt_priority[order_idx];
        
        bool exists = false;
        int64_t lo = 0, hi = n_unique_lineitems;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (unique_l_orderkey[mid] < order_key) {
                lo = mid + 1;
            } else if (unique_l_orderkey[mid] > order_key) {
                hi = mid;
            } else {
                exists = true;
                break;
            }
        }
        
        if (exists && priority >= 0 && priority < num_priorities) {
            atomicAdd(&block_counts[priority], 1ULL);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_priorities; i++) {
            if (block_counts[i] > 0) {
                atomicAdd(&priority_counts[i], block_counts[i]);
            }
        }
    }
}

std::vector<torch::Tensor> q4_launcher(
    torch::Tensor filt_orderkey,
    torch::Tensor filt_priority,
    torch::Tensor unique_l_orderkey,
    int64_t num_priorities
) {
    TORCH_CHECK(filt_orderkey.scalar_type() == torch::kInt64, "filt_orderkey must be int64");
    TORCH_CHECK(filt_priority.scalar_type() == torch::kInt32, "filt_priority must be int32");
    TORCH_CHECK(unique_l_orderkey.scalar_type() == torch::kInt64, "unique_l_orderkey must be int64");
    
    if (!filt_orderkey.is_cuda()) filt_orderkey = filt_orderkey.cuda();
    if (!filt_priority.is_cuda()) filt_priority = filt_priority.cuda();
    if (!unique_l_orderkey.is_cuda()) unique_l_orderkey = unique_l_orderkey.cuda();
    
    filt_orderkey = filt_orderkey.contiguous();
    filt_priority = filt_priority.contiguous();
    unique_l_orderkey = unique_l_orderkey.contiguous();
    
    auto sorted_result = unique_l_orderkey.sort();
    unique_l_orderkey = std::get<0>(sorted_result);
    
    int64_t n_filt_orders = filt_orderkey.numel();
    int64_t n_unique_lineitems = unique_l_orderkey.numel();
    
    auto priority_counts = torch::zeros({num_priorities}, filt_orderkey.options().dtype(torch::kInt64));
    
    if (n_filt_orders > 0 && n_unique_lineitems > 0) {
        int blocks = (n_filt_orders + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q4_exists_bincount_kernel<<<blocks, BLOCK_SIZE>>>(
            filt_orderkey.data_ptr<int64_t>(),
            filt_priority.data_ptr<int32_t>(),
            unique_l_orderkey.data_ptr<int64_t>(),
            n_filt_orders,
            n_unique_lineitems,
            num_priorities,
            (unsigned long long int*)priority_counts.data_ptr<int64_t>()
        );
        cudaDeviceSynchronize();
    }
    
    auto mask = priority_counts > 0;
    auto nonzero_indices = mask.nonzero().squeeze(-1);
    
    torch::Tensor priority_ids;
    torch::Tensor counts;
    
    if (nonzero_indices.numel() > 0) {
        priority_ids = torch::arange(num_priorities, priority_counts.options()).index_select(0, nonzero_indices);
        counts = priority_counts.index_select(0, nonzero_indices);
    } else {
        priority_ids = torch::empty({0}, priority_counts.options());
        counts = torch::empty({0}, priority_counts.options());
    }
    
    return {priority_ids, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_launcher", &q4_launcher, "TPC-H Q4 exists check + bincount (CUDA)");
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

def _launch_q4(
    filt_orderkey: torch.Tensor,
    filt_priority: torch.Tensor,
    unique_l_orderkey: torch.Tensor,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch CUDA kernel for exists check + bincount fusion.
    
    GPU: Binary search exists check + block-level reduction + atomic increment.
    Why faster: Fused kernel reduces memory passes, shared memory block reduction
                minimizes global atomics, binary search is O(log n) per order.
    """
    result = _EXT.q4_launcher(
        filt_orderkey, filt_priority, unique_l_orderkey, num_priorities
    )
    return result[0], result[1]

def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated TPC-H Q4 core computation.
    
    GPU: Filter orders by date (PyTorch), filter lineitems by commit<receipt (PyTorch),
         CUDA kernel for exists check (binary search) + bincount (fused).
    Why faster: Fused exists+bincount kernel with block-level shared memory reduction,
                fewer kernel launches and memory passes than reference torch.isin + bincount.
    """
    order_mask = (o_orderdate_days >= date_lo_days) & (o_orderdate_days < date_hi_days)
    filt_orderkey = o_orderkey[order_mask]
    filt_priority = o_orderpriority_codes[order_mask]
    
    line_mask = l_commitdate_days < l_receiptdate_days
    qualifying_l_orderkey = l_orderkey[line_mask]
    
    unique_keys = torch.unique(qualifying_l_orderkey)
    
    if filt_orderkey.numel() == 0 or unique_keys.numel() == 0:
        priority_ids = torch.arange(num_priorities, device=filt_orderkey.device, dtype=torch.int64)
        counts = torch.zeros(num_priorities, device=filt_orderkey.device, dtype=torch.int64)
        nz = counts > 0
        return priority_ids[nz], counts[nz]
    
    priority_ids_t, counts_t = _launch_q4(
        filt_orderkey, filt_priority, unique_keys, num_priorities
    )
    
    return priority_ids_t, counts_t


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt).
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

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
