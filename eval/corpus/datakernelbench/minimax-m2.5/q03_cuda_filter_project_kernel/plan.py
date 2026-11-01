import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Count valid rows matching date predicates
__global__ void count_filter_kernel(
    const int32_t* __restrict__ l_shipdate_days,
    const int32_t* __restrict__ o_orderdate_days,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    unsigned long long* __restrict__ out_count,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t lsd = l_shipdate_days[idx];
        int32_t ood = o_orderdate_days[idx];
        if (lsd > shipdate_lo_days && ood < orderdate_hi_days) {
            atomicAdd(out_count, 1ULL);
        }
    }
}

// Filter + project: extract matching rows with revenue computation
// Uses a simpler two-pass approach: first count, then write
__global__ void filter_project_kernel(
    const int64_t* __restrict__ l_orderkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_shippriority,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    int64_t* __restrict__ out_orderkey,
    int32_t* __restrict__ out_orderdate,
    int32_t* __restrict__ out_shippriority,
    double* __restrict__ out_revenue,
    unsigned long long* __restrict__ out_write_idx,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        int32_t lsd = l_shipdate_days[idx];
        int32_t ood = o_orderdate_days[idx];
        
        if (lsd > shipdate_lo_days && ood < orderdate_hi_days) {
            // Compute revenue
            double rev = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            
            // Get write position atomically
            unsigned long long write_pos = atomicAdd(out_write_idx, 1ULL);
            
            out_orderkey[write_pos] = l_orderkey[idx];
            out_orderdate[write_pos] = ood;
            out_shippriority[write_pos] = o_shippriority[idx];
            out_revenue[write_pos] = rev;
        }
    }
}

// Launcher - two-pass filter
std::vector<torch::Tensor> q3_filter_and_prepare(
    torch::Tensor l_orderkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor o_orderdate_days,
    torch::Tensor o_shippriority,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days
) {
    TORCH_CHECK(l_orderkey.is_cuda());
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(o_orderdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(o_shippriority.scalar_type() == torch::kInt32);

    l_orderkey = l_orderkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    o_orderdate_days = o_orderdate_days.contiguous();
    o_shippriority = o_shippriority.contiguous();

    int64_t n = l_orderkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: count valid rows
    auto count_tensor = torch::zeros({1}, torch::kInt64).cuda();
    count_filter_kernel<<<blocks, BLOCK_SIZE>>>(
        l_shipdate_days.data_ptr<int32_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        shipdate_lo_days,
        orderdate_hi_days,
        (unsigned long long*)count_tensor.data_ptr<int64_t>(),
        n
    );
    cudaDeviceSynchronize();
    int64_t filtered_count = count_tensor.item<int64_t>();

    if (filtered_count == 0) {
        return {
            torch::empty({0}, torch::kInt64).cuda(),
            torch::empty({0}, torch::kInt32).cuda(),
            torch::empty({0}, torch::kInt32).cuda(),
            torch::empty({0}, torch::kFloat64).cuda(),
            torch::tensor(0, torch::kInt64).cuda()
        };
    }

    // Phase 2: allocate output and filter
    auto out_orderkey = torch::empty({filtered_count}, torch::kInt64).cuda();
    auto out_orderdate = torch::empty({filtered_count}, torch::kInt32).cuda();
    auto out_shippriority = torch::empty({filtered_count}, torch::kInt32).cuda();
    auto out_revenue = torch::empty({filtered_count}, torch::kFloat64).cuda();
    auto write_idx = torch::zeros({1}, torch::kInt64).cuda();

    filter_project_kernel<<<blocks, BLOCK_SIZE>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        o_orderdate_days.data_ptr<int32_t>(),
        o_shippriority.data_ptr<int32_t>(),
        shipdate_lo_days,
        orderdate_hi_days,
        out_orderkey.data_ptr<int64_t>(),
        out_orderdate.data_ptr<int32_t>(),
        out_shippriority.data_ptr<int32_t>(),
        out_revenue.data_ptr<double>(),
        (unsigned long long*)write_idx.data_ptr<int64_t>(),
        n
    );
    cudaDeviceSynchronize();

    return {out_orderkey, out_orderdate, out_shippriority, out_revenue, write_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q3_filter_and_prepare", &q3_filter_and_prepare, 
          "TPC-H Q3: filter + project (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q3_cuda_")
    cu_path = os.path.join(tmpdir, "_q3_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q3_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()


def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized CUDA path for TPC-H Q3:
    1. CUDA kernel: filter rows by dates, compute revenue
    2. Build composite key (l_orderkey, o_orderdate, o_shippriority) - exact ref method
    3. PyTorch: unique + scatter_add for aggregation - exact ref method
    4. PyTorch: top-N sort - exact ref method
    
    Matches reference semantics exactly.
    """
    # Move to GPU
    l_orderkey = l_orderkey.cuda().contiguous()
    l_extendedprice = l_extendedprice.cuda().contiguous()
    l_discount = l_discount.cuda().contiguous()
    l_shipdate_days = l_shipdate_days.cuda().contiguous()
    o_orderdate_days = o_orderdate_days.cuda().contiguous()
    o_shippriority = o_shippriority.cuda().contiguous()

    # Phase 1: CUDA filter + compute revenue (same semantics as reference)
    out_lk, out_od, out_sp, out_rev, count_tensor = _EXT.q3_filter_and_prepare(
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days
    )

    n_filtered = count_tensor.item()
    if n_filtered == 0:
        return (
            torch.empty(0, dtype=torch.int64, device='cuda'),
            torch.empty(0, dtype=torch.float64, device='cuda'),
            torch.empty(0, dtype=torch.int32, device='cuda'),
            torch.empty(0, dtype=torch.int32, device='cuda')
        )

    # Phase 2: Build composite key - EXACTLY like reference
    lk = out_lk.to(torch.int64)
    od = out_od.to(torch.int64)
    sp = out_sp.to(torch.int64)
    rev = out_rev

    keys = torch.stack([lk, od, sp], dim=1)

    # Phase 3: Aggregate - EXACTLY like reference
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev.dtype, device=rev.device)
    revenue_sum.scatter_add_(0, inverse, rev)

    # Phase 4: Unpack unique keys
    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)

    # Phase 5: ORDER BY revenue DESC, o_orderdate ASC LIMIT top_n
    # EXACTLY like reference
    n = unique_keys.shape[0]
    k = min(top_n, n)

    # Sort by secondary first (o_orderdate ASC) - stable
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]

    # Sort by primary (revenue DESC) - stable
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]

    return lk_sorted, rev_sorted, od_sorted, sp_sorted


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 Shipping Priority Query.
                 - cuDF handles joins (customer->orders->lineitem)
                 - CUDA kernel handles date filter + revenue projection
                 - PyTorch handles aggregation (unique+scatter_add) + top-N
                 Why faster: CUDA filter pass reduces data going to PyTorch;
                             aggregation and sort leverage PyTorch's optimized primitives
    Params:
        mktsegment (str): customer market segment filter
        orderdate_hi (str): exclusive upper bound on o_orderdate
        shipdate_lo (str): exclusive lower bound on l_shipdate  
        top_n (int): LIMIT value
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert string dates to epoch-day integers (same as reference)
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # cuDF joins: customer -> orders -> lineitem (same as reference)
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day counts (same as reference)
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    # Call optimized CUDA path
    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate from int32 days back to datetime64[s] (same as reference)
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
