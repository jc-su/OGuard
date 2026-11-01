import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q5_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ out_ptr,
    int64_t num_nations,
    int64_t n_elements
) {
    // Per-block shared memory for reduction
    __shared__ double block_revenue[BLOCK_SIZE];
    // Also store per-thread nation_id for the active thread
    __shared__ int64_t block_nation[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_nid = 0;

    if (idx < n_elements) {
        my_nid = nation_id_ptr[idx];
        double price = price_ptr[idx];
        double disc = discount_ptr[idx];
        my_rev = price * (1.0 - disc);
    }

    // Store into shared memory
    block_revenue[threadIdx.x] = my_rev;
    block_nation[threadIdx.x] = my_nid;
    __syncthreads();

    // Block reduction for sum (but we can't just sum because different nation_ids)
    // For Q5, each block may have multiple nations; we need to atomically add to correct bucket
    // Better: per-thread atomicAdd to global out array based on nation_id
    // Since groups are small (~25 nations), collisions are rare; atomicAdd is fine here
    // but we still use shared mem reduce per nation? No, too complex.
    // Simpler: each thread does atomicAdd to its nation's bucket directly.
    // To reduce atomics, we do a block-level gather of distinct nations first? Overkill.
    // For small num_nations (<=25), per-thread atomicAdd is acceptable since blocks >> nations.
    // Let's just let each surviving thread do atomicAdd.
    
    // Reset shared memory for potential block-level reduction per nation (complex)
    // Easiest fusion: each thread directly atomics to out_ptr[nation_id]
    // Since num_nations <= 25, contention is low.
    
    __syncthreads(); // ensure all shared stores visible
    
    // Use shared memory to handle at most 1024 threads per block, each adding to its nation
    // To reduce global atomics from 1024 to at most num_nations per block:
    // For each possible nation_id (0..num_nations-1), reduce in shared memory then single atomic
    
    if (threadIdx.x < num_nations) {
        int64_t target_nation = (int64_t)threadIdx.x;
        double sum_for_nation = 0.0;
        
        // Check each thread in block for this nation
        // We'll do a parallel reduction across threads for each nation
        // Since num_nations is small (< 100 typically), this is fine
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (block_nation[i] == target_nation) {
                sum_for_nation += block_revenue[i];
            }
        }
        
        if (sum_for_nation != 0.0) {
            atomicAdd(&out_ptr[target_nation], sum_for_nation);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> q5_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor nation_id,
    int64_t num_nations
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(nation_id.scalar_type() == torch::kInt64);
    
    // Move to CUDA if not already
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!nation_id.is_cuda()) nation_id = nation_id.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    nation_id = nation_id.contiguous();
    
    int64_t n = price.numel();
    auto out = torch::zeros({num_nations}, price.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q5_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        nation_id.data_ptr<int64_t>(),
        out.data_ptr<double>(),
        num_nations,
        n
    );
    
    // Create sorted ordering
    // We'll do this in PyTorch land for simplicity; it's cheap for 25 elements
    return {out, torch::arange(num_nations, torch::TensorOptions().dtype(torch::kInt64).device(price.device()))};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 revenue grouped by nation (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q5_cuda_")
    cu_path = os.path.join(tmpdir, "_q5_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    # Import linecache to fix caching issues
    import linecache
    linecache.checkcache(cu_path)
    return load(
        name="q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs on GPU: fused CUDA kernel to compute revenue per nation.
    Returns (nation_ids sorted by revenue desc, revenue sorted desc).
    
    Why faster: single fused kernel replaces scatter_add + argsort on GPU,
    reducing kernel launch overhead and memory traffic.
    """
    out_rev, _ = _EXT.q5_launcher(
        l_extendedprice, l_discount, nation_id, num_nations
    )
    # Sort by revenue descending in PyTorch (cheap for up to 25 nations)
    order = torch.argsort(out_rev, descending=True)
    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=nation_id.device)
    return nation_ids[order], out_rev[order]


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
         FROM customer, orders, lineitem, supplier, nation, region
         WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
           AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
           AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
           AND r_name = 'ASIA'
           AND o_orderdate >= DATE '1994-01-01'
           AND o_orderdate <  DATE '1995-01-01'
         GROUP BY n_name ORDER BY revenue DESC;
    Description: TPC-H Q5 — Local supplier volume for a region over one year.
    Params:
      region_name (str): region to filter on.
      orderdate_lo (str): inclusive lower bound on order date.
      orderdate_hi (str): exclusive upper bound on order date.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region
    region_f = df_region[df_region["r_name"] == region_name]
    # Nation within region
    nation_f = df_nation.merge(region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey")

    # Orders filtered by date
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # Join customer with orders on custkey
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )

    # Join with lineitem on orderkey
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey", right_on="l_orderkey",
    )

    # Join with supplier on suppkey AND c_nationkey = s_nationkey
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey", right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation on s_nationkey = n_nationkey
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey", right_on="n_nationkey",
    )

    # Build nation_id mapping from distinct nations in filter
    nations_unique = nation_f[["n_nationkey", "n_name"]].drop_duplicates().sort_values("n_nationkey").reset_index(drop=True)
    nations_unique["nation_id"] = cudf.Series(range(len(nations_unique)), dtype="int64")

    final_df = final_df.merge(nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey")

    num_nations = int(len(nations_unique))

    nation_ids_t, revenue_t = _query_core(
        torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64),
        num_nations,
    )

    # Map back nation_id to n_name
    result_ids = cudf.Series(nation_ids_t, dtype="int64")
    result_rev = cudf.Series(revenue_t, dtype="float64")
    result = cudf.DataFrame({"nation_id": result_ids, "revenue": result_rev})
    result = result.merge(nations_unique[["nation_id", "n_name"]], on="nation_id")
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({
        "n_name": result["n_name"],
        "revenue": result["revenue"],
    })
