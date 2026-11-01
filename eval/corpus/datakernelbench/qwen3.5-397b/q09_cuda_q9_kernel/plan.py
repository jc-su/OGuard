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

__global__ void q9_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ nation_ptr,
    const int32_t* __restrict__ year_ptr,
    double* __restrict__ group_sums,
    int64_t n_elements,
    int32_t min_year,
    int32_t n_groups,
    int32_t n_years
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Grid-stride loop for memory-bound scan
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t nation = nation_ptr[i];
        int32_t year = year_ptr[i];
        int32_t year_offset = year - min_year;
        
        // Validate indices
        if (nation >= 0 && nation < 25 && year_offset >= 0 && year_offset < n_years) {
            int32_t group_idx = nation * n_years + year_offset;
            
            if (group_idx >= 0 && group_idx < n_groups) {
                // Compute amount = extendedprice * (1 - discount) - supplycost * quantity
                double amount = price_ptr[i] * (1.0 - discount_ptr[i]) 
                              - supplycost_ptr[i] * quantity_ptr[i];
                
                // Direct atomic add to global memory (correct, no race conditions)
                atomicAdd(&group_sums[group_idx], amount);
            }
        }
    }
}

torch::Tensor q9_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity,
    torch::Tensor nation,
    torch::Tensor year,
    int32_t min_year,
    int32_t n_groups,
    int32_t n_years
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(nation.scalar_type() == torch::kInt32, "nation must be int32");
    TORCH_CHECK(year.scalar_type() == torch::kInt32, "year must be int32");
    
    // Ensure CUDA and contiguous
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!nation.is_cuda()) nation = nation.cuda();
    if (!year.is_cuda()) year = year.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity = quantity.contiguous();
    nation = nation.contiguous();
    year = year.contiguous();
    
    int64_t n = price.numel();
    TORCH_CHECK(n == discount.numel(), "mismatched tensor lengths");
    TORCH_CHECK(n == supplycost.numel(), "mismatched tensor lengths");
    TORCH_CHECK(n == quantity.numel(), "mismatched tensor lengths");
    TORCH_CHECK(n == nation.numel(), "mismatched tensor lengths");
    TORCH_CHECK(n == year.numel(), "mismatched tensor lengths");
    
    // Output tensor for group sums (one per nation*year combination)
    auto out = torch::zeros({n_groups}, price.options());
    
    // Configure kernel launch for H100
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Cap blocks to avoid oversubscription on H100 (132 SMs)
    if (blocks > 132 * 8) blocks = 132 * 8;
    
    q9_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        supplycost.data_ptr<double>(),
        quantity.data_ptr<double>(),
        nation.data_ptr<int32_t>(),
        year.data_ptr<int32_t>(),
        out.data_ptr<double>(),
        n,
        min_year,
        n_groups,
        n_years
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_launcher", &q9_launcher, "TPC-H Q9 profit aggregation (CUDA fused kernel)");
}
'''

def _load_cuda():
    """
    Compile CUDA kernel at import time using torch.utils.cpp_extension.load().
    Writes _CUDA_SOURCE to temp file, compiles with nvcc for H100 (sm_90).
    """
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "_q9_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    return load(
        name="q9_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "-std=c++17"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Compile once at module import
_EXT = _load_cuda()


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
    nation_ids: torch.Tensor,
    order_years: torch.Tensor,
    min_year: int,
    n_groups: int,
    n_years: int,
) -> torch.Tensor:
    """
    Fused CUDA kernel for Q9 amount computation + groupby aggregation.
    
    Runs on GPU via hand-written CUDA kernel compiled with torch.utils.cpp_extension.load().
    
    Why faster than reference:
    - Single fused kernel pass: project (amount) + groupby accumulate in one memory scan
    - Grid-stride loop for optimal memory coalescing on H100
    - Direct atomicAdd to global memory (correct, no race conditions)
    - Avoids multiple cuDF/PyTorch round-trips for intermediate tensors
    
    Args:
        l_extendedprice: float64 tensor of lineitem extended prices
        l_discount: float64 tensor of lineitem discounts
        ps_supplycost: float64 tensor of partsupp supply costs
        l_quantity: float64 tensor of lineitem quantities
        nation_ids: int32 tensor of nation indices (0-24)
        order_years: int32 tensor of order years
        min_year: minimum year value for group index offset
        n_groups: total number of (nation, year) groups
        n_years: number of distinct years in the data
    
    Returns:
        torch.Tensor: float64 tensor of shape (n_groups,) with sum_profit per group
    """
    group_sums = _EXT.q9_launcher(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
        nation_ids,
        order_years,
        min_year,
        n_groups,
        n_years,
    )
    return group_sums


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL:
        SELECT nation, o_year, sum(amount) AS sum_profit
        FROM (
            SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year,
                   l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
            FROM part, supplier, lineitem, partsupp, orders, nation
            WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey
              AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey
              AND p_name LIKE '%green%'
        ) AS profit
        GROUP BY nation, o_year
        ORDER BY nation, o_year DESC;
    
    Description:
        TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year
        for parts whose name matches a pattern.
        
        GPU Acceleration Strategy:
        - cuDF handles: string filtering (p_name LIKE), multi-table joins, date extraction
        - CUDA kernel handles: amount computation + groupby aggregation in one fused pass
        - Benefits: single memory scan for hot path, grid-stride loop, fewer HBM passes
        - Precision: float64 throughout for TPC-H decimal accuracy
    
    Params:
        name_pattern (str): substring to filter part names (without % wildcards). Default: "green"
    
    Returns:
        cudf.DataFrame: columns [nation, o_year, sum_profit] sorted by nation ASC, o_year DESC
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]
    
    # Filter parts by name pattern (string operation stays in cuDF)
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]
    
    # Join chain: lineitem -> parts -> partsupp -> supplier -> orders -> nation
    li_parts = df_lineitem.merge(parts_filtered, left_on="l_partkey", right_on="p_partkey")
    
    li_ps = li_parts.merge(
        df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )
    
    li_ps_s = li_ps.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    
    li_ps_s_o = li_ps_s.merge(
        df_orders[["o_orderkey", "o_orderdate"]],
        left_on="l_orderkey",
        right_on="o_orderkey",
    )
    
    joined = li_ps_s_o.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )
    
    # Extract year from order date (int32 for efficient kernel indexing)
    order_years_series = joined["o_orderdate"].dt.year.astype("int32")
    min_year = int(order_years_series.min())
    max_year = int(order_years_series.max())
    n_years = max_year - min_year + 1
    
    # Get unique nations present in the joined data (not all 25 nations)
    # Use to_arrow().to_pylist() instead of tolist() for cuDF compatibility
    nation_unique = joined["n_name"].unique().to_arrow().to_pylist()
    nation_to_id = {name: idx for idx, name in enumerate(nation_unique)}
    nation_ids_series = joined["n_name"].map(nation_to_id).astype("int32")
    
    # Number of groups = actual nations in data * years
    n_nations = len(nation_unique)
    n_groups = n_nations * n_years
    
    # Convert joined columns to GPU tensors (float64 for numerics, int32 for indices)
    price = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    supplycost = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    quantity = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)
    nation_ids = torch.as_tensor(nation_ids_series.to_cupy(), dtype=torch.int32)
    order_years = torch.as_tensor(order_years_series.to_cupy(), dtype=torch.int32)
    
    # Run fused CUDA kernel: project + groupby accumulate in one pass
    group_sums = _query_core(
        price,
        discount,
        supplycost,
        quantity,
        nation_ids,
        order_years,
        min_year,
        n_groups,
        n_years,
    )
    
    # Decode group sums back to (nation, year) pairs
    # Only include groups that have non-zero sums (matching groupby behavior)
    results = []
    for nation_idx in range(n_nations):
        nation_name = nation_unique[nation_idx]
        for year_offset in range(n_years):
            year = min_year + year_offset
            group_idx = nation_idx * n_years + year_offset
            sum_profit = float(group_sums[group_idx].item())
            # Include all groups with data (non-zero sum after aggregation)
            if sum_profit != 0.0:
                results.append({
                    "nation": nation_name,
                    "o_year": year,
                    "sum_profit": sum_profit,
                })
    
    # Build result DataFrame
    if results:
        result_df = cudf.DataFrame(results)
        # Sort by nation ASC, o_year DESC (matching SQL ORDER BY)
        result_df = result_df.sort_values(["nation", "o_year"], ascending=[True, False])
        result_df = result_df.reset_index(drop=True)
    else:
        result_df = cudf.DataFrame({
            "nation": cudf.Series([], dtype="object"),
            "o_year": cudf.Series([], dtype="int32"),
            "sum_profit": cudf.Series([], dtype="float64"),
        })
    
    return result_df
