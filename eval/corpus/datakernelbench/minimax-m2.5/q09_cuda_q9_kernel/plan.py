import os
import tempfile
import torch
from torch.utils.cpp_extension import load
import cudf
import pandas as pd

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q9_kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    const int32_t* __restrict__ nation_key,
    const int32_t* __restrict__ year,
    double* __restrict__ out,
    int64_t n,
    int32_t num_nations,
    int32_t year_min,
    int32_t year_range
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double amt = price[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
        int32_t nation = nation_key[idx];
        int32_t yr = year[idx];
        int32_t group = nation * year_range + (yr - year_min);
        atomicAdd(&out[group], amt);
    }
}

void q9_aggregate(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity,
    torch::Tensor nation_key,
    torch::Tensor year,
    torch::Tensor out,
    int32_t num_nations,
    int32_t year_min,
    int32_t year_range
) {
    TORCH_CHECK(price.is_cuda());
    TORCH_CHECK(discount.is_cuda());
    TORCH_CHECK(supplycost.is_cuda());
    TORCH_CHECK(quantity.is_cuda());
    TORCH_CHECK(nation_key.is_cuda());
    TORCH_CHECK(year.is_cuda());
    TORCH_CHECK(out.is_cuda());

    price = price.contiguous();
    discount = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity = quantity.contiguous();
    nation_key = nation_key.contiguous();
    year = year.contiguous();

    int64_t n = price.numel();
    const int block_size = BLOCK_SIZE;
    int grid = (n + block_size - 1) / block_size;
    q9_kernel<<<grid, block_size>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        supplycost.data_ptr<double>(),
        quantity.data_ptr<double>(),
        nation_key.data_ptr<int32_t>(),
        year.data_ptr<int32_t>(),
        out.data_ptr<double>(),
        n,
        num_nations,
        year_min,
        year_range
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_aggregate", &q9_aggregate, "Q9 aggregate kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "_q9_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q9_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    Description: TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year for parts whose name matches a pattern. This implementation performs joins in cuDF, then uses a fused CUDA kernel to compute and aggregate the profit amounts in a single pass, avoiding materialization of the intermediate amount column.
    Params: name_pattern (str): substring to filter part names (without % wildcards).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # Build nation key -> name mapping on CPU (string column cannot go to cupy)
    nation_key_to_name = dict(zip(
        df_nation["n_nationkey"].to_pandas(),
        df_nation["n_name"].to_pandas()
    ))
    num_nations = len(df_nation)  # typically 25

    # Filter part by name pattern
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # Joins (same order as reference)
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

    # Empty result case
    if len(joined) == 0:
        return cudf.DataFrame({
            "nation": cudf.Series([], dtype="object"),
            "o_year": cudf.Series([], dtype="int32"),
            "sum_profit": cudf.Series([], dtype="float64"),
        })

    # Extract needed columns as cuPy arrays and then as CUDA tensors
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64, device="cuda").contiguous()
    discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64, device="cuda").contiguous()
    supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64, device="cuda").contiguous()
    quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64, device="cuda").contiguous()
    nation_t = torch.as_tensor(joined["n_nationkey"].astype("int32").to_cupy(), dtype=torch.int32, device="cuda").contiguous()
    year_t = torch.as_tensor(joined["o_orderdate"].dt.year.astype("int32").to_cupy(), dtype=torch.int32, device="cuda").contiguous()

    # Determine year range
    year_arr = joined["o_orderdate"].dt.year.astype("int32").to_cupy()
    year_min = int(year_arr.min())
    year_max = int(year_arr.max())
    year_range = year_max - year_min + 1

    num_groups = num_nations * year_range

    # Allocate output tensor for aggregated sums
    sum_per_group = torch.zeros(num_groups, dtype=torch.float64, device="cuda")

    # Launch CUDA kernel
    _EXT.q9_aggregate(
        price_t,
        discount_t,
        supplycost_t,
        quantity_t,
        nation_t,
        year_t,
        sum_per_group,
        num_nations,
        year_min,
        year_range,
    )

    # Bring results back to host
    sums_np = sum_per_group.cpu().numpy()

    # Generate result rows using the CPU mapping
    rows = []
    for nid in range(num_nations):
        name = nation_key_to_name.get(nid)
        if name is None:
            continue
        for yr in range(year_max, year_min - 1, -1):
            idx = nid * year_range + (yr - year_min)
            s = sums_np[idx]
            if s != 0.0:
                rows.append((name, yr, s))

    # Construct final DataFrame with correct order
    if not rows:
        result = cudf.DataFrame({
            "nation": cudf.Series([], dtype="object"),
            "o_year": cudf.Series([], dtype="int32"),
            "sum_profit": cudf.Series([], dtype="float64"),
        })
    else:
        pdf = pd.DataFrame(rows, columns=["nation", "o_year", "sum_profit"])
        pdf = pdf.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)
        result = cudf.DataFrame(pdf)

    return result
