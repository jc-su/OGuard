import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA kernel source (fused aggregation)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Upper bound on number of distinct nations (TPC‑H has 25)
#ifndef MAX_GROUPS
#define MAX_GROUPS 64
#endif

// ----------------------------------------------------------------------
// Kernel: per‑nation revenue aggregation
// ----------------------------------------------------------------------
__global__ void agg_kernel(
    const int64_t * __restrict__ nation_id,
    const double  * __restrict__ price,
    const double  * __restrict__ discount,
    double        * __restrict__ out,
    int64_t         n,
    int64_t         num_groups)
{
    // Shared‑memory buffer: one accumulator per group.
    __shared__ double s_out[MAX_GROUPS];

    // Initialise shared buffer.
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        s_out[i] = 0.0;
    }
    __syncthreads();

    // Process assigned rows.
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = nation_id[idx];                 // group identifier
        double rev = price[idx] * (1.0 - discount[idx]); // revenue contribution
        atomicAdd(&s_out[g], rev);                 // lock‑free per‑group update
    }
    __syncthreads();

    // Flush block‑level results to global memory (one atomic per group per block).
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        atomicAdd(&out[i], s_out[i]);
    }
}

// ----------------------------------------------------------------------
// Launcher: Python‑visible entry point
// ----------------------------------------------------------------------
torch::Tensor agg_launcher(
    torch::Tensor nation_id,
    torch::Tensor price,
    torch::Tensor discount,
    int64_t num_groups)
{
    TORCH_CHECK(nation_id.scalar_type() == torch::kInt64,
                "nation_id must be int64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64,
                "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64,
                "discount must be float64");

    // Ensure tensors live on CUDA and are contiguous.
    if (!nation_id.is_cuda())   nation_id   = nation_id.cuda();
    if (!price.is_cuda())       price       = price.cuda();
    if (!discount.is_cuda())    discount    = discount.cuda();

    nation_id = nation_id.contiguous();
    price     = price.contiguous();
    discount  = discount.contiguous();

    int64_t n = nation_id.numel();

    // Output buffer: one double per nation.
    auto out = torch::zeros({num_groups}, price.options());

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    agg_kernel<<<grid, block>>>(
        nation_id.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n,
        num_groups
    );

    // Synchronize and propagate any launch errors.
    cudaDeviceSynchronize();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("agg_launcher", &agg_launcher,
          "TPC‑H Q5 fused aggregation (CUDA)");
}
'''

def _load_cuda_extension() -> torch.nn.Module:
    """Compile the CUDA extension once and cache the module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q5_cuda_")
    cu_path = os.path.join(tmpdir, "q5_kernel.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="tpch_q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load (or compile) the CUDA extension.
_EXT = _load_cuda_extension()


def _launch_agg(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """
    Execute the fused aggregation kernel.
    Returns a ``torch.float64`` tensor of shape (num_nations,).
    """
    return _EXT.agg_launcher(
        nation_id,
        l_extendedprice,
        l_discount,
        num_nations,
    )


def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑nation revenue and return IDs/revenues sorted descending.
    """
    revenue = _launch_agg(
        nation_id,
        l_extendedprice,
        l_discount,
        num_nations,
    )
    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=revenue.device)
    order = torch.argsort(revenue, descending=True)
    return nation_ids[order], revenue[order]


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT n_name,
               SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM   customer, orders, lineitem, supplier, nation, region
        WHERE  c_custkey = o_custkey
               AND l_orderkey = o_orderkey
               AND l_suppkey = s_suppkey
               AND c_nationkey = s_nationkey
               AND s_nationkey = n_nationkey
               AND n_regionkey = r_regionkey
               AND r_name = :region_name
               AND o_orderdate >= :orderdate_lo
               AND o_orderdate <  :orderdate_hi
        GROUP BY n_name
        ORDER BY revenue DESC;

    Description:
        All joins are performed with cuDF (GPU‑accelerated).  The
        aggregation (revenue per nation) is off‑loaded to a custom CUDA kernel
        that fuses the revenue expression and uses a block‑wise reduction with
        a tiny number of global atomics, eliminating the per‑thread atomic
        contention of the reference ``torch.scatter_add`` implementation.

    Params:
        region_name (str): Region filter (e.g., 'ASIA').
        orderdate_lo (str): Inclusive lower bound for ``o_orderdate`` (YYYY‑MM‑DD).
        orderdate_hi (str): Exclusive upper bound for ``o_orderdate``.
    Returns:
        cudf.DataFrame with columns ``n_name`` (str) and ``revenue`` (float64)
        sorted by revenue descending.
    """
    # ------------------------------------------------------------------
    # Load individual tables.
    # ------------------------------------------------------------------
    df_customer = tables["customer"]
    df_orders   = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation   = tables["nation"]
    df_region   = tables["region"]

    # ------------------------------------------------------------------
    # Convert date bounds to integer days since epoch.
    # ------------------------------------------------------------------
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # Region filter.
    # ------------------------------------------------------------------
    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]

    # ------------------------------------------------------------------
    # Nations belonging to the chosen region.
    # ------------------------------------------------------------------
    nation_f = df_nation.merge(
        region_f,
        left_on="n_regionkey",
        right_on="r_regionkey"
    )  # columns: n_nationkey, n_name, n_regionkey, r_regionkey

    # ------------------------------------------------------------------
    # Orders filtered by order date.
    # ------------------------------------------------------------------
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days) &
        (orders_f["o_orderdate_days"] < hi_days)
    ][["o_orderkey", "o_custkey"]]

    # ------------------------------------------------------------------
    # Join chain up to the final projection.
    # ------------------------------------------------------------------
    # Customer ↔ Orders
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f,
        left_on="c_custkey",
        right_on="o_custkey"
    )[["c_nationkey", "o_orderkey"]]

    # Orders ↔ LineItem
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey"
    )[["c_nationkey", "l_suppkey", "l_extendedprice", "l_discount"]]

    # LineItem ↔ Supplier
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey"
    )[["c_nationkey", "s_nationkey", "l_extendedprice", "l_discount"]]

    # Enforce equality of customer and supplier nations.
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation to get nation name and key.
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey"
    )[["n_nationkey", "n_name", "l_extendedprice", "l_discount"]]

    # ------------------------------------------------------------------
    # Build a dense nation_id mapping (0 … num_nations‑1).
    # ------------------------------------------------------------------
    nations_unique = (
        nation_f[["n_nationkey", "n_name"]]
        .drop_duplicates()
        .sort_values("n_nationkey")
        .reset_index(drop=True)
    )
    nations_unique["nation_id"] = cudf.Series(
        range(len(nations_unique)), dtype="int64"
    )

    final_df = final_df.merge(
        nations_unique[["n_nationkey", "nation_id"]],
        on="n_nationkey"
    )
    num_nations = int(len(nations_unique))

    # ------------------------------------------------------------------
    # Convert required columns to torch tensors (already on GPU).
    # ------------------------------------------------------------------
    nation_id_t = torch.as_tensor(
        final_df["nation_id"].to_cupy(), dtype=torch.int64
    )
    price_t = torch.as_tensor(
        final_df["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    discount_t = torch.as_tensor(
        final_df["l_discount"].to_cupy(), dtype=torch.float64
    )

    # ------------------------------------------------------------------
    # Aggregation via fused CUDA kernel.
    # ------------------------------------------------------------------
    nation_ids_t, revenue_t = _query_core(
        nation_id_t, price_t, discount_t, num_nations
    )

    # ------------------------------------------------------------------
    # Assemble final result DataFrame.
    # ------------------------------------------------------------------
    result_ids = cudf.Series(nation_ids_t.cpu(), dtype="int64")
    result_rev = cudf.Series(revenue_t.cpu(), dtype="float64")
    result = cudf.DataFrame(
        {"nation_id": result_ids, "revenue": result_rev}
    )
    result = result.merge(
        nations_unique[["nation_id", "n_name"]], on="nation_id"
    )
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame(
        {"n_name": result["n_name"], "revenue": result["revenue"]}
    )
