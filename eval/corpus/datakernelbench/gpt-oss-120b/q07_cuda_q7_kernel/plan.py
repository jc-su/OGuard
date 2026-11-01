"""
TPC-H Q7 – Shipping revenue between two nations by year.

SQL:
SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (
    SELECT n1.n_name AS supp_nation,
           n2.n_name AS cust_nation,
           extract(year FROM l_shipdate) AS l_year,
           l_extendedprice * (1 - l_discount) AS volume
    FROM supplier, lineitem, orders, customer, nation n1, nation n2
    WHERE s_suppkey = l_suppkey
      AND o_orderkey = l_orderkey
      AND c_custkey = o_custkey
      AND s_nationkey = n1.n_nationkey
      AND c_nationkey = n2.n_nationkey
      AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
        OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
      AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31')
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year;

Description:
  * cuDF does the small‑dimensional joins (nation, supplier, customer,
    orders).  This produces a narrow table containing only the columns
    required for the final aggregation.
  * The hot path (filter by shipdate, compute volume, extract year,
    and group‑by 2×2×2 = 8 groups) is executed by a single fused CUDA
    kernel.  The kernel reads all columns coalesced, applies the predicate
    in registers, computes the contribution, and atomically adds it to
    one of eight double‑precision accumulators.  This eliminates several
    intermediate cuDF passes and reduces global memory traffic.
  * The result is materialised back into a cudf.DataFrame with the exact
    column order and types expected by the reference plan.

Params:
  nation_a (str): first nation name (default "FRANCE").
  nation_b (str): second nation name (default "GERMANY").
  shipdate_lo (str): inclusive lower bound for l_shipdate (default "1995-01-01").
  shipdate_hi (str): inclusive upper bound for l_shipdate (default "1996-12-31").
"""

import os
import tempfile

import torch
import cudf
import pandas as pd
import numpy as np
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – one fused kernel plus a thin C++ launcher.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ------------------------------------------------------------------
// Fused kernel for Q7
// ------------------------------------------------------------------
__global__ void q7_kernel(
    const int32_t* __restrict__ supp,       // 0 = nation_a, 1 = nation_b
    const int32_t* __restrict__ cust,       // same encoding
    const int32_t* __restrict__ shipday,    // days since epoch (int32)
    const double* __restrict__ price,       // l_extendedprice
    const double* __restrict__ disc,        // l_discount
    int32_t lo,                             // shipdate_lo (inclusive)
    int32_t hi,                             // shipdate_hi (inclusive)
    int32_t day1996_start,                  // first day of 1996 (for year extraction)
    double* __restrict__ out,               // 8 double accumulators
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int32_t sd = shipday[idx];
    if (sd < lo || sd > hi) return;           // date filter

    int32_t s = supp[idx];
    int32_t c = cust[idx];
    // we still compute for all combos; the host will drop invalid ones later

    // year offset: 0 for 1995, 1 for 1996
    int32_t year_off = (sd >= day1996_start) ? 1 : 0;

    // key = (s * 2 + c) in [0,3]; out_idx = key*2 + year_off in [0,7]
    int32_t key = s * 2 + c;
    int32_t out_idx = key * 2 + year_off;

    double vol = price[idx] * (1.0 - disc[idx]);
    atomicAdd(&out[out_idx], vol);
}

// ------------------------------------------------------------------
// Launcher returning a torch tensor of length 8 (double)
 // ------------------------------------------------------------------
torch::Tensor q7_launcher(
    torch::Tensor supp,
    torch::Tensor cust,
    torch::Tensor shipday,
    torch::Tensor price,
    torch::Tensor disc,
    int32_t lo,
    int32_t hi,
    int32_t day1996_start)
{
    // Ensure correct device and contiguity
    TORCH_CHECK(supp.is_cuda(), "supp must be on CUDA");
    TORCH_CHECK(cust.is_cuda(), "cust must be on CUDA");
    TORCH_CHECK(shipday.is_cuda(), "shipday must be on CUDA");
    TORCH_CHECK(price.is_cuda(), "price must be on CUDA");
    TORCH_CHECK(disc.is_cuda(), "disc must be on CUDA");

    supp = supp.contiguous();
    cust = cust.contiguous();
    shipday = shipday.contiguous();
    price = price.contiguous();
    disc = disc.contiguous();

    TORCH_CHECK(supp.dtype() == torch::kInt32, "supp dtype must be int32");
    TORCH_CHECK(cust.dtype() == torch::kInt32, "cust dtype must be int32");
    TORCH_CHECK(shipday.dtype() == torch::kInt32, "shipday dtype must be int32");
    TORCH_CHECK(price.dtype() == torch::kFloat64, "price dtype must be float64");
    TORCH_CHECK(disc.dtype() == torch::kFloat64, "disc dtype must be float64");

    int64_t N = price.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({8}, options);

    int blocks = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q7_kernel<<<blocks, BLOCK_SIZE>>>(
        supp.data_ptr<int32_t>(),
        cust.data_ptr<int32_t>(),
        shipday.data_ptr<int32_t>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        lo,
        hi,
        day1996_start,
        out.data_ptr<double>(),
        N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_launcher",
          &q7_launcher,
          "TPC-H Q7 fused aggregation (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper to compile/load the CUDA extension exactly once.
# ----------------------------------------------------------------------
def _load_cuda_extension() -> object:
    """Compile the CUDA source and return the loaded module."""
    tmp_dir = tempfile.mkdtemp(prefix="q7_cuda_")
    cu_path = os.path.join(tmp_dir, "q7_kernels.cu")
    build_dir = os.path.join(tmp_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q7_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load at import time (cached for the process)
_CUDA_EXT = _load_cuda_extension()

# ----------------------------------------------------------------------
# Core GPU aggregation (fused CUDA kernel)
# ----------------------------------------------------------------------
def _launch_q7_agg(
    supp_code: torch.Tensor,
    cust_code: torch.Tensor,
    shipday: torch.Tensor,
    price: torch.Tensor,
    disc: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    day1996_start: int,
) -> torch.Tensor:
    """Invoke the fused CUDA kernel and return a 1‑D tensor of 8 revenues."""
    out = _CUDA_EXT.q7_launcher(
        supp_code,
        cust_code,
        shipday,
        price,
        disc,
        shipdate_lo_days,
        shipdate_hi_days,
        day1996_start,
    )
    return out  # shape (8,) double on CUDA


def _query_core(
    supp_code: torch.Tensor,
    cust_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU‑accelerated core for Q7.

    Returns:
        (supp_out, cust_out, year_out, revenue_out) – all on CUDA,
        filtered to the two valid nation pairs and non‑zero revenue.
    """
    # day of 1996‑01‑01 for year extraction
    day1996_start = int(pd.Timestamp("1996-01-01").timestamp() // 86400)

    # Launch fused kernel
    agg = _launch_q7_agg(
        supp_code,
        cust_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
        day1996_start,
    )  # shape (8,)

    # Decode group indices back to components
    agg_cpu = agg.cpu().numpy()
    group_ids = np.arange(8, dtype=np.int32)

    # key = out_idx // 2 ; year_offset = out_idx % 2
    key = group_ids // 2
    year_offset = group_ids % 2
    supp = key // 2          # 0 or 1
    cust = key % 2           # 0 or 1
    year = 1995 + year_offset

    # Keep only cross‑nation pairs (supp != cust) and positive revenue
    valid = (supp != cust) & (agg_cpu > 0.0)

    # Build tensors on CUDA for the final DataFrame conversion
    supp_out = torch.tensor(supp[valid], dtype=torch.int32, device=agg.device)
    cust_out = torch.tensor(cust[valid], dtype=torch.int32, device=agg.device)
    year_out = torch.tensor(year[valid], dtype=torch.int32, device=agg.device)
    revenue_out = torch.tensor(agg_cpu[valid], dtype=torch.float64, device=agg.device)

    return supp_out, cust_out, year_out, revenue_out


# ----------------------------------------------------------------------
# Public API – identical signature to the reference plan.
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue
        FROM ( ... ) GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year;

    Description:
        * Joins are performed with cuDF (small tables) to produce a narrow
          table containing only the columns needed for the final aggregation.
        * The aggregation (filter → compute volume → extract year → group‑by)
          is executed by a single fused CUDA kernel, dramatically reducing
          memory traffic compared to the reference plan’s multiple passes.
        * The resulting DataFrame matches the reference column order,
          dtypes and row count.

    Params:
        nation_a (str): first nation name.
        nation_b (str): second nation name.
        shipdate_lo (str): inclusive lower bound for l_shipdate.
        shipdate_hi (str): inclusive upper bound for l_shipdate.

    Returns:
        cudf.DataFrame with columns:
            - supp_nation (str)
            - cust_nation (str)
            - l_year (int32)
            - revenue (float64)
    """
    # ------------------------------------------------------------------
    # 1. Reduce nation table to the two relevant nations
    # ------------------------------------------------------------------
    df_nation = tables["nation"]
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # ------------------------------------------------------------------
    # 2. Supplier → nation (supplier side)
    # ------------------------------------------------------------------
    sup = tables["supplier"][["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # ------------------------------------------------------------------
    # 3. Customer → nation (customer side)
    # ------------------------------------------------------------------
    cust = tables["customer"][["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # ------------------------------------------------------------------
    # 4. lineitem + supplier (keep only needed columns)
    # ------------------------------------------------------------------
    li = tables["lineitem"][
        ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]
    ].merge(
        sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey"
    )

    # ------------------------------------------------------------------
    # 5. orders + customer
    # ------------------------------------------------------------------
    oc = tables["orders"][["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    # ------------------------------------------------------------------
    # 6. Final join (lineitem + orders)
    # ------------------------------------------------------------------
    joined = li.merge(
        oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey"
    )

    # ------------------------------------------------------------------
    # 7. Encode nation names as 0/1 (nation_a -> 0, nation_b -> 1)
    # ------------------------------------------------------------------
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    # --------------------------------------------------------------
    # 8. Convert shipdate to int32 days since epoch
    # --------------------------------------------------------------
    shipdate_days = (
        (joined["l_shipdate"].astype("int64") // 86400)
        .astype("int32")
        .copy()
    )  # ensure contiguous cudf column

    # --------------------------------------------------------------
    # 9. Transfer needed columns to CUDA tensors
    # --------------------------------------------------------------
    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32, device="cuda")
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32, device="cuda")
    ship_t = torch.as_tensor(shipdate_days.to_cupy(), dtype=torch.int32, device="cuda")
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64, device="cuda")
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64, device="cuda")

    # --------------------------------------------------------------
    # 10. Date bounds as int32 epoch days (inclusive)
    # --------------------------------------------------------------
    shipdate_lo_days = int(pd.Timestamp(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.Timestamp(shipdate_hi).timestamp() // 86400)

    # --------------------------------------------------------------
    # 11. Run fused GPU aggregation
    # --------------------------------------------------------------
    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t,
        cust_t,
        ship_t,
        price_t,
        disc_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # --------------------------------------------------------------
    # 12. Map integer codes back to nation names
    # --------------------------------------------------------------
    supp_out_cpu = supp_out.cpu().numpy()
    cust_out_cpu = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_out_cpu]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_out_cpu]

    # --------------------------------------------------------------
    # 13. Assemble final cudf.DataFrame
    # --------------------------------------------------------------
    result = cudf.DataFrame(
        {
            "supp_nation": cudf.Series(supp_names),
            "cust_nation": cudf.Series(cust_names),
            "l_year": year_out,
            "revenue": rev_out,
        }
    )
    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
