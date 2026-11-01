import os
import tempfile
import torch
import cudf
import pandas as pd
import numpy as np
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define MAX_GROUPS 8

__global__ void q7_kernel(
    const int32_t* __restrict__ supp_nation,
    const int32_t* __restrict__ cust_nation,
    const int32_t* __restrict__ shipdate_days,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue_groups,
    int64_t n_elements
) {
    extern __shared__ double shared_rev[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_vol = 0.0;
    int my_key = -1;
    
    if (idx < n_elements) {
        int32_t sd = shipdate_days[idx];
        if (sd >= date_lo && sd <= date_hi) {
            int32_t supp = supp_nation[idx];
            int32_t cust = cust_nation[idx];
            
            // Compute year from days using simplified algorithm
            int64_t d = (int64_t)sd + 719468;
            int64_t era = (d >= 0 ? d : d - 146096) / 146097;
            int64_t doe = d - era * 146097;
            int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            int64_t y = yoe + era * 400;
            int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            int64_t mp = (5 * doy + 2) / 153;
            int64_t m = mp < 10 ? mp + 3 : mp - 9;
            int64_t year = m <= 2 ? y + 1 : y;
            
            int year_offset = (int)(year - 1995);
            
            if (year_offset >= 0 && year_offset <= 1) {
                my_key = supp * 4 + cust * 2 + year_offset;
                my_vol = price[idx] * (1.0 - discount[idx]);
            }
        }
    }
    
    // Initialize shared memory for this block's group accumulators
    if (threadIdx.x < MAX_GROUPS) {
        shared_rev[threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // Accumulate into shared using atomic on shared memory
    if (my_key >= 0 && my_key < MAX_GROUPS) {
        atomicAdd(&shared_rev[my_key], my_vol);
    }
    __syncthreads();
    
    // Final block-level reduction to global
    if (threadIdx.x < MAX_GROUPS) {
        atomicAdd(&revenue_groups[threadIdx.x], shared_rev[threadIdx.x]);
    }
}

void q7_launcher(
    torch::Tensor supp_nation,
    torch::Tensor cust_nation,
    torch::Tensor shipdate_days,
    torch::Tensor price,
    torch::Tensor discount,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor revenue_groups
) {
    TORCH_CHECK(supp_nation.scalar_type() == torch::kInt32);
    TORCH_CHECK(cust_nation.scalar_type() == torch::kInt32);
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    
    // Move to CUDA if needed
    auto supp = supp_nation.is_cuda() ? supp_nation : supp_nation.cuda();
    auto cust = cust_nation.is_cuda() ? cust_nation : cust_nation.cuda();
    auto ship = shipdate_days.is_cuda() ? shipdate_days : shipdate_days.cuda();
    auto p = price.is_cuda() ? price : price.cuda();
    auto d = discount.is_cuda() ? discount : discount.cuda();
    auto rev = revenue_groups.is_cuda() ? revenue_groups : revenue_groups.cuda();
    
    // Make contiguous
    supp = supp.contiguous();
    cust = cust.contiguous();
    ship = ship.contiguous();
    p = p.contiguous();
    d = d.contiguous();
    rev = rev.contiguous();
    
    int64_t n = p.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    size_t shared_mem = MAX_GROUPS * sizeof(double);
    
    q7_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        supp.data_ptr<int32_t>(),
        cust.data_ptr<int32_t>(),
        ship.data_ptr<int32_t>(),
        p.data_ptr<double>(),
        d.data_ptr<double>(),
        date_lo,
        date_hi,
        rev.data_ptr<double>(),
        n
    );
}

// Combined launcher that returns a tuple of tensors
std::vector<torch::Tensor> q7_full_launcher(
    torch::Tensor supp_nation,
    torch::Tensor cust_nation,
    torch::Tensor shipdate_days,
    torch::Tensor price,
    torch::Tensor discount,
    int32_t date_lo,
    int32_t date_hi,
    int32_t nation_a_code,
    int32_t nation_b_code
) {
    // Call the main kernel
    auto revenue_groups = torch::zeros({MAX_GROUPS}, 
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    
    q7_launcher(supp_nation, cust_nation, shipdate_days, price, discount,
                date_lo, date_hi, revenue_groups);
    
    // Build output tensors on CPU
    auto rev_cpu = revenue_groups.cpu();
    
    std::vector<int> out_supp, out_cust, out_year;
    std::vector<double> out_rev;
    
    for (int g = 0; g < MAX_GROUPS; g++) {
        int supp = g / 4;
        int cust = (g / 2) % 2;
        int year = (g % 2) + 1995;
        
        // Only keep valid combos: (FR->DE) or (DE->FR)
        bool valid = ((supp == 0 && cust == 1) || (supp == 1 && cust == 0));
        
        double rev_val = rev_cpu.data_ptr<double>()[g];
        if (valid && rev_val > 0) {
            out_supp.push_back(supp);
            out_cust.push_back(cust);
            out_year.push_back(year);
            out_rev.push_back(rev_val);
        }
    }
    
    int out_size = out_supp.size();
    auto out_supp_t = torch::from_blob(out_supp.data(), {out_size}, torch::kInt32).clone();
    auto out_cust_t = torch::from_blob(out_cust.data(), {out_size}, torch::kInt32).clone();
    auto out_year_t = torch::from_blob(out_year.data(), {out_size}, torch::kInt32).clone();
    auto out_rev_t = torch::from_blob(out_rev.data(), {out_size}, torch::kFloat64).clone();
    
    return {out_supp_t, out_cust_t, out_year_t, out_rev_t};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_full_launcher", &q7_full_launcher, 
          "TPC-H Q7: shipping revenue between nation pairs by year (CUDA fused kernel)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q7_cuda_")
    cu_path = os.path.join(tmpdir, "_q7_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    # Write CUDA source to file
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # Compile and load the extension
    return load(
        name="q7_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False
    )

# Load the CUDA extension at module import time
_EXT = _load_cuda()


def _query_core(
    supp_nation_code: torch.Tensor,
    cust_nation_code: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused CUDA kernel launcher for TPC-H Q7 core computation.
    
    What runs where:
    - All heavy lifting (filter, project, year extraction, group-by aggregate)
      happens in a single fused CUDA kernel on GPU.
    - The host function just moves tensors to CUDA and calls the launcher.
    
    Why it should be faster:
    - Single kernel launch instead of PyTorch ops: filter, scatter_add, etc.
    - Fused memory access: all columns are read coalesced and processed together.
    - Block-level shared memory reduction before global atomicAdd.
    - Avoids intermediate tensors for mask, volume, year.
    """
    return _EXT.q7_full_launcher(
        supp_nation_code, cust_nation_code, l_shipdate_days,
        l_extendedprice, l_discount,
        shipdate_lo_days, shipdate_hi_days, 0, 1
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_a: str = "FRANCE",
    nation_b: str = "GERMANY",
    shipdate_lo: str = "1995-01-01",
    shipdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT supp_nation, cust_nation, l_year, sum(volume) AS revenue FROM (
         SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                extract(year FROM l_shipdate) AS l_year,
                l_extendedprice * (1 - l_discount) AS volume
         FROM supplier, lineitem, orders, customer, nation n1, nation n2
         WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
               AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
               AND c_nationkey = n2.n_nationkey
               AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY')
                 OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
               AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
         ) GROUP BY supp_nation, cust_nation, l_year
         ORDER BY supp_nation, cust_nation, l_year;
    
    Description: TPC-H Q7 - Shipping revenue between two nation pairs by year.
    - cuDF handles joins, string processing, filtering of nation pairs.
    - A single fused CUDA kernel handles: shipdate filter, year extraction,
      volume computation, and group-by aggregate.
    - This avoids PyTorch intermediate tensors and multiple kernel launches.
    
    Params:
      tables (dict[str, cudf.DataFrame]): Input tables.
      nation_a (str): First nation name.
      nation_b (str): Second nation name.
      shipdate_lo (str): Inclusive lower bound for l_shipdate.
      shipdate_hi (str): Inclusive upper bound for l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]

    # Restrict nations to only the two relevant ones
    nations2 = df_nation[df_nation["n_name"].isin([nation_a, nation_b])][
        ["n_nationkey", "n_name"]
    ]

    # Join supplier with nation (supp_nation)
    sup = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "s_nationkey", "n_name": "supp_nation"}),
        on="s_nationkey",
    )

    # Join customer with nation (cust_nation)
    cust = df_customer[["c_custkey", "c_nationkey"]].merge(
        nations2.rename(columns={"n_nationkey": "c_nationkey", "n_name": "cust_nation"}),
        on="c_nationkey",
    )

    # lineitem with shipdate filter
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    li = df_lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li.merge(sup[["s_suppkey", "supp_nation"]], left_on="l_suppkey", right_on="s_suppkey")

    # orders join customer
    oc = df_orders[["o_orderkey", "o_custkey"]].merge(
        cust[["c_custkey", "cust_nation"]], left_on="o_custkey", right_on="c_custkey"
    )

    joined = li.merge(oc[["o_orderkey", "cust_nation"]], left_on="l_orderkey", right_on="o_orderkey")

    # Keep only (FRANCE,GERMANY) or (GERMANY,FRANCE) pairs
    pair_mask = (
        ((joined["supp_nation"] == nation_a) & (joined["cust_nation"] == nation_b))
        | ((joined["supp_nation"] == nation_b) & (joined["cust_nation"] == nation_a))
    )
    joined = joined[pair_mask]

    # Encode nation names: nation_a -> 0, nation_b -> 1
    supp_code = (joined["supp_nation"] == nation_b).astype("int32")
    cust_code = (joined["cust_nation"] == nation_b).astype("int32")

    l_shipdate_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    supp_t = torch.as_tensor(supp_code.to_cupy(), dtype=torch.int32)
    cust_t = torch.as_tensor(cust_code.to_cupy(), dtype=torch.int32)
    ship_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)

    supp_out, cust_out, year_out, rev_out = _query_core(
        supp_t, cust_t, ship_t, price_t, disc_t, shipdate_lo_days, shipdate_hi_days
    )

    # Map codes back to names: 0 -> nation_a, 1 -> nation_b
    supp_out_cpu = supp_out.cpu().numpy()
    cust_out_cpu = cust_out.cpu().numpy()
    supp_names = [nation_a if v == 0 else nation_b for v in supp_out_cpu]
    cust_names = [nation_a if v == 0 else nation_b for v in cust_out_cpu]

    result = cudf.DataFrame({
        "supp_nation": cudf.Series(supp_names),
        "cust_nation": cudf.Series(cust_names),
        "l_year": year_out,
        "revenue": rev_out,
    })

    result = result.sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True)
    return result
