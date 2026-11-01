import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 512
#define NUM_GROUPS 8

__global__ void q7_kernel(
    const int32_t* __restrict__ supp_nation_code,
    const int32_t* __restrict__ cust_nation_code,
    const int32_t* __restrict__ l_shipdate_days,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_count,
    int64_t n_elements
) {
    __shared__ double block_revenue[NUM_GROUPS][BLOCK_SIZE];
    __shared__ int block_count[NUM_GROUPS][BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_revenue[NUM_GROUPS] = {0.0};
    int my_count[NUM_GROUPS] = {0};

    if (idx < n_elements) {
        int32_t sd = l_shipdate_days[idx];
        bool keep = (sd >= shipdate_lo_days) && (sd <= shipdate_hi_days);
        if (keep) {
            double price = l_extendedprice[idx];
            double disc = l_discount[idx];
            double volume = price * (1.0 - disc);

            // Extract year from shipdate days
            int64_t days = (int64_t)sd + 719468;
            int64_t era = days >= 0 ? days / 146097 : (days - 146096) / 146097;
            int64_t doe = days - era * 146097;
            int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            int32_t calc_y = (int32_t)(yoe + era * 400);
            int32_t doy = (int32_t)doe - (365 * yoe + yoe / 4 - yoe / 100);
            int32_t mp = (5 * doy + 2) / 153;
            int32_t m = mp < 10 ? mp + 3 : mp - 9;
            int32_t year = m <= 2 ? calc_y + 1 : calc_y;

            int32_t supp = supp_nation_code[idx];
            int32_t cust = cust_nation_code[idx];
            int32_t year_offset = year - 1995;

            // Build composite group key
            int64_t key = (int64_t)supp * 4 + (int64_t)cust * 2 + (int64_t)year_offset;

            // Accumulate in registers for each group
            for (int g = 0; g < NUM_GROUPS; ++g) {
                if (key == g) {
                    my_revenue[g] = volume;
                    my_count[g] = 1;
                    break;
                }
            }
        }
    }

    // Store partial results in shared memory
    for (int g = 0; g < NUM_GROUPS; ++g) {
        block_revenue[g][threadIdx.x] = my_revenue[g];
        block_count[g][threadIdx.x] = my_count[g];
    }

    __syncthreads();

    // Block-level reduction for each group
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_revenue[g][threadIdx.x] += block_revenue[g][threadIdx.x + s];
                block_count[g][threadIdx.x] += block_count[g][threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (block_revenue[g][0] > 0.0 || block_count[g][0] > 0) {
                atomicAdd(&out_revenue[g], block_revenue[g][0]);
            }
            if (block_count[g][0] > 0) {
                atomicAdd(&out_count[g], block_count[g][0]);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_launch", [](torch::Tensor supp_nation_code,
                          torch::Tensor cust_nation_code,
                          torch::Tensor l_shipdate_days,
                          torch::Tensor l_extendedprice,
                          torch::Tensor l_discount,
                          int32_t shipdate_lo_days,
                          int32_t shipdate_hi_days) {
        TORCH_CHECK(supp_nation_code.scalar_type() == torch::kInt32);
        TORCH_CHECK(cust_nation_code.scalar_type() == torch::kInt32);
        TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32);
        TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
        TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64);

        if (!supp_nation_code.device().is_cuda()) supp_nation_code = supp_nation_code.cuda();
        if (!cust_nation_code.device().is_cuda()) cust_nation_code = cust_nation_code.cuda();
        if (!l_shipdate_days.device().is_cuda()) l_shipdate_days = l_shipdate_days.cuda();
        if (!l_extendedprice.device().is_cuda()) l_extendedprice = l_extendedprice.cuda();
        if (!l_discount.device().is_cuda()) l_discount = l_discount.cuda();

        supp_nation_code = supp_nation_code.contiguous();
        cust_nation_code = cust_nation_code.contiguous();
        l_shipdate_days = l_shipdate_days.contiguous();
        l_extendedprice = l_extendedprice.contiguous();
        l_discount = l_discount.contiguous();

        int64_t n = supp_nation_code.numel();
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto out_revenue = torch::zeros({NUM_GROUPS}, l_extendedprice.options().dtype(torch::kFloat64));
        auto out_count = torch::zeros({NUM_GROUPS}, torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)).options().dtype(torch::kInt32));

        q7_kernel<<<blocks, BLOCK_SIZE>>>(
            supp_nation_code.data_ptr<int32_t>(),
            cust_nation_code.data_ptr<int32_t>(),
            l_shipdate_days.data_ptr<int32_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            shipdate_lo_days,
            shipdate_hi_days,
            out_revenue.data_ptr<double>(),
            out_count.data_ptr<int32_t>(),
            n
        );

        return std::make_tuple(out_revenue, out_count);
    }, "TPC-H Q7 revenue per group (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q7_cuda_")
    cu_path = os.path.join(tmpdir, "_q7_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
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
    )

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
    Fused CUDA kernel combining shipdate filter, year extraction, group key computation,
    and revenue aggregation.
    - Filter rows by shipdate range
    - Calculate revenue = extendedprice * (1 - discount)
    - Extract year from int32 days since epoch
    - Compute composite group key (supp*4 + cust*2 + year_offset)
    - Aggregate revenue per group via block-level reduction + atomicAdd
    Returns grouped results for valid nation pairs across years.
    """
    out_revenue, out_count = _EXT.q7_launch(
        supp_nation_code, cust_nation_code, l_shipdate_days,
        l_extendedprice, l_discount,
        shipdate_lo_days, shipdate_hi_days,
    )

    # Map counts back to original logic
    group_ids = torch.arange(8, dtype=torch.int64, device="cuda")
    out_supp = (group_ids // 4).to(torch.int32)
    out_cust = ((group_ids // 2) % 2).to(torch.int32)
    out_year = (group_ids % 2).to(torch.int32) + 1995

    # Filter valid nation pairs (FRANCE-GERMANY or GERMANY-FRANCE)
    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (out_revenue > 0)

    return out_supp[valid], out_cust[valid], out_year[valid], out_revenue[valid]

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
    Params:
      nation_a (str): first nation name.
      nation_b (str): second nation name.
      shipdate_lo (str): inclusive lower bound for l_shipdate.
      shipdate_hi (str): inclusive upper bound for l_shipdate.
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
