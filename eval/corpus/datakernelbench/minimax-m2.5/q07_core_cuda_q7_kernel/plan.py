import os
import tempfile
import torch
from torch.utils.cpp_extension import load
import cudf
import pandas as pd

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q7_kernel(
    const int32_t* __restrict__ supp,
    const int32_t* __restrict__ cust,
    const int32_t* __restrict__ shipdate,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ revenue,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t sd = shipdate[idx];
    if (sd < date_lo || sd > date_hi) return;

    // Compute year from epoch days (civil_from_days algorithm)
    int64_t days = (int64_t)sd + 719468;
    int64_t era = (days >= 0 ? days : days - 146096) / 146097;
    int64_t doe = days - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp = (5 * doy + 2) / 153;
    int64_t m = (mp < 10) ? mp + 3 : mp - 9;
    int64_t year = (m <= 2) ? y + 1 : y;

    int year_offset = (int)(year - 1995);
    if (year_offset < 0 || year_offset > 1) return;

    int s = supp[idx];
    int c = cust[idx];
    int group = s * 4 + c * 2 + year_offset;
    double vol = price[idx] * (1.0 - discount[idx]);
    atomicAdd(&revenue[group], vol);
}

torch::Tensor q7_launcher(
    torch::Tensor supp,
    torch::Tensor cust,
    torch::Tensor shipdate,
    torch::Tensor price,
    torch::Tensor discount,
    int32_t date_lo,
    int32_t date_hi)
{
    TORCH_CHECK(supp.scalar_type() == torch::kInt32, "supp must be int32");
    TORCH_CHECK(cust.scalar_type() == torch::kInt32, "cust must be int32");
    TORCH_CHECK(shipdate.scalar_type() == torch::kInt32, "shipdate must be int32");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");

    if (!supp.is_cuda()) supp = supp.cuda();
    if (!cust.is_cuda()) cust = cust.cuda();
    if (!shipdate.is_cuda()) shipdate = shipdate.cuda();
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();

    supp = supp.contiguous();
    cust = cust.contiguous();
    shipdate = shipdate.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();

    int64_t n = price.numel();
    auto revenue = torch::zeros({8}, price.options());

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q7_kernel<<<blocks, BLOCK_SIZE>>>(
        supp.data_ptr<int32_t>(),
        cust.data_ptr<int32_t>(),
        shipdate.data_ptr<int32_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        date_lo,
        date_hi,
        revenue.data_ptr<double>(),
        n);
    return revenue;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q7_launcher", &q7_launcher, "TPC-H Q7 revenue grouping (CUDA)");
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


def _launch_q7(
    supp: torch.Tensor,
    cust: torch.Tensor,
    shipdate: torch.Tensor,
    price: torch.Tensor,
    discount: torch.Tensor,
    date_lo: int,
    date_hi: int,
) -> torch.Tensor:
    # Ensure tensors are on CUDA
    if not supp.is_cuda:
        supp = supp.cuda()
    if not cust.is_cuda:
        cust = cust.cuda()
    if not shipdate.is_cuda:
        shipdate = shipdate.cuda()
    if not price.is_cuda:
        price = price.cuda()
    if not discount.is_cuda:
        discount = discount.cuda()

    # Make contiguous
    supp = supp.contiguous()
    cust = cust.contiguous()
    shipdate = shipdate.contiguous()
    price = price.contiguous()
    discount = discount.contiguous()

    # Invoke CUDA kernel
    revenue = _EXT.q7_launcher(
        supp, cust, shipdate, price, discount, date_lo, date_hi
    )
    return revenue


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
    Core GPU computation: filter lineitem rows by shipdate,
    compute volume, group by supp/cust/year, sum revenue.
    Returns four 1-D tensors: supp_nation, cust_nation, l_year, revenue.
    """
    revenue = _launch_q7(
        supp_nation_code,
        cust_nation_code,
        l_shipdate_days,
        l_extendedprice,
        l_discount,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    # Build group keys (0..7) and filter valid combos
    group_ids = torch.arange(8, dtype=torch.int64, device=revenue.device)
    out_supp = (group_ids // 4).to(torch.int32)
    out_cust = ((group_ids // 2) % 2).to(torch.int32)
    out_year = (group_ids % 2).to(torch.int32) + 1995

    valid = ((out_supp == 0) & (out_cust == 1)) | ((out_supp == 1) & (out_cust == 0))
    valid = valid & (revenue > 0)

    return out_supp[valid], out_cust[valid], out_year[valid], revenue[valid]


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
