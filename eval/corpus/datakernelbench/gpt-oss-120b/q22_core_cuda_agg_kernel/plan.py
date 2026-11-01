import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source (device kernels + C++ launchers)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define MAX_CODE   100   // country codes are two‑digit numbers 00‑99

// ------------------------------------------------------------------
// Kernel 1: compute sum and count of balances for average
// ------------------------------------------------------------------
__global__ void avg_kernel(
    const double* __restrict__ acctbal,
    const bool*   __restrict__ mask,
    double*       __restrict__ out_sum,
    unsigned long long* __restrict__ out_cnt,
    int64_t n)
{
    double my_sum = 0.0;
    unsigned long long my_cnt = 0ULL;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (mask[idx] && acctbal[idx] > 0.0) {
            my_sum = acctbal[idx];
            my_cnt = 1ULL;
        }
    }

    // shared‑memory reduction
    extern __shared__ char shm[];
    double* s_sum = (double*)shm;
    unsigned long long* s_cnt = (unsigned long long*)(s_sum + BLOCK_SIZE);
    s_sum[threadIdx.x] = my_sum;
    s_cnt[threadIdx.x] = my_cnt;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_sum, s_sum[0]);
        atomicAdd(out_cnt, s_cnt[0]);
    }
}

// ------------------------------------------------------------------
// Kernel 2: final aggregation per country code
// ------------------------------------------------------------------
__global__ void agg_kernel(
    const int64_t* __restrict__ custkey,
    const double*  __restrict__ acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool*    __restrict__ mask,
    const int64_t* __restrict__ orders,
    int64_t n_orders,
    double avg,
    unsigned long long* __restrict__ out_cnt,
    double*        __restrict__ out_sum,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (!mask[idx]) return;

    double bal = acctbal[idx];
    if (bal <= avg) return;

    int64_t key = custkey[idx];

    // binary search in sorted orders array
    int64_t lo = 0, hi = n_orders - 1;
    bool has = false;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t v = orders[mid];
        if (v == key) { has = true; break; }
        if (v < key) lo = mid + 1;
        else hi = mid - 1;
    }
    if (has) return;

    int32_t code = cntrycode[idx];
    if (code < 0 || code >= MAX_CODE) return;

    atomicAdd(&out_cnt[code], 1ULL);
    atomicAdd(&out_sum[code], bal);
}

// ------------------------------------------------------------------
// Launcher for avg_kernel
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> avg_launcher(
    torch::Tensor acctbal,
    torch::Tensor mask)
{
    TORCH_CHECK(acctbal.is_cuda(), "acctbal must be CUDA");
    TORCH_CHECK(mask.is_cuda(),   "mask must be CUDA");
    TORCH_CHECK(acctbal.dtype() == torch::kFloat64, "acctbal dtype must be float64");
    TORCH_CHECK(mask.dtype() == torch::kBool,       "mask dtype must be bool");

    int64_t n = acctbal.numel();
    auto opts_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(acctbal.device());
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(acctbal.device());

    torch::Tensor sum = torch::zeros({1}, opts_f64);
    torch::Tensor cnt = torch::zeros({1}, opts_i64);   // will be used as uint64 in kernel

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem = BLOCK_SIZE * (sizeof(double) + sizeof(unsigned long long));

    avg_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        acctbal.data_ptr<double>(),
        mask.data_ptr<bool>(),
        sum.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()),
        n
    );
    cudaDeviceSynchronize();
    return std::make_tuple(sum, cnt);
}

// ------------------------------------------------------------------
// Launcher for agg_kernel
// ------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> agg_launcher(
    torch::Tensor custkey,
    torch::Tensor acctbal,
    torch::Tensor cntrycode,
    torch::Tensor mask,
    torch::Tensor orders_unique,
    double avg)
{
    TORCH_CHECK(custkey.is_cuda(),   "custkey must be CUDA");
    TORCH_CHECK(acctbal.is_cuda(),   "acctbal must be CUDA");
    TORCH_CHECK(cntrycode.is_cuda(),"cntrycode must be CUDA");
    TORCH_CHECK(mask.is_cuda(),      "mask must be CUDA");
    TORCH_CHECK(orders_unique.is_cuda(),"orders_unique must be CUDA");

    TORCH_CHECK(custkey.dtype() == torch::kInt64,   "custkey dtype must be int64");
    TORCH_CHECK(acctbal.dtype() == torch::kFloat64, "acctbal dtype must be float64");
    TORCH_CHECK(cntrycode.dtype() == torch::kInt32, "cntrycode dtype must be int32");
    TORCH_CHECK(mask.dtype() == torch::kBool,       "mask dtype must be bool");
    TORCH_CHECK(orders_unique.dtype() == torch::kInt64, "orders_unique dtype must be int64");

    int64_t n = custkey.numel();
    int64_t n_orders = orders_unique.numel();

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(custkey.device());
    auto opts_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(custkey.device());

    torch::Tensor out_cnt = torch::zeros({MAX_CODE}, opts_i64);   // used as uint64 internally
    torch::Tensor out_sum = torch::zeros({MAX_CODE}, opts_f64);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    agg_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        acctbal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        mask.data_ptr<bool>(),
        orders_unique.data_ptr<int64_t>(),
        n_orders,
        avg,
        reinterpret_cast<unsigned long long*>(out_cnt.data_ptr<int64_t>()),
        out_sum.data_ptr<double>(),
        n
    );
    cudaDeviceSynchronize();
    return std::make_tuple(out_cnt, out_sum);
}

// ------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_launcher", &avg_launcher, "Compute sum and count for avg (CUDA)");
    m.def("agg_launcher", &agg_launcher, "Final aggregation per country code (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Compile / load the CUDA extension (once per import)
# ----------------------------------------------------------------------
def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q22_cuda_")
    cu_path = os.path.join(tmpdir, "_q22_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q22_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()
_MAX_CODE = 100  # country codes 00‑99 inclusive

# ----------------------------------------------------------------------
# Private helper: compute average balance (GPU reduction)
# ----------------------------------------------------------------------
def _compute_avg(c_acctbal: torch.Tensor, in_mask: torch.Tensor) -> float:
    sum_t, cnt_t = _EXT.avg_launcher(c_acctbal, in_mask)
    total = sum_t.item()
    count = cnt_t.item()
    return total / count if count != 0 else 0.0

# ----------------------------------------------------------------------
# Private helper: run final aggregation kernel
# ----------------------------------------------------------------------
def _run_aggregation(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
    avg_bal: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cnt_arr, sum_arr = _EXT.agg_launcher(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_mask,
        o_custkey_unique,
        avg_bal,
    )
    mask = cnt_arr > 0
    codes = torch.arange(_MAX_CODE, dtype=torch.int32, device=cnt_arr.device)[mask]
    numcust = cnt_arr[mask]
    totacctbal = sum_arr[mask]
    return codes, numcust, totacctbal

# ----------------------------------------------------------------------
# Public query core (signature fixed by level constraints)
# ----------------------------------------------------------------------
def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fuse the original eager‑torch logic into two CUDA kernels:
      1. Compute avg(c_acctbal) over customers belonging to the target
         country‑code set and having a positive balance.
      2. Scan customers once more, apply the avg threshold and the NOT‑EXISTS
         test, and atomically accumulate count and balance per country code.
    This reduces memory traffic from several separate PyTorch passes to two
    tightly‑coalesced GPU kernels, yielding a speedup on H100 for this
    memory‑bound query.
    """
    # ensure contiguous device tensors
    c_custkey = c_custkey.contiguous()
    c_acctbal = c_acctbal.contiguous()
    cntrycode = cntrycode.contiguous()
    in_cntrycode_mask = in_cntrycode_mask.contiguous()
    o_custkey_unique = o_custkey_unique.contiguous()

    # Step 1 – average balance
    avg_bal = _compute_avg(c_acctbal, in_cntrycode_mask)

    # Step 2 – final aggregation per country code
    codes, numcust, totacctbal = _run_aggregation(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        o_custkey_unique,
        avg_bal,
    )
    return codes, numcust, totacctbal

# ----------------------------------------------------------------------
# Run query (verbatim from reference, unchanged)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
) -> cudf.DataFrame:
    """
    SQL: SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
         FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
               FROM customer
               WHERE substring(c_phone FROM 1 FOR 2) IN (...)
                 AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                                  WHERE c_acctbal > 0.00
                                    AND substring(c_phone FROM 1 FOR 2) IN (...))
                 AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
              ) AS custsale
         GROUP BY cntrycode ORDER BY cntrycode;
    Description: TPC‑H Q22 - Global sales opportunity query. Identifies customers in
                 target country codes with above‑average balances who have placed no orders,
                 grouped and aggregated by country code.
    Params: cntrycodes (tuple[str,...]): target 2‑digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Derive cntrycode: first 2 chars of c_phone, convert to int (codes are numeric strings)
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    # Build in‑set mask on GPU
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    # Unique o_custkey from orders (for NOT EXISTS), sorted for binary search
    o_custkey_unique_series = df_orders["o_custkey"].unique().sort_values()

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    # Format cntrycode back to 2‑char zero‑padded string
    codes_cudf = cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t),
        "totacctbal": cudf.Series(totacctbal_t),
    })
