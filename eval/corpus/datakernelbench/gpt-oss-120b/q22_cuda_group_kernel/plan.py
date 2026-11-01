import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – all device kernels and C++ launchers are embedded here.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ---------------------------------------------------------------
// Kernel 1: sum balances and count rows where mask && bal>0.
// ---------------------------------------------------------------
__global__ void avg_kernel(
    const double* __restrict__ bal,
    const bool*   __restrict__ mask,
    double*       __restrict__ sum_out,
    unsigned long long* __restrict__ cnt_out,
    int64_t N)
{
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ unsigned long long s_cnt[BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;

    double my_sum = 0.0;
    unsigned long long my_cnt = 0ULL;

    if (idx < N) {
        bool keep = mask[idx] && bal[idx] > 0.0;
        if (keep) {
            my_sum = bal[idx];
            my_cnt = 1ULL;
        }
    }
    s_sum[tid] = my_sum;
    s_cnt[tid] = my_cnt;
    __syncthreads();

    // reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        __syncthreads();
    }

    // block‑level atomics
    if (tid == 0) {
        atomicAdd(sum_out, s_sum[0]);
        atomicAdd(cnt_out, s_cnt[0]);
    }
}

// ---------------------------------------------------------------
// Kernel 2: grouped aggregation after avg & NOT‑EXISTS filters.
// ---------------------------------------------------------------
__global__ void group_kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ bal,
    const int32_t* __restrict__ cntrycode,
    const bool*   __restrict__ in_mask,
    double avg_val,
    const int64_t* __restrict__ sorted_orders,
    int64_t orders_n,
    const int32_t* __restrict__ target_codes,
    int32_t n_codes,
    unsigned long long* __restrict__ grp_counts, // stored as uint64, later cast to int64
    double*            __restrict__ grp_sums,
    int64_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!in_mask[idx]) return;

    double bal_v = bal[idx];
    if (bal_v <= avg_val) return;

    // ----- NOT EXISTS test (binary search) -----
    int64_t key = custkey[idx];
    int64_t lo = 0, hi = orders_n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t mid_val = sorted_orders[mid];
        if (mid_val < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < orders_n && sorted_orders[lo] == key) return; // has order → discard

    // ----- map country code to group index (≤7 codes) -----
    int32_t code = cntrycode[idx];
    int32_t gidx = -1;
    for (int i = 0; i < n_codes; ++i) {
        if (target_codes[i] == code) {
            gidx = i;
            break;
        }
    }
    if (gidx < 0) return; // safety

    atomicAdd(&grp_counts[gidx], 1ULL);
    atomicAdd(&grp_sums[gidx], bal_v);
}

// ---------------------------------------------------------------
// Launcher for average balance – returns (sum, count) tensors.
// ---------------------------------------------------------------
std::vector<torch::Tensor> compute_avg(torch::Tensor bal, torch::Tensor mask) {
    TORCH_CHECK(bal.is_cuda() && mask.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(bal.scalar_type() == torch::kFloat64,
                "balance column must be float64");
    TORCH_CHECK(mask.scalar_type() == torch::kBool,
                "mask must be bool");

    auto sum = torch::zeros({1}, bal.options()); // double
    auto cnt = torch::zeros({1},
        torch::dtype(torch::kUInt64).device(bal.device())); // uint64

    int64_t N = bal.numel();
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    avg_kernel<<<blocks, BLOCK_SIZE>>>(
        bal.data_ptr<double>(),
        mask.data_ptr<bool>(),
        sum.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(cnt.data_ptr<uint64_t>()),
        N);
    cudaDeviceSynchronize();

    return {sum, cnt};
}

// ---------------------------------------------------------------
// Launcher for grouped aggregation – returns (counts, sums).
// ---------------------------------------------------------------
std::vector<torch::Tensor> group_by_cntry(
    torch::Tensor custkey,
    torch::Tensor bal,
    torch::Tensor cntrycode,
    torch::Tensor in_mask,
    double avg_val,
    torch::Tensor sorted_orders,
    torch::Tensor target_codes)
{
    TORCH_CHECK(custkey.is_cuda() && bal.is_cuda() && cntrycode.is_cuda()
                && in_mask.is_cuda() && sorted_orders.is_cuda()
                && target_codes.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(bal.scalar_type() == torch::kFloat64,
                "balance column must be float64");
    TORCH_CHECK(cntrycode.scalar_type() == torch::kInt32,
                "cntrycode must be int32");
    TORCH_CHECK(custkey.scalar_type() == torch::kInt64,
                "custkey must be int64");
    TORCH_CHECK(in_mask.scalar_type() == torch::kBool,
                "mask must be bool");

    int32_t n_codes = target_codes.numel();

    // counts as int64 (Long) – internally we treat them as uint64 for atomicAdd.
    auto counts = torch::zeros({n_codes},
        torch::dtype(torch::kInt64).device(bal.device()));
    auto sums   = torch::zeros({n_codes}, bal.options()); // float64

    int64_t N = custkey.numel();
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    group_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        bal.data_ptr<double>(),
        cntrycode.data_ptr<int32_t>(),
        in_mask.data_ptr<bool>(),
        avg_val,
        sorted_orders.data_ptr<int64_t>(),
        sorted_orders.numel(),
        target_codes.data_ptr<int32_t>(),
        n_codes,
        reinterpret_cast<unsigned long long*>(counts.data_ptr<int64_t>()), // int64 ↔ uint64 alias
        sums.data_ptr<double>(),
        N);
    cudaDeviceSynchronize();

    return {counts, sums};
}

// ---------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_avg", &compute_avg,
          "Return (sum, count) tensors for average balance.");
    m.def("group_by_cntry", &group_by_cntry,
          "Group‑by cntrycode after applying avg and NOT‑EXISTS filters.");
}
'''

# ----------------------------------------------------------------------
# Compile/load the CUDA extension (once per interpreter session).
# ----------------------------------------------------------------------
def _load_cuda():
    """Write the embedded CUDA source to a temporary file and compile it."""
    tmpdir = tempfile.mkdtemp(prefix="q22_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
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
        verbose=False,
    )

_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Core GPU‑accelerated query implementation.
# ----------------------------------------------------------------------
def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
    target_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU‑fused implementation of TPC‑H Q22 core logic.

    Steps performed on the GPU:
      1️⃣ Compute average balance for rows where mask && balance>0.
      2️⃣ Sort unique order custkeys for binary‑search NOT‑EXISTS test.
      3️⃣ Execute a fused kernel that:
         • filters by average balance,
         • checks NOT‑EXISTS via binary search,
         • atomically accumulates per‑country count (int64) and sum (float64).
    Returns (codes, counts, sums) tensors, where `counts` is int64.
    """
    # 1️⃣ Average balance
    sum_cnt = _EXT.compute_avg(c_acctbal, in_mask)      # [sum, cnt]
    sum_bal = sum_cnt[0]
    cnt_bal = sum_cnt[1]
    avg_bal = sum_bal.item() / max(cnt_bal.item(), 1)

    # 2️⃣ Sort order custkeys
    sorted_orders, _ = torch.sort(o_custkey_unique)

    # 3️⃣ Group‑by aggregation
    grp_counts, grp_sums = _EXT.group_by_cntry(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_mask,
        avg_bal,
        sorted_orders,
        target_codes,
    )
    # grp_counts already int64
    return target_codes, grp_counts, grp_sums

# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17')
) -> cudf.DataFrame:
    """
    SQL:
        SELECT cntrycode,
               count(*) AS numcust,
               sum(c_acctbal) AS totacctbal
        FROM (
            SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode,
                   c_acctbal
            FROM customer
            WHERE substring(c_phone FROM 1 FOR 2) IN (...)
              AND c_acctbal > (SELECT avg(c_acctbal)
                               FROM customer
                               WHERE c_acctbal > 0.00
                                 AND substring(c_phone FROM 1 FOR 2) IN (...))
              AND NOT EXISTS (SELECT *
                               FROM orders
                               WHERE o_custkey = c_custkey)
        ) AS custsale
        GROUP BY cntrycode
        ORDER BY cntrycode;
    Description:
        The original multi‑step cuDF pipeline is replaced by two fused
        CUDA kernels: one computes the required average balance, the
        second performs the full filter (including the NOT‑EXISTS
        binary search) and aggregates per country code.  This reduces
        HBM traffic dramatically on an H100, keeping the kernels
        memory‑bound and achieving high throughput.
    Params:
        tables (dict[str, cudf.DataFrame]): mapping of table name to DataFrame.
        cntrycodes (tuple[str, ...]): target 2‑digit country codes.
    Returns:
        cudf.DataFrame with columns ['cntrycode', 'numcust', 'totacctbal'].
    """
    # ------------------------------------------------------------------
    # 1️⃣ Load required columns from cuDF and move to GPU.
    # ------------------------------------------------------------------
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Extract first two characters of phone as numeric code.
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    # Boolean mask for target country codes.
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    # Transfer columns to CUDA tensors.
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(),
                                  dtype=torch.int64, device="cuda")
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(),
                                  dtype=torch.float64, device="cuda")
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(),
                                  dtype=torch.int32, device="cuda")
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(),
                                dtype=torch.bool, device="cuda")

    # Unique order custkeys for NOT‑EXISTS test.
    o_custkey_unique_series = df_orders["o_custkey"].unique()
    o_custkey_unique_t = torch.as_tensor(o_custkey_unique_series.to_cupy(),
                                         dtype=torch.int64, device="cuda")

    # ------------------------------------------------------------------
    # 2️⃣ Sorted list of target country codes (ascending order for output).
    # ------------------------------------------------------------------
    target_codes_sorted = sorted(codes_int)
    target_codes_t = torch.tensor(target_codes_sorted,
                                  dtype=torch.int32, device="cuda")

    # ------------------------------------------------------------------
    # 3️⃣ Execute fused GPU kernels.
    # ------------------------------------------------------------------
    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t,
        c_acctbal_t,
        cntrycode_t,
        in_mask_t,
        o_custkey_unique_t,
        target_codes_t,
    )

    # ------------------------------------------------------------------
    # 4️⃣ Convert results back to cuDF (CPU side).
    # ------------------------------------------------------------------
    codes_cudf = cudf.Series(codes_t.cpu()).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust":   cudf.Series(numcust_t.cpu()),
        "totacctbal": cudf.Series(totacctbal_t.cpu()),
    })
