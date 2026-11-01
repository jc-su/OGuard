#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ acctbal_ptr,
    const int32_t* __restrict__ cntrycode_ptr,
    const int32_t* __restrict__ in_cntrycode_mask_ptr,
    const int64_t* __restrict__ sorted_okeys_ptr,
    int64_t n_okeys,
    double avg_val,
    int32_t* __restrict__ out_cntrycodes,
    double* __restrict__ out_acctbals,
    int32_t* __restrict__ out_count,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n) return;

    if (!in_cntrycode_mask_ptr[idx]) return;

    double bal = acctbal_ptr[idx];
    if (!(bal > avg_val)) return;

    int64_t ck = custkey_ptr[idx];

    // Binary search sorted_okeys
    int64_t lo = 0, hi = n_okeys;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (sorted_okeys_ptr[mid] < ck) lo = mid + 1;
        else hi = mid;
    }

    if (lo < n_okeys && sorted_okeys_ptr[lo] == ck) return;

    // Atomically write to output buffer
    int32_t pos = atomicAdd(out_count, 1);
    if (pos < n) {
        out_cntrycodes[pos] = cntrycode_ptr[idx];
        out_acctbals[pos] = bal;
    }
}
