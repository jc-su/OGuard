#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ bal,
    const int32_t* __restrict__ cntrycode,
    const bool*   __restrict__ in_mask,
    double avg_val,
    const int64_t* __restrict__ sorted_orders,
    int64_t orders_n,
    const int32_t* __restrict__ target_codes,
    int32_t n_codes,
    unsigned long long* __restrict__ grp_counts,
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
