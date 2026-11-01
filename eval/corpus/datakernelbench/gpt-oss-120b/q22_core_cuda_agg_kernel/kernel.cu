#include <cstdint>

#define MAX_CODE   100   // country codes are two‑digit numbers 00‑99

extern "C" __global__ void kernel(
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
