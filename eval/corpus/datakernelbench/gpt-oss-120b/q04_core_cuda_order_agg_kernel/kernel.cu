#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate,
    const int32_t* __restrict__ o_priority,
    const int32_t* __restrict__ flags,
    int32_t date_lo,
    int32_t date_hi,
    int64_t* __restrict__ counts,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t d = o_orderdate[idx];
    if (d >= date_lo && d < date_hi) {
        int64_t key = o_orderkey[idx];
        if (flags[key] != 0) {
            int32_t p = o_priority[idx];
            // atomicAdd on int64_t via unsigned long long cast (supported on sm_90)
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts[p]), 1ULL);
        }
    }
}
