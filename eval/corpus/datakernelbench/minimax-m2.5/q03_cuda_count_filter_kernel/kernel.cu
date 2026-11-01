#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ l_shipdate_days,
    const int32_t* __restrict__ o_orderdate_days,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    unsigned long long* __restrict__ out_count,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t lsd = l_shipdate_days[idx];
        int32_t ood = o_orderdate_days[idx];
        if (lsd > shipdate_lo_days && ood < orderdate_hi_days) {
            atomicAdd(out_count, 1ULL);
        }
    }
}
