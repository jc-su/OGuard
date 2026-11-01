#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_date,
    const int32_t* __restrict__ o_priority,
    const uint32_t* __restrict__ bitmap,
    unsigned long long* __restrict__ counts,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int64_t key = o_orderkey[idx];
        if (key > 0 && bitmap[key]) {
            int32_t d = o_date[idx];
            if (d >= date_lo && d < date_hi) {
                int32_t p = o_priority[idx];
                atomicAdd(&counts[p], (unsigned long long)1);
            }
        }
    }
}
