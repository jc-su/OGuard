#include <cstdint>

#define MAX_PRIORITIES 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_priority_codes,
    const uint8_t* __restrict__ late_bitmap,
    int64_t* __restrict__ counts,
    int64_t n_orders,
    int64_t bitmap_size,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities
) {
    __shared__ unsigned long long shared_counts[MAX_PRIORITIES];

    for (int i = threadIdx.x; i < MAX_PRIORITIES; i += blockDim.x) {
        shared_counts[i] = 0ULL;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int32_t od = o_orderdate_days[idx];
        if (od >= date_lo_days && od < date_hi_days) {
            int64_t key = o_orderkey[idx];
            if (key >= 0 && key < bitmap_size && late_bitmap[key] != 0) {
                int32_t code = o_priority_codes[idx];
                if (code >= 0 && code < num_priorities) {
                    atomicAdd(&shared_counts[code], 1ULL);
                }
            }
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_priorities; i += blockDim.x) {
        unsigned long long v = shared_counts[i];
        if (v != 0ULL) {
            atomicAdd(reinterpret_cast<unsigned long long*>(counts) + i, v);
        }
    }
}
