#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate_days,
    const int32_t* __restrict__ o_orderpriority_codes,
    const uint8_t* __restrict__ has_late_lineitem,
    int64_t* __restrict__ counts,
    int64_t n_orders,
    int64_t max_key,
    int32_t date_lo_days,
    int32_t date_hi_days,
    int32_t num_priorities
) {
    extern __shared__ unsigned long long s_counts[];

    for (int32_t p = threadIdx.x; p < num_priorities; p += blockDim.x) {
        s_counts[p] = 0ULL;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_orders) {
        int32_t od = o_orderdate_days[idx];
        if (od >= date_lo_days && od < date_hi_days) {
            int64_t key = o_orderkey[idx];
            if (key >= 0 && key <= max_key && has_late_lineitem[key] != 0) {
                int32_t prio = o_orderpriority_codes[idx];
                if (prio >= 0 && prio < num_priorities) {
                    atomicAdd(&s_counts[prio], 1ULL);
                }
            }
        }
    }

    __syncthreads();

    for (int32_t p = threadIdx.x; p < num_priorities; p += blockDim.x) {
        unsigned long long v = s_counts[p];
        if (v != 0ULL) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(counts + p),
                v
            );
        }
    }
}
