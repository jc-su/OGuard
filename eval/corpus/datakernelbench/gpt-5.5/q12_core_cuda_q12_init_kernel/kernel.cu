#include <cstdint>

extern "C" __global__ void kernel(
    int64_t* __restrict__ group_ids,
    int64_t* __restrict__ high_counts,
    int64_t* __restrict__ low_counts
) {
    if (threadIdx.x == 0) {
        group_ids[0] = 0;
        group_ids[1] = 1;
        high_counts[0] = 0;
        high_counts[1] = 0;
        low_counts[0] = 0;
        low_counts[1] = 0;
    }
}
