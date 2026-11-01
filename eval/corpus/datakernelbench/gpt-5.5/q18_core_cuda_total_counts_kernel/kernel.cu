#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ unique_offsets,
    const int64_t* __restrict__ unique_counts,
    const int64_t* __restrict__ big_offsets,
    const int64_t* __restrict__ big_counts,
    int64_t n_blocks,
    int64_t* __restrict__ totals
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t last = n_blocks - 1;
        totals[0] = unique_offsets[last] + unique_counts[last];
        totals[1] = big_offsets[last] + big_counts[last];
    }
}
