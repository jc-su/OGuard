#include <stdint.h>

extern "C" __global__ void kernel(int32_t *group, int64_t *value, uint8_t *valid,
                                   int64_t *counts, int64_t *sums, int64_t n, int64_t g) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t byte_idx = i / 8;
        int bit_idx = i % 8;
        uint8_t valid_bit = (valid[byte_idx] >> bit_idx) & 1;

        if (valid_bit) {
            int32_t grp = group[i];
            atomicAdd((unsigned long long *)&counts[grp], 1ULL);
            atomicAdd((unsigned long long *)&sums[grp], (unsigned long long)value[i]);
        }
    }
}
