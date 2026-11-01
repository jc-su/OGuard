#include <cstdint>

extern "C" __global__ void kernel(
    const uint8_t* valid,
    int64_t* count,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t byte_idx = idx / 8;
        int32_t bit_idx = idx % 8;
        if ((valid[byte_idx] & (1 << bit_idx)) == 0) {
            atomicAdd((unsigned long long*)count, 1ULL);
        }
    }
}
