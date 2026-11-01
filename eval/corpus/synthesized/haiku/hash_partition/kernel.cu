#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* keys,
    int32_t* part,
    int32_t* counts,
    int64_t n,
    int64_t p
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        // Hash: (x * 0x9E3779B97F4A7C15) >> 32 as unsigned
        uint64_t h = ((uint64_t)key * (uint64_t)0x9E3779B97F4A7C15) >> 32;
        int32_t partition = (int32_t)(h % p);
        part[idx] = partition;
        atomicAdd(&counts[partition], 1);
    }
}
