#include <cstdint>

__device__ inline uint64_t hash_pair(uint64_t p, uint64_t s) {
    uint64_t h = p ^ (s * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    int32_t* __restrict__ hash_table,
    int64_t H, int64_t N_ps) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ps) {
        uint64_t p = ps_partkey[i];
        uint64_t s = ps_suppkey[i];
        uint64_t h = hash_pair(p, s) & (H - 1);
        while (true) {
            int32_t old = atomicCAS(&hash_table[h], -1, (int32_t)i);
            if (old == -1 || old == (int32_t)i) break;
            h = (h + 1) & (H - 1);
        }
    }
}
