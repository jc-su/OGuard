#include <cstdint>

__device__ inline uint64_t hash_pair(uint64_t p, uint64_t s) {
    uint64_t h = p ^ (s * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const int32_t* __restrict__ hash_table,
    double* __restrict__ ps_sum_qty,
    int64_t H, int64_t N_l) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_l) {
        uint64_t p = l_partkey[i];
        uint64_t s = l_suppkey[i];
        uint64_t h = hash_pair(p, s) & (H - 1);
        while (true) {
            int32_t idx = hash_table[h];
            if (idx == -1) break;
            if (ps_partkey[idx] == p && ps_suppkey[idx] == s) {
                atomicAdd(&ps_sum_qty[idx], l_quantity[i]);
                break;
            }
            h = (h + 1) & (H - 1);
        }
    }
}
