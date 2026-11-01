#include <cstdint>

__device__ inline uint64_t hash_single(uint64_t s) {
    uint64_t h = s;
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const unsigned long long int* __restrict__ supp_hash_set,
    int64_t* __restrict__ out_idx,
    int32_t* __restrict__ out_count,
    int64_t H_s, int64_t N_s) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_s) {
        uint64_t s = s_suppkey[i];
        uint64_t h = hash_single(s) & (H_s - 1);
        bool found = false;
        while (true) {
            unsigned long long int val = supp_hash_set[h];
            if (val == (unsigned long long int)-1) break;
            if (val == (unsigned long long int)s) {
                found = true;
                break;
            }
            h = (h + 1) & (H_s - 1);
        }
        if (found) {
            int32_t pos = atomicAdd(out_count, 1);
            out_idx[pos] = s_row_idx[i];
        }
    }
}
