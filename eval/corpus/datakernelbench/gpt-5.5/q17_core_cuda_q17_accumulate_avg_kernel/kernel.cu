#include <stdint.h>
#include <limits.h>

#define EMPTY_KEY LLONG_MIN

__device__ __forceinline__ uint64_t q17_hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int64_t q17_lookup_key(
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_size,
    int64_t key
) {
    uint64_t mask = static_cast<uint64_t>(hash_size - 1);
    uint64_t pos = q17_hash_u64(static_cast<uint64_t>(key)) & mask;

    #pragma unroll 1
    for (int64_t probe = 0; probe < hash_size; ++probe) {
        int64_t found = hash_keys[pos];
        if (found == key) {
            return hash_vals[pos];
        }
        if (found == EMPTY_KEY) {
            return -1;
        }
        pos = (pos + 1ULL) & mask;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_line,
    int64_t hash_size
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t row = tid; row < n_line; row += stride) {
        int64_t part_idx = q17_lookup_key(hash_keys, hash_vals, hash_size, l_partkey[row]);
        if (part_idx >= 0) {
            atomicAdd(&sum_qty[part_idx], l_quantity[row]);
            atomicAdd(&cnt_qty[part_idx], 1.0);
        }
    }
}
