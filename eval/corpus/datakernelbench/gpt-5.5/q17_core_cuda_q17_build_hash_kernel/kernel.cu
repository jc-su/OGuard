#include <stdint.h>
#include <limits.h>

#define EMPTY_KEY LLONG_MIN

__device__ __forceinline__ uint64_t q17_hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ part_keys,
    int64_t* __restrict__ hash_keys,
    int64_t* __restrict__ hash_vals,
    int64_t n_parts,
    int64_t hash_size
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    uint64_t mask = static_cast<uint64_t>(hash_size - 1);
    const unsigned long long empty = static_cast<unsigned long long>(EMPTY_KEY);

    for (int64_t idx = tid; idx < n_parts; idx += stride) {
        int64_t key = part_keys[idx];
        uint64_t pos = q17_hash_u64(static_cast<uint64_t>(key)) & mask;

        #pragma unroll 1
        while (true) {
            unsigned long long old = atomicCAS(
                reinterpret_cast<unsigned long long*>(&hash_keys[pos]),
                empty,
                static_cast<unsigned long long>(key)
            );
            int64_t old_signed = static_cast<int64_t>(old);
            if (old_signed == EMPTY_KEY || old_signed == key) {
                hash_vals[pos] = idx;
                break;
            }
            pos = (pos + 1ULL) & mask;
        }
    }
}
