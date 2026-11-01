#include <stdint.h>

#define EMPTY_KEY (-1LL)

__device__ __forceinline__ uint64_t q15_hash_i64(int64_t key) {
    uint64_t x = static_cast<uint64_t>(key);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    int64_t* __restrict__ hash_keys,
    int64_t* __restrict__ hash_vals,
    int64_t n_sup,
    int64_t hash_mask
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_sup) {
        return;
    }

    int64_t key = s_suppkey[idx];
    uint64_t h = q15_hash_i64(key);
    int64_t slot = static_cast<int64_t>(h & static_cast<uint64_t>(hash_mask));

    #pragma unroll 1
    while (true) {
        unsigned long long* slot_ptr =
            reinterpret_cast<unsigned long long*>(&hash_keys[slot]);
        unsigned long long old = atomicCAS(
            slot_ptr,
            static_cast<unsigned long long>(EMPTY_KEY),
            static_cast<unsigned long long>(key)
        );
        int64_t old_i = static_cast<int64_t>(old);
        if (old_i == EMPTY_KEY || old_i == key) {
            hash_vals[slot] = idx;
            return;
        }
        slot = (slot + 1) & hash_mask;
    }
}
