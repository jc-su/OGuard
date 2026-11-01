#include <stdint.h>

#define EMPTY_KEY 0xffffffffffffffffULL

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    unsigned long long* __restrict__ hash_table,
    uint64_t hash_mask,
    int64_t* __restrict__ counts,
    int64_t num_groups
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int64_t gid_i64 = group_ids[idx];
    int64_t sid_i64 = supp_ids[idx];

    if (gid_i64 < 0 || gid_i64 >= num_groups) {
        return;
    }

    uint64_t gid = static_cast<uint64_t>(gid_i64);
    uint64_t sid = static_cast<uint64_t>(static_cast<uint32_t>(sid_i64));
    uint64_t key = (gid << 32) | sid;

    if (key == EMPTY_KEY) {
        return;
    }

    uint64_t pos = mix64(key) & hash_mask;

    while (true) {
        unsigned long long old = atomicCAS(
            &hash_table[pos],
            static_cast<unsigned long long>(EMPTY_KEY),
            static_cast<unsigned long long>(key)
        );

        if (old == EMPTY_KEY) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(counts + gid_i64),
                static_cast<unsigned long long>(1)
            );
            return;
        }

        if (old == key) {
            return;
        }

        pos = (pos + 1ULL) & hash_mask;
    }
}
