#include <cstdint>

#define EMPTY_KEY 0xffffffffffffffffULL

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t make_pair_key_i64(int64_t partkey, int64_t suppkey, int64_t key_base) {
    return ((uint64_t)partkey) * ((uint64_t)key_base) + ((uint64_t)suppkey);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    uint64_t* __restrict__ table_keys,
    int64_t* __restrict__ table_vals,
    int64_t nps,
    int64_t key_base,
    int64_t table_mask
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nps) {
        return;
    }

    uint64_t key = make_pair_key_i64(ps_partkey[idx], ps_suppkey[idx], key_base);
    uint64_t h = mix64(key) & (uint64_t)table_mask;

    while (true) {
        uint64_t old = atomicCAS(
            (unsigned long long*)&table_keys[h],
            (unsigned long long)EMPTY_KEY,
            (unsigned long long)key
        );
        if (old == EMPTY_KEY || old == key) {
            table_vals[h] = idx;
            return;
        }
        h = (h + 1ULL) & (uint64_t)table_mask;
    }
}
