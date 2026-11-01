#include <stdint.h>

__device__ __forceinline__ uint64_t mix_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t hash_pair_i64(int64_t a, int64_t b) {
    uint64_t x = mix_u64((uint64_t)a);
    uint64_t y = mix_u64((uint64_t)b + 0x9e3779b97f4a7c15ULL);
    return x ^ (y + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ ht_part,
    const int64_t* __restrict__ ht_supp,
    const int64_t* __restrict__ ht_idx,
    double* __restrict__ sums,
    int64_t nli,
    int64_t cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nli) return;

    int64_t p = l_partkey[i];
    int64_t s = l_suppkey[i];
    uint64_t mask = (uint64_t)(cap - 1);
    uint64_t pos = hash_pair_i64(p, s) & mask;

    for (int64_t probe = 0; probe < cap; ++probe) {
        int64_t slot_idx = ht_idx[pos];
        if (slot_idx == -1) {
            return;
        }
        if (ht_part[pos] == p && ht_supp[pos] == s) {
            double q = l_quantity[i];
            atomicAdd(&sums[slot_idx], q);
            return;
        }
        pos = (pos + 1) & mask;
    }
}
