#include <stdint.h>

__device__ __forceinline__ uint64_t mix_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t hash_i64(int64_t a) {
    return mix_u64((uint64_t)a);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const int64_t* __restrict__ qual_supp_set,
    int64_t* __restrict__ out_idx,
    int64_t* __restrict__ out_count,
    int64_t ns,
    int64_t supp_cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns) return;

    int64_t s = s_suppkey[i];
    uint64_t mask = (uint64_t)(supp_cap - 1);
    uint64_t pos = hash_i64(s) & mask;

    for (int64_t probe = 0; probe < supp_cap; ++probe) {
        int64_t v = qual_supp_set[pos];
        if (v == -1) {
            return;
        }
        if (v == s) {
            unsigned long long write_pos = atomicAdd(
                (unsigned long long*)out_count,
                1ULL
            );
            out_idx[(int64_t)write_pos] = s_row_idx[i];
            return;
        }
        pos = (pos + 1) & mask;
    }
}
