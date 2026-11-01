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
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ sums,
    int64_t* __restrict__ qual_supp_set,
    int64_t nps,
    int64_t supp_cap
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nps) return;

    double sum_q = sums[i];
    if (!(sum_q > 0.0)) return;

    double avail = (double)ps_availqty[i];
    if (!(avail > 0.5 * sum_q)) return;

    int64_t s = ps_suppkey[i];
    uint64_t mask = (uint64_t)(supp_cap - 1);
    uint64_t pos = hash_i64(s) & mask;

    for (int64_t probe = 0; probe < supp_cap; ++probe) {
        int64_t old = (int64_t)atomicCAS(
            (unsigned long long*)&qual_supp_set[pos],
            0xffffffffffffffffULL,
            (unsigned long long)s
        );
        if (old == -1 || old == s) {
            return;
        }
        pos = (pos + 1) & mask;
    }
}
