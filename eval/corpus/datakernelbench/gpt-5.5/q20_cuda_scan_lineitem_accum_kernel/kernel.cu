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
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ l_shipdate_days,
    const uint64_t* __restrict__ table_keys,
    const int64_t* __restrict__ table_vals,
    double* __restrict__ ps_sum_qty,
    int64_t nli,
    int64_t key_base,
    int64_t table_mask,
    int32_t date_lo,
    int32_t date_hi
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nli) {
        return;
    }

    int32_t ship_day = l_shipdate_days[idx];
    if (ship_day < date_lo || ship_day >= date_hi) {
        return;
    }

    uint64_t key = make_pair_key_i64(l_partkey[idx], l_suppkey[idx], key_base);
    uint64_t h = mix64(key) & (uint64_t)table_mask;

    while (true) {
        uint64_t found = table_keys[h];
        if (found == EMPTY_KEY) {
            return;
        }
        if (found == key) {
            int64_t ps_idx = table_vals[h];
            atomicAdd(&ps_sum_qty[ps_idx], l_quantity[idx]);
            return;
        }
        h = (h + 1ULL) & (uint64_t)table_mask;
    }
}
