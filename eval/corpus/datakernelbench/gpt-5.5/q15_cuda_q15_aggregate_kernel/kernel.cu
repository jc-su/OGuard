#include <stdint.h>

#define EMPTY_KEY (-1LL)

__device__ __forceinline__ uint64_t q15_hash_i64(int64_t key) {
    uint64_t x = static_cast<uint64_t>(key);
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ int64_t q15_lookup_supplier(
    int64_t key,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_mask
) {
    uint64_t h = q15_hash_i64(key);
    int64_t slot = static_cast<int64_t>(h & static_cast<uint64_t>(hash_mask));

    #pragma unroll 1
    for (int64_t probe = 0; probe <= hash_mask; ++probe) {
        int64_t cur = hash_keys[slot];
        if (cur == key) {
            return hash_vals[slot];
        }
        if (cur == EMPTY_KEY) {
            return -1;
        }
        slot = (slot + 1) & hash_mask;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ hash_keys,
    const int64_t* __restrict__ hash_vals,
    int64_t hash_mask,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ totals,
    uint8_t* __restrict__ present,
    int64_t n_line
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_line) {
        return;
    }

    int32_t shipdate = l_shipdate_days[idx];
    if (shipdate >= shipdate_lo_days && shipdate < shipdate_hi_days) {
        int64_t suppkey = l_suppkey[idx];
        int64_t supplier_idx = q15_lookup_supplier(suppkey, hash_keys, hash_vals, hash_mask);
        if (supplier_idx >= 0) {
            double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&totals[supplier_idx], revenue);
            present[supplier_idx] = static_cast<uint8_t>(1);
        }
    }
}
