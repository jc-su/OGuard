#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compare two keys (partkey, suppkey)
__device__ __forceinline__ bool key_less(int64_t p1, int64_t s1, int64_t p2, int64_t s2) {
    if (p1 != p2) return p1 < p2;
    return s1 < s2;
}

// Lower bound: first index where key >= target
__device__ int64_t lower_bound(
    const int64_t* p_keys, const int64_t* s_keys, int64_t n,
    int64_t target_p, int64_t target_s
) {
    int64_t low = 0, high = n;
    while (low < high) {
        int64_t mid = low + (high - low) / 2;
        if (key_less(p_keys[mid], s_keys[mid], target_p, target_s)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Upper bound: first index where key > target
__device__ int64_t upper_bound(
    const int64_t* p_keys, const int64_t* s_keys, int64_t n,
    int64_t target_p, int64_t target_s
) {
    int64_t low = 0, high = n;
    while (low < high) {
        int64_t mid = low + (high - low) / 2;
        if (key_less(target_p, target_s, p_keys[mid], s_keys[mid])) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

extern "C" __global__ void kernel(
    const int64_t* ps_partkey,
    const int64_t* ps_suppkey,
    const double* ps_availqty,
    int64_t n_ps,
    const int64_t* li_partkey,
    const int64_t* li_suppkey,
    const double* li_quantity,
    int64_t n_li,
    int64_t* out_suppkeys,
    int* out_count
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ps) return;

    int64_t p_key = ps_partkey[idx];
    int64_t s_key = ps_suppkey[idx];
    double avail = ps_availqty[idx];

    // Find range in lineitem using binary search
    // Both arrays are sorted by (partkey, suppkey)
    int64_t start = lower_bound(li_partkey, li_suppkey, n_li, p_key, s_key);
    int64_t end = upper_bound(li_partkey, li_suppkey, n_li, p_key, s_key);

    double sum_qty = 0.0;
    // Sum quantities in range
    for (int64_t i = start; i < end; ++i) {
        sum_qty += li_quantity[i];
    }

    // Check condition: sum_qty > 0 AND avail > 0.5 * sum_qty
    // Reference requires sum_qty > 0 (there must be lineitem rows in date range)
    if (sum_qty > 0.0 && avail > 0.5 * sum_qty) {
        int pos = atomicAdd(out_count, 1);
        out_suppkeys[pos] = s_key;
    }
}
