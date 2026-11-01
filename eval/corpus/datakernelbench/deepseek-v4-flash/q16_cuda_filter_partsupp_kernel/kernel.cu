#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const int32_t* __restrict__ is_bad_supplier,
    int64_t* __restrict__ out_partkey,
    int64_t* __restrict__ out_suppkey,
    int64_t* __restrict__ out_count,
    int64_t n,
    int64_t max_supp_key
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx < n) {
        int64_t supp = ps_suppkey[idx];
        bool bad = false;
        if (supp >= 0 && supp < max_supp_key) {
            bad = (is_bad_supplier[supp] != 0);
        }
        if (!bad) {
            int64_t write_idx = atomicAdd((unsigned long long*)out_count, 1ULL);
            out_partkey[write_idx] = ps_partkey[idx];
            out_suppkey[write_idx] = supp;
        }
    }
}
