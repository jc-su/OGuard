#include <cuda_runtime.h>

__device__ inline double atomicAdd_double(double *addr, double val) {
    unsigned long long *address_as_ull = (unsigned long long *)addr;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    double* __restrict__ sum_qty_ptr,
    double* __restrict__ cnt_qty_ptr,
    const int64_t n_items,
    const int64_t n_parts
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_items; i += blockDim.x * gridDim.x) {
        int64_t pkey = l_partkey_ptr[i];
        int64_t lo = 0, hi = n_parts;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (part_keys_ptr[mid] < pkey) lo = mid + 1;
            else hi = mid;
        }
        if (lo < n_parts && part_keys_ptr[lo] == pkey) {
            atomicAdd_double(&sum_qty_ptr[lo], l_quantity_ptr[i]);
            atomicAdd_double(&cnt_qty_ptr[lo], 1.0);
        }
    }
}
