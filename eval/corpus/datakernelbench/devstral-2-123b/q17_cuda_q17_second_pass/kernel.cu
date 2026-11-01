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
    const double* __restrict__ l_extendedprice_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    const double* __restrict__ sum_qty_ptr,
    const double* __restrict__ cnt_qty_ptr,
    double* __restrict__ sum_price_ptr,
    const int64_t n_items,
    const int64_t n_parts
) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    double my_sum_price = 0.0;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_items; i += blockDim.x * gridDim.x) {
        int64_t pkey = l_partkey_ptr[i];
        int64_t lo = 0, hi = n_parts;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (part_keys_ptr[mid] < pkey) lo = mid + 1;
            else hi = mid;
        }
        if (lo < n_parts && part_keys_ptr[lo] == pkey) {
            double cnt = cnt_qty_ptr[lo];
            double avg_qty = cnt > 0 ? sum_qty_ptr[lo] / cnt : 0.0;
            if (l_quantity_ptr[i] < 0.2 * avg_qty) {
                my_sum_price += l_extendedprice_ptr[i];
            }
        }
    }

    sdata[tid] = my_sum_price;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd_double(sum_price_ptr, sdata[0]);
    }
}
