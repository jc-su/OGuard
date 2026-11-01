#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

__device__ __forceinline__ int64_t binary_search(const int64_t* arr, int64_t n, int64_t key) {
    int64_t low = 0;
    int64_t high = n;
    while (low < high) {
        int64_t mid = low + ((high - low) >> 1);
        if (arr[mid] < key) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low < n && arr[low] == key) {
        return low;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int64_t* __restrict__ part_keys,
    const double* __restrict__ sum_qty,
    const double* __restrict__ count_qty,
    double* __restrict__ revenue,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_rev = 0.0;
    if (idx < n_lineitems) {
        int64_t pkey = l_partkey[idx];
        int64_t pidx = binary_search(part_keys, n_parts, pkey);
        if (pidx != -1) {
            double s = sum_qty[pidx];
            double c = count_qty[pidx];
            double avg = (c > 0.0) ? (s / c) : 0.0;
            double thresh = 0.2 * avg;
            if (l_quantity[idx] < thresh) {
                my_rev = l_extendedprice[idx];
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(revenue, block_revenue[0]);
    }
}
