#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__device__ __forceinline__ int64_t binary_search(
    const int64_t* __restrict__ arr,
    int64_t n,
    int64_t key)
{
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = arr[mid];
        if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < n && arr[lo] == key) return lo;
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_partkey,
    const double*  __restrict__ l_quantity,
    const double*  __restrict__ l_extendedprice,
    const int64_t* __restrict__ sorted_part_keys,
    const double*  __restrict__ thresholds,   // 0.2 * avg_qty per part
    int64_t n_parts,
    double* __restrict__ out,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    if (idx < n_rows) {
        int64_t pk = l_partkey[idx];
        int64_t part_idx = binary_search(sorted_part_keys, n_parts, pk);
        if (part_idx >= 0) {
            double q = l_quantity[idx];
            double thresh = thresholds[part_idx];
            if (q < thresh) {
                my_rev = l_extendedprice[idx];
            }
        }
    }
    __shared__ double block_sum[BLOCK_SIZE];
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    // reduction within block
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum[0]);
    }
}
