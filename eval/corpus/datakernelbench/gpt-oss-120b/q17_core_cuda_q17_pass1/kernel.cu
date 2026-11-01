#include <cuda.h>
#include <cuda_runtime.h>

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
    const int64_t* __restrict__ sorted_part_keys,
    int64_t n_parts,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int64_t pk = l_partkey[idx];
    int64_t part_idx = binary_search(sorted_part_keys, n_parts, pk);
    if (part_idx >= 0) {
        double q = l_quantity[idx];
        atomicAdd(&sum_qty[part_idx], q);
        atomicAdd(&cnt_qty[part_idx], 1.0);
    }
}
