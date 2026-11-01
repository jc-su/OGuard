#include <cuda_runtime.h>
#include <stdint.h>

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
    const int64_t* __restrict__ part_keys,
    double* __restrict__ sum_qty,
    double* __restrict__ count_qty,
    int64_t n_lineitems,
    int64_t n_parts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int64_t pkey = l_partkey[idx];
    int64_t pidx = binary_search(part_keys, n_parts, pkey);
    if (pidx != -1) {
        double qty = l_quantity[idx];
        atomicAdd(&sum_qty[pidx], qty);
        atomicAdd(&count_qty[pidx], 1.0);
    }
}
