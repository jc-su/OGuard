#include <cstdint>

__device__ int64_t binary_search(const int64_t* arr, int64_t n, int64_t key) {
    int64_t lo = 0;
    int64_t hi = n - 1;
    while (lo <= hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = arr[mid];
        if (val == key) {
            return mid;
        } else if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ line_partkey,
    const double* __restrict__ line_quantity,
    const int64_t* __restrict__ sorted_keys,
    int64_t n_parts,
    double* __restrict__ sum_qty,
    double* __restrict__ cnt_qty,
    int64_t n_rows)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int64_t key = line_partkey[idx];
    int64_t part_idx = binary_search(sorted_keys, n_parts, key);
    if (part_idx >= 0) {
        // atomically accumulate quantity and count (as double to stay fp64)
        atomicAdd(&sum_qty[part_idx], line_quantity[idx]);
        atomicAdd(&cnt_qty[part_idx], 1.0);
    }
}
