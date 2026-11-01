#include <cuda_runtime.h>

__device__ int find_part_index(const long long* arr, int n, long long val) {
    int left = 0;
    int right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        long long mid_val = arr[mid];
        if (mid_val == val) {
            return mid;
        } else if (mid_val < val) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

extern "C" __global__ void kernel(
    const long long* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const long long* __restrict__ sorted_part_keys,
    int n_parts,
    double* __restrict__ part_sums,
    double* __restrict__ part_counts,
    int64_t n_rows
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n_rows; i += stride) {
        long long pk = l_partkey[i];
        int p_idx = find_part_index(sorted_part_keys, n_parts, pk);
        if (p_idx != -1) {
            double qty = l_quantity[i];
            atomicAdd(&part_sums[p_idx], qty);
            atomicAdd(&part_counts[p_idx], 1.0);
        }
    }
}
