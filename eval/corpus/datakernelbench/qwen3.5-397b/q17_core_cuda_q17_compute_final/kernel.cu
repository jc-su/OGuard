#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

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
    const double* __restrict__ l_extendedprice,
    const long long* __restrict__ sorted_part_keys,
    int n_parts,
    const double* __restrict__ part_avgs,
    double* __restrict__ out_total,
    int64_t n_rows
) {
    __shared__ double s_block_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    double my_sum = 0.0;

    for (int64_t i = idx; i < n_rows; i += stride) {
        long long pk = l_partkey[i];
        int p_idx = find_part_index(sorted_part_keys, n_parts, pk);
        if (p_idx != -1) {
            double qty = l_quantity[i];
            double avg = part_avgs[p_idx];
            if (qty < 0.2 * avg) {
                my_sum += l_extendedprice[i];
            }
        }
    }

    s_block_sum[tid] = my_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_block_sum[tid] += s_block_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_total, s_block_sum[0]);
    }
}
