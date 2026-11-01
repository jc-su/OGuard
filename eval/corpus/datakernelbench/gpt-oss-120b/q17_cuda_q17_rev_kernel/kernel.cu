#include <cstdint>

#define BLOCK_SIZE 1024

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
    const double* __restrict__ line_price,
    const int64_t* __restrict__ sorted_keys,
    const double* __restrict__ thresholds,
    int64_t n_parts,
    double* __restrict__ out_sum,
    int64_t n_rows)
{
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_rows) {
        int64_t key = line_partkey[idx];
        int64_t part_idx = binary_search(sorted_keys, n_parts, key);
        if (part_idx >= 0) {
            double qty = line_quantity[idx];
            double th = thresholds[part_idx];
            if (qty < th) {
                my_rev = line_price[idx];
            }
        }
    }
    block_sum[threadIdx.x] = my_rev;
    __syncthreads();

    // intra‑block reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum[0]);
    }
}
