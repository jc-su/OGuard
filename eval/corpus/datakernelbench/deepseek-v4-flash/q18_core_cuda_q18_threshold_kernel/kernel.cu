#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const int64_t* __restrict__ sorted_keys,
    const double* __restrict__ sorted_qty,
    int64_t* __restrict__ out_keys,
    double* __restrict__ out_sums,
    int64_t* __restrict__ out_count,
    double qty_threshold,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool is_start = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);
    if (!is_start) return;

    double sum = 0.0;
    int64_t key = sorted_keys[idx];
    int64_t j = idx;
    while (j < n && sorted_keys[j] == key) {
        sum += sorted_qty[j];
        j++;
    }

    if (sum > qty_threshold) {
        int64_t pos = atomicAdd((unsigned long long int*)out_count, 1ULL);
        out_keys[pos] = key;
        out_sums[pos] = sum;
    }
}
