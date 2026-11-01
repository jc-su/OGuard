#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void kernel(
    const double* __restrict__ qty,
    const int64_t* __restrict__ group_idx,
    double* __restrict__ group_sum,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = group_idx[idx];
        atomicAdd(&group_sum[g], qty[idx]);
    }
}
