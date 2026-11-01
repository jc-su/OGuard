#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const double* __restrict__ quantities,
    double* __restrict__ output,
    int64_t N
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t grp = group_ids[idx];
        double val = quantities[idx];
        atomicAdd(&output[grp], val);
    }
}
