#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ disc_ptr,
    double* __restrict__ volume_out,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        volume_out[idx] = price_ptr[idx] * (1.0 - disc_ptr[idx]);
    }
}
