#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    const int64_t* __restrict__ inverse_ptr,
    double* __restrict__ agg_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        double value = supplycost_ptr[idx] * availqty_ptr[idx];
        int64_t key_idx = inverse_ptr[idx];
        atomicAdd(&agg_ptr[key_idx], value);
    }
}
