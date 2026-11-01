#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ custkey_ptr,
    double* __restrict__ revenue_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double price = price_ptr[idx];
        double discount = discount_ptr[idx];
        int64_t custkey = custkey_ptr[idx];
        double revenue = price * (1.0 - discount);
        atomicAdd(&revenue_ptr[custkey], revenue);
    }
}
