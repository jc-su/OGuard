#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ extendedprice_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ revenue_ptr,
    int64_t n_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        revenue_ptr[idx] = extendedprice_ptr[idx] * (1.0 - discount_ptr[idx]);
    }
}
