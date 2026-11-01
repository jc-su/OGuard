#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_id_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t nid = nation_id_ptr[idx];
        double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
        atomicAdd(&out_ptr[nid], rev);
    }
}
