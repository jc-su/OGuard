#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ out_revenue_ptr,
    int64_t n_rows,
    int64_t out_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        int64_t ck = custkey_ptr[idx];
        if (ck >= 0 && ck < out_size) {
            double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            atomicAdd(out_revenue_ptr + ck, rev);
        }
    }
}
