#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    double* __restrict__ revenue_dense,
    uint8_t* __restrict__ seen,
    int64_t n,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        const int64_t k = custkey[idx];
        if (k >= 0 && k < dense_size) {
            const double rev = extendedprice[idx] * (1.0 - discount[idx]);
            atomicAdd(revenue_dense + k, rev);
            seen[k] = static_cast<uint8_t>(1);
        }
    }
}
