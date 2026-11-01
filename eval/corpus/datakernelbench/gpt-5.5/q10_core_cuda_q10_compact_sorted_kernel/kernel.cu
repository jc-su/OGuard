#include <cstdint>

extern "C" __global__ void kernel(
    const uint8_t* __restrict__ seen,
    const int64_t* __restrict__ prefix,
    const double* __restrict__ revenue_dense,
    int64_t* __restrict__ keys_out,
    double* __restrict__ revenue_out,
    int64_t dense_size
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < dense_size && seen[idx]) {
        const int64_t pos = prefix[idx];
        keys_out[pos] = idx;
        revenue_out[pos] = revenue_dense[idx];
    }
}
