#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ codes,
    const double* __restrict__ bals,
    int32_t n_filtered,
    const int32_t* __restrict__ unique_codes,
    int32_t* __restrict__ numcust_out,
    double* __restrict__ totacctbal_out,
    int32_t n_unique
) {
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= n_filtered) return;

    int32_t cc = codes[idx];
    double bal = bals[idx];

    // Linear search for small group-by
    for (int32_t i = 0; i < n_unique; i++) {
        if (unique_codes[i] == cc) {
            atomicAdd(&numcust_out[i], 1);
            atomicAdd(&totacctbal_out[i], bal);
            return;
        }
    }
}
