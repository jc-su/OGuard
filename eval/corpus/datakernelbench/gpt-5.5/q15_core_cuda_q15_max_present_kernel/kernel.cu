#include <math_constants.h>
#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ totals,
    const uint8_t* __restrict__ present,
    double* __restrict__ partial,
    int64_t n
) {
    __shared__ double sh[BLOCK_SIZE];

    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double v = -CUDART_INF;
    if (i < n && present[i] != 0) {
        v = totals[i];
    }
    sh[threadIdx.x] = v;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double other = sh[threadIdx.x + s];
            if (other > sh[threadIdx.x]) {
                sh[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sh[0];
    }
}
