#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ quantity,
    const int32_t* __restrict__ shipdate,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double sdata[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < n) {
        int32_t sd = shipdate[idx];
        double d = discount[idx];
        double q = quantity[idx];
        if (sd >= date_lo && sd < date_hi && d >= disc_lo && d <= disc_hi && q < qty_hi) {
            val = price[idx] * d;
        }
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}
