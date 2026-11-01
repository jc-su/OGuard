#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

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
    int64_t N)
{
    __shared__ double shmem[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < N) {
        int32_t sd = shipdate[idx];
        double d  = discount[idx];
        double q  = quantity[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi) &&
                    (d  >= disc_lo) && (d <= disc_hi) &&
                    (q  < qty_hi);
        if (keep) {
            my_val = price[idx] * d;
        }
    }

    shmem[threadIdx.x] = my_val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out, shmem[0]);
    }
}
