#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements)
{
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < n_elements) {
        int32_t sd   = shipdate_ptr[idx];
        double  disc = discount_ptr[idx];
        double  qty  = quantity_ptr[idx];
        bool keep = (sd   >= date_lo) && (sd   <  date_hi) &&
                    (disc >= disc_lo) && (disc <= disc_hi) &&
                    (qty  <  qty_hi);
        if (keep) {
            my_val = price_ptr[idx] * disc;
        }
    }

    block_sum[threadIdx.x] = my_val;
    __syncthreads();

    // reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_sum[0]);
    }
}
