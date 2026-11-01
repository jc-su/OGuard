#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_disc,
    const int32_t* __restrict__ l_shipdate,
    const int64_t* __restrict__ supp_pos,
    double* __restrict__ block_totals,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_li,
    int64_t n_sup,
    int64_t block_stride
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    double* my_block_totals = block_totals + (int64_t)blockIdx.x * block_stride;

    for (int64_t idx = tid; idx < n_li; idx += stride) {
        int32_t sd = l_shipdate[idx];
        if (sd < date_lo || sd >= date_hi) continue;

        int64_t pos = supp_pos[idx];
        if (pos < 0) continue;
        
        double rev = l_price[idx] * (1.0 - l_disc[idx]);
        atomicAdd(&my_block_totals[pos], rev);
    }
}
