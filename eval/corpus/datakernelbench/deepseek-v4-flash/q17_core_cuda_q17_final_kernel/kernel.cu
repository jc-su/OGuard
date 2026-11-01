#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ sel_idx,
    const double* __restrict__ sel_qty,
    const double* __restrict__ sel_price,
    const double* __restrict__ avg_qty_ptr,
    int64_t n_sel,
    double* __restrict__ out_ptr
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;

    if (idx < n_sel) {
        double qty = sel_qty[idx];
        double avg_val = avg_qty_ptr[sel_idx[idx]];
        double threshold = 0.2 * avg_val;
        if (qty < threshold) {
            my_rev = sel_price[idx];
        }
    }

    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}
