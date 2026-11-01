#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const int32_t date_lo,
    const int32_t date_hi,
    const double disc_lo,
    const double disc_hi,
    const double qty_hi,
    double* __restrict__ out_ptr,
    const int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    double my_rev = 0.0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        double disc = discount_ptr[idx];
        double qty = quantity_ptr[idx];
        bool keep = (date_lo <= sd) && (sd < date_hi) && (disc_lo <= disc) && (disc <= disc_hi) && (qty < qty_hi);
        if (keep) {
            my_rev = price_ptr[idx] * disc;
        }
    }
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}
