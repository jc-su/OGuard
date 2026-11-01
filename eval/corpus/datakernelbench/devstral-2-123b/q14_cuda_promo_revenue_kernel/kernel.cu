#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double total_rev_out[],
    double promo_rev_out[],
    int64_t n_elements
) {
    __shared__ double block_total_rev[BLOCK_SIZE];
    __shared__ double block_promo_rev[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi);
        if (keep) {
            double price = price_ptr[idx];
            double disc = discount_ptr[idx];
            double rev = price * (1.0 - disc);
            my_total = rev;
            if (promo_ptr[idx]) my_promo = rev;
        }
    }
    block_total_rev[threadIdx.x] = my_total;
    block_promo_rev[threadIdx.x] = my_promo;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total_rev[threadIdx.x] += block_total_rev[threadIdx.x + s];
            block_promo_rev[threadIdx.x] += block_promo_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd((double*)total_rev_out, block_total_rev[0]);
        atomicAdd((double*)promo_rev_out, block_promo_rev[0]);
    }
}
