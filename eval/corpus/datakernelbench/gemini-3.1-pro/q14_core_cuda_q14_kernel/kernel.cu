#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_total,
    double* __restrict__ out_promo,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;

    // Conditionally read price, discount, and promo flags only if the shipdate 
    // filter passes. This saves a massive amount of memory bandwidth because 
    // selectivity is very low (~1.2% for a one-month window).
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        if (sd >= date_lo && sd < date_hi) {
            double price = price_ptr[idx];
            double discount = discount_ptr[idx];
            bool promo = promo_ptr[idx];
            
            double rev = price * (1.0 - discount);
            my_total += rev;
            if (promo) {
                my_promo += rev;
            }
        }
    }

    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Single global atomic per block
    if (threadIdx.x == 0) {
        atomicAdd(out_total, block_total[0]);
        atomicAdd(out_promo, block_promo[0]);
    }
}
