#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const bool* __restrict__ p_type_is_promo,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ total_revenue,
    double* __restrict__ promo_revenue,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    
    if (idx < n_elements) {
        int32_t sd = l_shipdate_days[idx];
        // Filter: shipdate_lo <= sd < shipdate_hi
        if (sd >= shipdate_lo_days && sd < shipdate_hi_days) {
            double price = l_extendedprice[idx];
            double disc = l_discount[idx];
            double revenue = price * (1.0 - disc);
            my_total = revenue;
            if (p_type_is_promo[idx]) {
                my_promo = revenue;
            }
        }
    }
    
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to global sums
    if (threadIdx.x == 0) {
        atomicAdd(total_revenue, block_total[0]);
        atomicAdd(promo_revenue, block_promo[0]);
    }
}
