#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* total_ptr,
    double* promo_ptr_out,
    int64_t n_elements
) {
    __shared__ double s_total[BLOCK_SIZE];
    __shared__ double s_promo[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi);
        if (keep) {
            double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total = revenue;
            if (promo_ptr[idx]) my_promo = revenue;
        }
    }
    s_total[threadIdx.x] = my_total;
    s_promo[threadIdx.x] = my_promo;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_total[threadIdx.x] += s_total[threadIdx.x + s];
            s_promo[threadIdx.x] += s_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(total_ptr, s_total[0]);
        atomicAdd(promo_ptr_out, s_promo[0]);
    }
}
