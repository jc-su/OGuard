#define GROUP_COUNT 2

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_code,
    const int32_t* __restrict__ priority_high,
    const int32_t* __restrict__ commit_days,
    const int32_t* __restrict__ receipt_days,
    const int32_t* __restrict__ ship_days,
    int32_t date_lo,
    int32_t date_hi,
    unsigned long long* __restrict__ high_out,
    unsigned long long* __restrict__ low_out,
    int64_t n_elements
) {
    __shared__ unsigned long long block_high_counts[GROUP_COUNT];
    __shared__ unsigned long long block_low_counts[GROUP_COUNT];

    for (int i = 0; i < GROUP_COUNT; i++) {
        if (threadIdx.x < GROUP_COUNT) {
            block_high_counts[threadIdx.x] = 0;
            block_low_counts[threadIdx.x] = 0;
        }
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements)
        return;

    int32_t c_date = commit_days[idx];
    int32_t r_date = receipt_days[idx];
    int32_t s_date = ship_days[idx];
    int32_t sm = shipmode_code[idx];
    int32_t h_flag = priority_high[idx];

    bool keep = (c_date < r_date)
              && (s_date < c_date)
              && (r_date >= date_lo)
              && (r_date < date_hi);

    if (keep) {
        unsigned long long thread_high = h_flag;
        unsigned long long thread_low = 1 - h_flag;

        atomicAdd(&block_high_counts[sm], thread_high);
        atomicAdd(&block_low_counts[sm], thread_low);
    }
    __syncthreads();

    if (threadIdx.x < GROUP_COUNT) {
        atomicAdd(&high_out[threadIdx.x], block_high_counts[threadIdx.x]);
        atomicAdd(&low_out[threadIdx.x], block_low_counts[threadIdx.x]);
    }
}
