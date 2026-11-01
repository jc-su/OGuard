#include <cstdint>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_code_ptr,
    const int32_t* __restrict__ priority_high_ptr,
    const int32_t* __restrict__ l_commitdate_ptr,
    const int32_t* __restrict__ l_receiptdate_ptr,
    const int32_t* __restrict__ l_shipdate_ptr,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int64_t* __restrict__ high0_out_ptr,
    int64_t* __restrict__ high1_out_ptr,
    int64_t* __restrict__ low0_out_ptr,
    int64_t* __restrict__ low1_out_ptr,
    int64_t n_elements
) {
    __shared__ int64_t block_high0[BLOCK_SIZE];
    __shared__ int64_t block_high1[BLOCK_SIZE];
    __shared__ int64_t block_low0[BLOCK_SIZE];
    __shared__ int64_t block_low1[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t my_high0 = 0;
    int64_t my_high1 = 0;
    int64_t my_low0 = 0;
    int64_t my_low1 = 0;

    if (idx < n_elements) {
        int32_t commit = l_commitdate_ptr[idx];
        int32_t receipt = l_receiptdate_ptr[idx];
        int32_t ship = l_shipdate_ptr[idx];
        int32_t code = shipmode_code_ptr[idx];
        int32_t prio = priority_high_ptr[idx];

        if (commit < receipt && ship < commit && receipt >= receipt_lo_days && receipt < receipt_hi_days) {
            if (code == 0) {
                if (prio) my_high0 = 1;
                else my_low0 = 1;
            } else {
                if (prio) my_high1 = 1;
                else my_low1 = 1;
            }
        }
    }

    block_high0[threadIdx.x] = my_high0;
    block_high1[threadIdx.x] = my_high1;
    block_low0[threadIdx.x] = my_low0;
    block_low1[threadIdx.x] = my_low1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_high0[threadIdx.x] += block_high0[threadIdx.x + s];
            block_high1[threadIdx.x] += block_high1[threadIdx.x + s];
            block_low0[threadIdx.x] += block_low0[threadIdx.x + s];
            block_low1[threadIdx.x] += block_low1[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)high0_out_ptr, (unsigned long long)block_high0[0]);
        atomicAdd((unsigned long long*)high1_out_ptr, (unsigned long long)block_high1[0]);
        atomicAdd((unsigned long long*)low0_out_ptr, (unsigned long long)block_low0[0]);
        atomicAdd((unsigned long long*)low1_out_ptr, (unsigned long long)block_low1[0]);
    }
}
