#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commit_ptr,
    const int32_t* __restrict__ receipt_ptr,
    const int32_t* __restrict__ ship_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ high_out,
    unsigned long long* __restrict__ low_out,
    int64_t n_elements
) {
    __shared__ unsigned long long block_high[BLOCK_SIZE];
    __shared__ unsigned long long block_low[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long my_high = 0;
    unsigned long long my_low = 0;
    if (idx < n_elements) {
        int32_t code = shipmode_ptr[idx];
        if (code == 1) {
            int32_t commit = commit_ptr[idx];
            int32_t receipt = receipt_ptr[idx];
            int32_t ship = ship_ptr[idx];
            bool keep = (commit < receipt) && (ship < commit) && (receipt >= receipt_lo) && (receipt < receipt_hi);
            if (keep) {
                int32_t high = priority_ptr[idx];
                if (high) my_high = 1;
                else my_low = 1;
            }
        }
    }
    block_high[threadIdx.x] = my_high;
    block_low[threadIdx.x] = my_low;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_high[threadIdx.x] += block_high[threadIdx.x + s];
            block_low[threadIdx.x] += block_low[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&high_out[1], block_high[0]);
        atomicAdd(&low_out[1], block_low[0]);
    }
}
