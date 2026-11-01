#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Kernel to compute per-block valid counts and prefix sums
extern "C" __global__ void kernel(
    const int32_t* l_shipdate,
    const int32_t* o_orderdate,
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    int* block_counts,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid = 0;
    if (idx < n) {
        if (l_shipdate[idx] > shipdate_lo && o_orderdate[idx] < orderdate_hi) {
            valid = 1;
        }
    }
    
    __shared__ int s_count[BLOCK_SIZE];
    s_count[threadIdx.x] = valid;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_count[threadIdx.x] += s_count[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = s_count[0];
    }
}
