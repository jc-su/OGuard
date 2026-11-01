#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Fused kernel: filter, project, and write valid rows with correct global indices
extern "C" __global__ void kernel(
    const int64_t* l_orderkey,
    const double* l_price,
    const double* l_discount,
    const int32_t* l_shipdate,
    const int32_t* o_orderdate,
    const int32_t* o_shippriority,
    int32_t shipdate_lo,
    int32_t orderdate_hi,
    const int* block_prefix_sums,
    int64_t* out_lk,
    double* out_rev,
    int32_t* out_od,
    int32_t* out_sp,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    bool is_valid = (l_shipdate[idx] > shipdate_lo && o_orderdate[idx] < orderdate_hi);
    
    __shared__ int s_valid[BLOCK_SIZE];
    s_valid[threadIdx.x] = is_valid ? 1 : 0;
    __syncthreads();
    
    // Inclusive prefix sum within block
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int val = 0;
        if (threadIdx.x >= stride) {
            val = s_valid[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            s_valid[threadIdx.x] += val;
        }
        __syncthreads();
    }
    
    if (is_valid) {
        // Global write index = prefix sum of all previous blocks + local rank
        int local_rank = s_valid[threadIdx.x] - 1;  // 0-indexed within block
        int64_t write_idx = (int64_t)block_prefix_sums[blockIdx.x] + local_rank;
        
        out_lk[write_idx] = l_orderkey[idx];
        out_od[write_idx] = o_orderdate[idx];
        out_sp[write_idx] = o_shippriority[idx];
        out_rev[write_idx] = l_price[idx] * (1.0 - l_discount[idx]);
    }
}
