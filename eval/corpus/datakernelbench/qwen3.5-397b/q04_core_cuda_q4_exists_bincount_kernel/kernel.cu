#include <cuda_runtime.h>

#define MAX_PRIORITIES 10

extern "C" __global__ void kernel(
    const int64_t* __restrict__ filt_orderkey,
    const int32_t* __restrict__ filt_priority,
    const int64_t* __restrict__ unique_l_orderkey,
    int64_t n_filt_orders,
    int64_t n_unique_lineitems,
    int64_t num_priorities,
    unsigned long long int* __restrict__ priority_counts
) {
    __shared__ unsigned long long int block_counts[MAX_PRIORITIES];
    
    for (int i = threadIdx.x; i < num_priorities; i += blockDim.x) {
        block_counts[i] = 0;
    }
    __syncthreads();
    
    int64_t order_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (order_idx < n_filt_orders) {
        int64_t order_key = filt_orderkey[order_idx];
        int32_t priority = filt_priority[order_idx];
        
        bool exists = false;
        int64_t lo = 0, hi = n_unique_lineitems;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (unique_l_orderkey[mid] < order_key) {
                lo = mid + 1;
            } else if (unique_l_orderkey[mid] > order_key) {
                hi = mid;
            } else {
                exists = true;
                break;
            }
        }
        
        if (exists && priority >= 0 && priority < num_priorities) {
            atomicAdd(&block_counts[priority], 1ULL);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_priorities; i++) {
            if (block_counts[i] > 0) {
                atomicAdd(&priority_counts[i], block_counts[i]);
            }
        }
    }
}
