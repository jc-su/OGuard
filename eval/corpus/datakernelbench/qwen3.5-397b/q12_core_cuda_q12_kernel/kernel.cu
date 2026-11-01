#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commit_ptr,
    const int32_t* __restrict__ receipt_ptr,
    const int32_t* __restrict__ ship_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    unsigned long long* __restrict__ out_counts,
    int64_t n_elements
) {
    __shared__ unsigned long long s_counts[4];
    
    // Initialize shared memory
    if (threadIdx.x < 4) {
        s_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Each thread accumulates locally
    unsigned long long my_counts[4] = {0, 0, 0, 0};
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t commit = commit_ptr[idx];
        int32_t receipt = receipt_ptr[idx];
        int32_t ship = ship_ptr[idx];
        
        bool keep = (commit < receipt) && (ship < commit) &&
                    (receipt >= receipt_lo) && (receipt < receipt_hi);
        
        if (keep) {
            int32_t mode = shipmode_ptr[idx];
            int32_t high = priority_ptr[idx];
            
            if (mode >= 0 && mode < 2) {
                // bucket layout: [mode0_high, mode0_low, mode1_high, mode1_low]
                int32_t bucket = mode * 2 + (high ? 0 : 1);
                my_counts[bucket]++;
            }
        }
    }
    
    // Add local counts to shared memory using atomics
    for (int b = 0; b < 4; b++) {
        atomicAdd(&s_counts[b], my_counts[b]);
    }
    __syncthreads();
    
    // Thread 0 writes to global output
    if (threadIdx.x == 0) {
        atomicAdd(&out_counts[0], s_counts[0]);
        atomicAdd(&out_counts[1], s_counts[1]);
        atomicAdd(&out_counts[2], s_counts[2]);
        atomicAdd(&out_counts[3], s_counts[3]);
    }
}
