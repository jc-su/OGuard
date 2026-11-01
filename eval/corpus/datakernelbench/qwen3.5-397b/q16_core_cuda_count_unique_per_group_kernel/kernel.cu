#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ out_counts,
    int64_t n,
    int64_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Check if this is the first occurrence of this (group, supp) pair
        bool is_first = true;
        if (idx > 0) {
            is_first = (group_ids[idx] != group_ids[idx-1]) || 
                       (supp_ids[idx] != supp_ids[idx-1]);
        }
        
        if (is_first) {
            atomicAdd((unsigned long long*)&out_counts[group_ids[idx]], 1ULL);
        }
    }
}
