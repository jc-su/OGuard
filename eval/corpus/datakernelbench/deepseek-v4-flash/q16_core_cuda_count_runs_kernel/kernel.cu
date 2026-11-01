#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ sorted_keys,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    int64_t* __restrict__ out_counts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_new_run = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx-1]);
        if (is_new_run) {
            int64_t orig_idx = sorted_indices[idx];
            int64_t g = group_ids[orig_idx];
            atomicAdd((unsigned long long*)&out_counts[g], 1ULL);
        }
    }
}
