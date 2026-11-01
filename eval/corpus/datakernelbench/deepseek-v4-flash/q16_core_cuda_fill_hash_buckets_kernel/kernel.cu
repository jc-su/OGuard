#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    int64_t* __restrict__ keys,  // hash = group_id * 1000003 + supp_id
    int64_t* __restrict__ indices
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = group_ids[idx];
        int64_t s = supp_ids[idx];
        keys[idx] = g * 1000003 + s;  // hash
        indices[idx] = idx;
    }
}
