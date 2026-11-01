#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ out_composites,
    int64_t* __restrict__ out_count,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx < n) {
        int64_t write_idx = atomicAdd((unsigned long long*)out_count, 1ULL);
        // Encode as two 32-bit halves: (group_id << 32) | (supp_id & 0xFFFFFFFF)
        int64_t composite = (group_ids[idx] << 32) | (supp_ids[idx] & 0xFFFFFFFFLL);
        out_composites[write_idx] = composite;
    }
}
