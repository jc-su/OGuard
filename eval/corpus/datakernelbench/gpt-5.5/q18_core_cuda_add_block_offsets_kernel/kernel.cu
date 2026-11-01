#include <cstdint>

extern "C" __global__ void kernel(
    int64_t* __restrict__ offsets,
    const int64_t* __restrict__ block_offsets,
    int64_t n
) {
    int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid < n) {
        offsets[gid] += block_offsets[blockIdx.x];
    }
}
