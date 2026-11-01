#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ packed,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        packed[idx] = (group_ids[idx] << 32) | (supp_ids[idx] & 0xFFFFFFFFLL);
    }
}
