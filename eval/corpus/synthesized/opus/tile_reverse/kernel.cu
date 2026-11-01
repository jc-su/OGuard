#include <cstdint>

// Reads are contiguous per warp; the mirrored writes of a warp also cover one
// contiguous range, so both sides stay coalesced.
extern "C" __global__ void __launch_bounds__(256)
kernel(const int32_t* __restrict__ values, int32_t* __restrict__ out, int64_t n)
{
    constexpr int64_t TILE = 1024;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const int64_t start = (i / TILE) * TILE;
        const int64_t end = start + TILE < n ? start + TILE : n;
        out[start + end - 1 - i] = values[i];
    }
}
