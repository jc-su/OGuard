#include <cstdint>

// One warp per segment, with warps taking segments in a grid-stride loop.
extern "C" __global__ void kernel(const int32_t* offsets, const int64_t* values, int64_t* maxes,
                                  int64_t g) {
    const int lane = threadIdx.x & 31;
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t nwarps = ((int64_t)gridDim.x * blockDim.x) >> 5;
    for (int64_t j = warp; j < g; j += nwarps) {
        int64_t m = INT64_MIN;
        const int64_t end = offsets[j + 1];
        for (int64_t k = (int64_t)offsets[j] + lane; k < end; k += 32) {
            int64_t v = values[k];
            m = v > m ? v : m;
        }
        for (int d = 16; d > 0; d >>= 1) {
            int64_t other = __shfl_down_sync(0xFFFFFFFFu, m, d);
            m = other > m ? other : m;
        }
        if (lane == 0) maxes[j] = m;
    }
}
