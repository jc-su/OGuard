#include <climits>
#include <cstdint>

// One warp per segment. The test seg < g is uniform within a warp because the
// block size (256) is a multiple of the warp size, so every lane takes part in
// the shuffles.
extern "C" __global__ void __launch_bounds__(256)
kernel(const int32_t* __restrict__ offsets, const int64_t* __restrict__ values,
       int64_t* __restrict__ maxes, int64_t g)
{
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t seg = t / 32;
    const int lane = threadIdx.x & 31;
    if (seg >= g) {
        return;
    }
    const int64_t start = offsets[seg];
    const int64_t end = offsets[seg + 1];
    long long best = LLONG_MIN;
    for (int64_t j = start + lane; j < end; j += 32) {
        const long long v = values[j];
        best = v > best ? v : best;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const long long other = __shfl_down_sync(0xffffffffu, best, delta);
        best = other > best ? other : best;
    }
    if (lane == 0) {
        maxes[seg] = best;
    }
}
