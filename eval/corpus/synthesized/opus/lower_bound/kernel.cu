#include <cstdint>

extern "C" __global__ void __launch_bounds__(256)
kernel(const int64_t* __restrict__ keys, const int64_t* __restrict__ sorted,
       int32_t* __restrict__ out, int64_t n, int64_t s)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t key = keys[i];
    int64_t lo = 0;
    int64_t hi = s;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (sorted[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out[i] = static_cast<int32_t>(lo);
}
