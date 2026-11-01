#include <cstdint>

extern "C" __global__ void __launch_bounds__(256)
kernel(const int32_t* __restrict__ group, const int64_t* __restrict__ value,
       int64_t* best, int64_t n, int64_t capacity)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int32_t k = group[i];
    const long long v = value[i];
    // best[k] only grows, so a value not above a (possibly stale) read of
    // best[k] cannot change it and the atomic can be skipped.
    if (v > best[k]) {
        atomicMax(reinterpret_cast<long long*>(best + k), v);
    }
}
