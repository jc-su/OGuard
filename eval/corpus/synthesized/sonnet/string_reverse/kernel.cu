#include <cstdint>

// One thread per output byte. The string that owns byte p is the last s with offsets[s] <= p,
// so bytes are spread evenly over threads however the string lengths are distributed.
extern "C" __global__ void kernel(const int32_t* offsets, const uint8_t* chars, uint8_t* out,
                                  int64_t n) {
    const int64_t total = offsets[n];
    const int64_t start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t p = start; p < total; p += stride) {
        int64_t lo = 0, hi = n + 1;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (offsets[mid] <= p)
                lo = mid + 1;
            else
                hi = mid;
        }
        const int64_t s = lo - 1;
        out[p] = chars[(int64_t)offsets[s] + offsets[s + 1] - 1 - p];
    }
}
