#include <cstdint>

// One thread per byte. The thread finds the string that holds its byte by a
// binary search for the first offset above the byte position, then writes the
// byte to the mirrored position of that string. Launch with at least
// offsets[n] - offsets[0] threads.
extern "C" __global__ void __launch_bounds__(256)
kernel(const int32_t* __restrict__ offsets, const uint8_t* __restrict__ chars,
       uint8_t* __restrict__ out, int64_t n)
{
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t p = static_cast<int64_t>(offsets[0]) + t;
    if (p >= offsets[n]) {
        return;
    }
    int64_t lo = 1;
    int64_t hi = n;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (offsets[mid] > p) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    const int64_t start = offsets[lo - 1];
    const int64_t end = offsets[lo];
    out[start + end - 1 - p] = chars[p];
}
