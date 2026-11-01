#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const uint8_t* __restrict__ bitmap,
    bool* __restrict__ out,
    int64_t n,
    int64_t bitmap_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t k = keys[idx];
        out[idx] = (k >= 0 && k < bitmap_size && bitmap[k] != static_cast<uint8_t>(0));
    }
}
