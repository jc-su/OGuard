#include <cstdint>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    uint8_t* __restrict__ bitmap,
    int64_t n_keys,
    int64_t bitmap_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_keys) {
        int64_t k = keys[idx];
        if (k >= 0 && k < bitmap_size) {
            bitmap[k] = static_cast<uint8_t>(1);
        }
    }
}
