// String prefix: write the first min(k, length) bytes of each string, padded with 0.
#include <cstdint>

extern "C" __global__ void kernel(const int32_t* offsets, const uint8_t* chars, uint8_t* out,
                                  int64_t n, int64_t k) {
    // One thread per output byte, so the stores of a warp are contiguous.
    const int64_t total = n * k;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < total;
         j += stride) {
        const int64_t i = j / k;
        const int64_t c = j - i * k;
        const int64_t start = offsets[i];
        out[j] = c < offsets[i + 1] - start ? chars[start + c] : 0;
    }
}
