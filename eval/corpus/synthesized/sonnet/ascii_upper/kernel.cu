#include <cstdint>

extern "C" __global__ void kernel(const int32_t* __restrict__ offsets, const uint8_t* __restrict__ chars,
                                  uint8_t* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < offsets[n]) {
    uint8_t c = chars[i];
    out[i] = (c >= 'a' && c <= 'z') ? c - 32 : c;
  }
}
