#include <cstdint>

extern "C" __global__ void kernel(const int32_t* __restrict__ a_offsets, const uint8_t* __restrict__ a_chars,
                                  const int32_t* __restrict__ b_offsets, const uint8_t* __restrict__ b_chars,
                                  int8_t* __restrict__ match, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int32_t a0 = a_offsets[i], b0 = b_offsets[i];
    int32_t len = a_offsets[i + 1] - a0;
    bool eq = len == b_offsets[i + 1] - b0;
    for (int32_t j = 0; eq && j < len; j++) eq = a_chars[a0 + j] == b_chars[b0 + j];
    match[i] = eq ? 1 : 0;
  }
}
