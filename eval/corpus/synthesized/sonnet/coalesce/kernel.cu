#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ a, const uint8_t* __restrict__ valid,
                                  const double* __restrict__ b, double* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = ((valid[i >> 3] >> (i & 7)) & 1) ? a[i] : b[i];
}
