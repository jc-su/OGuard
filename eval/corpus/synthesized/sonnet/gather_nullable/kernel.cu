#include <cstdint>

extern "C" __global__ void kernel(const double* __restrict__ col, const int32_t* __restrict__ idx,
                                  double* __restrict__ out, int64_t m, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    int32_t j = idx[i];
    out[i] = j < 0 ? __longlong_as_double(0x7FF8000000000000LL) : col[j];
  }
}
