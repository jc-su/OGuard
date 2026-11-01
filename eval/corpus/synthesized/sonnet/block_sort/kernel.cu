#include <cstdint>

// One block of 256 threads per tile of 1024 rows: bitonic sort in shared memory.
// Rows past the end of the column are padded with INT32_MAX and not written back.
extern "C" __global__ void kernel(const int32_t* __restrict__ keys, int32_t* __restrict__ out, int64_t n) {
  __shared__ int32_t s[1024];
  int tid = threadIdx.x;
  int64_t tile = (int64_t)blockIdx.x * 1024;
  for (int c = 0; c < 4; c++) {
    int r = tid + c * 256;
    s[r] = tile + r < n ? keys[tile + r] : 0x7FFFFFFF;
  }
  __syncthreads();
  for (int k = 2; k <= 1024; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int c = 0; c < 2; c++) {
        int t = tid + c * 256;
        int i = ((t & ~(j - 1)) << 1) | (t & (j - 1));
        int l = i | j;
        int32_t x = s[i], y = s[l];
        bool asc = (i & k) == 0;
        if (asc ? x > y : x < y) {
          s[i] = y;
          s[l] = x;
        }
      }
      __syncthreads();
    }
  }
  for (int c = 0; c < 4; c++) {
    int r = tid + c * 256;
    if (tile + r < n) out[tile + r] = s[r];
  }
}
