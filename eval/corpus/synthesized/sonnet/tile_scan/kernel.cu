#include <cstdint>

// One block of 256 threads per tile of 1024 rows; each thread owns 4 consecutive rows.
extern "C" __global__ void kernel(const int32_t* __restrict__ values, int32_t* __restrict__ out,
                                  int32_t* __restrict__ totals, int64_t n) {
  __shared__ int32_t warp_sums[8];
  int tid = threadIdx.x;
  int64_t base = (int64_t)blockIdx.x * 1024 + tid * 4;
  int32_t v[4];
  for (int q = 0; q < 4; q++) v[q] = base + q < n ? values[base + q] : 0;
  int32_t t = v[0] + v[1] + v[2] + v[3];
  int32_t incl = t;
  int lane = tid & 31;
  for (int d = 1; d < 32; d <<= 1) {
    int32_t x = __shfl_up_sync(0xFFFFFFFFu, incl, d);
    if (lane >= d) incl += x;
  }
  if (lane == 31) warp_sums[tid >> 5] = incl;
  __syncthreads();
  int32_t excl = incl - t;
  for (int w = 0; w < (tid >> 5); w++) excl += warp_sums[w];
  if (tid == 0) {
    int32_t total = 0;
    for (int w = 0; w < 8; w++) total += warp_sums[w];
    totals[blockIdx.x] = total;
  }
  for (int q = 0; q < 4; q++) {
    if (base + q < n) out[base + q] = excl;
    excl += v[q];
  }
}
