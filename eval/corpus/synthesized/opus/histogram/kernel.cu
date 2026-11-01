// Task 11, histogram: count the keys by the byte (key >> shift) & 255.
//
// This is the counting pass of one radix sort digit. Each block counts its keys
// in a private 256-bin histogram in shared memory, then adds its nonzero bins
// to counts in global memory. counts must be zero on entry. The grid-stride
// loop accepts any grid size, so the caller can launch a few blocks per SM and
// keep the number of global atomic adds (256 per block) small.
#include <cstdint>

extern "C" __global__ void kernel(const uint32_t* __restrict__ keys,
                                  int32_t* __restrict__ counts,
                                  int64_t shift,
                                  int64_t n) {
  constexpr int kBins = 256;
  __shared__ int32_t local[kBins];
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) local[b] = 0;
  __syncthreads();

  const uint32_t s = static_cast<uint32_t>(shift);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    atomicAdd(&local[(keys[i] >> s) & 0xFFu], 1);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    const int32_t count = local[b];
    if (count != 0) atomicAdd(&counts[b], count);
  }
}
