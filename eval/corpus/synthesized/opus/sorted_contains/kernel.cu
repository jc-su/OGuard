// sorted_contains: match[i] = 1 when keys[i] is one of the sorted, distinct members.
//
// One thread per key runs a binary search for the last member <= key and
// compares it with the key. The search has no branch on the comparison (the
// step is a select), so its step count depends only on s and the lanes of a
// warp stay converged. The top levels of the search are the same few members
// for every key and stay in L1; the kernel is bound by the random reads of the
// bottom levels. (Measured on the A6000: a shared-memory table of the top
// levels and several interleaved searches per thread were no faster.)
//
// The kernel uses a grid-stride loop, so it is correct for any grid; the
// launch uses one thread per key.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

namespace sorted_contains {
constexpr int kBlock = 256;
}  // namespace sorted_contains

extern "C" __global__ void __launch_bounds__(sorted_contains::kBlock)
kernel(const int64_t* __restrict__ keys, const int64_t* __restrict__ members,
       int8_t* __restrict__ match, int64_t n, int64_t s) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const int64_t key = __ldg(keys + i);
    int8_t found = 0;
    if (s > 0 && __ldg(members) <= key) {
      int64_t pos = 0;  // invariant: members[pos] <= key
      for (int64_t len = s; len > 1;) {
        const int64_t half = len >> 1;
        pos = __ldg(members + pos + half) <= key ? pos + half : pos;
        len -= half;
      }
      found = __ldg(members + pos) == key;
    }
    match[i] = found;
  }
}
