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
// Every member index is clamped to last = s - 1 where it is formed. The clamp
// never changes the value (pos + half <= pos + len - 1 <= s - 1), but it makes
// the bound of each access explicit, without a loop invariant that relates
// pos and len. The loads are plain loads: at low optimization levels __ldg
// stays a function call.
//
// Launch: 256 threads per block, ceil(n / 256) blocks (one thread per key).
#include <cstdint>

namespace sorted_contains {
constexpr int kBlock = 256;
}  // namespace sorted_contains

extern "C" __global__ void __launch_bounds__(sorted_contains::kBlock)
kernel(const int64_t* __restrict__ keys, const int64_t* __restrict__ members,
       int8_t* __restrict__ match, int64_t n, int64_t s) {
  using namespace sorted_contains;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (i >= n) return;
  const int64_t key = keys[i];
  int8_t found = 0;
  if (s > 0) {
    const int64_t last = s - 1;
    int64_t pos = 0;  // members[pos] <= key, or pos = 0
    for (int64_t len = s; len > 1;) {
      const int64_t half = len >> 1;
      const int64_t probe = pos + half < last ? pos + half : last;
      pos = members[probe] <= key ? probe : pos;
      len -= half;
    }
    found = members[pos < last ? pos : last] == key;
  }
  match[i] = found;
}
