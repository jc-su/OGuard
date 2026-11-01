// sorted_contains: match[i] = 1 when keys[i] is one of the sorted, distinct members.
//
// One thread per key finds the last member <= key by binary lifting and
// compares it with the key: starting from pos = 0, it tries the steps
// bit = 2^k, 2^(k-1), ..., 1 (2^k the highest power of two <= s) and moves to
// pos + bit when that member exists and is <= key. The step count depends
// only on s, so the lanes of a warp stay converged. The top levels of the
// search are the same few members for every key and stay in L1; the kernel is
// bound by the random reads of the bottom levels.
//
// Bounds: every access is proved by the code at the access alone, without a
// minimum, a maximum or a loop invariant. A key is read and a match written
// only under the two comparisons 0 <= i and i < n, and a member only under the
// two comparisons 0 <= index and index < s of the very index used. The loop
// runs on the step, which comes from s by bit operations, and not on a
// comparison with s itself, so the compiler keeps these comparisons as they
// are written. The loads are plain loads: at low optimization levels __ldg
// stays a function call.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

namespace sorted_contains {
constexpr int kBlock = 256;
}  // namespace sorted_contains

extern "C" __global__ void __launch_bounds__(sorted_contains::kBlock)
kernel(const int64_t* __restrict__ keys, const int64_t* __restrict__ members,
       int8_t* __restrict__ match, int64_t n, int64_t s) {
  using namespace sorted_contains;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (i < 0 || i >= n) return;
  const int64_t key = keys[i];
  // The highest power of two <= s, or 0 when s = 0.
  int64_t high = s;
  high |= high >> 1;
  high |= high >> 2;
  high |= high >> 4;
  high |= high >> 8;
  high |= high >> 16;
  high |= high >> 32;
  int64_t pos = 0;  // members[pos] <= key, or pos = 0
  for (int64_t bit = high - (high >> 1); bit > 0; bit >>= 1) {
    const int64_t probe = pos + bit;
    if (probe >= 0 && probe < s && members[probe] <= key) pos = probe;
  }
  int8_t found = 0;
  if (pos >= 0 && pos < s) found = members[pos] == key;
  match[i] = found;
}
