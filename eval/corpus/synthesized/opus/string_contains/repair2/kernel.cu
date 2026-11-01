// string_contains: match[i] = 1 when the byte string i contains the pattern.
//
// Each block handles kBlock consecutive strings in two phases, so that both
// short and long strings, and any mix of them, keep the lanes busy:
//
// 1. Thread per string: a thread tests the candidate start positions of a
//    short string (at most kShortBytes bytes) one after another, and a long
//    string is left for step 2.
// 2. Warp per string: a warp takes the long strings of its own 32 lanes,
//    found with a ballot, one after another. A step of a warp tests 32
//    consecutive start positions, one per lane (the lanes read consecutive
//    bytes), the warp votes with a ballot, and it stops at the first match.
//
// A candidate is compared byte by byte and the comparison stops at the first
// mismatch. The empty pattern occurs in every string.
//
// Bounds: every access is proved by the code at the access alone, without a
// minimum, a maximum or a loop invariant. A string index s is read from
// `offsets` and written to `match` only under 0 <= s and s < n (so
// s + 1 < n + 1 too), and the column length is offsets[n]. A byte position
// and a pattern position are masked with kPositionMask = 2^31 - 1, which
// changes no position of a column with int32 offsets and keeps the index at
// least 0; a byte at position c is then read only under c < offsets[n] and
// c < the end of its string, and a pattern byte j only under j < k. Each index
// is the very value that its comparisons test, and the mask keeps the compiler
// from turning it into a pointer that steps with the loop. The kernel nests at
// most three loops, and has no per-thread array and no shared memory. The
// loads are plain loads, and the ballot and bit count use the NVVM builtins
// that the CUDA headers wrap: at low optimization levels __ldg, __ballot_sync
// and __ffs stay function calls, while the builtins compile to single
// instructions.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

extern __device__ __device_builtin__ unsigned int __nvvm_vote_ballot_sync(unsigned int mask,
                                                                          int pred);
extern __device__ __device_builtin__ int __nvvm_clz_i(int x);

namespace string_contains {
constexpr int kBlock = 256;
constexpr int kShortBytes = 128;
constexpr int64_t kPositionMask = 0x7FFFFFFF;  // byte and pattern positions are below 2^31

// Whether the pattern occurs at byte position pos of a string that ends at
// `end`, in a column of `size` bytes.
__device__ __forceinline__ bool occurs_at(const uint8_t* __restrict__ chars,
                                          const uint8_t* __restrict__ pattern, int64_t pos,
                                          int64_t end, int64_t size, int64_t k) {
  int64_t j = 0;
  while (j < k) {
    const int64_t c = (pos + j) & kPositionMask;
    const int64_t t = j & kPositionMask;
    if (c >= size || c >= end || t >= k) break;
    if (chars[c] != pattern[t]) break;
    ++j;
  }
  return j == k;
}
}  // namespace string_contains

extern "C" __global__ void __launch_bounds__(string_contains::kBlock)
kernel(const int32_t* __restrict__ offsets, const uint8_t* __restrict__ chars,
       const uint8_t* __restrict__ pattern, int8_t* __restrict__ match, int64_t n,
       int64_t k) {
  using namespace string_contains;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int64_t s = static_cast<int64_t>(blockIdx.x) * kBlock + tid;
  const int64_t size = n >= 0 ? offsets[n] : 0;  // the length of the column, offsets[n]

  // 1. Thread per short string; the long ones wait for step 2.
  bool long_string = false;
  if (s >= 0 && s < n) {
    const int64_t start = offsets[s];
    const int64_t end = offsets[s + 1];
    if (k > 0 && end - start > kShortBytes) {
      long_string = true;
    } else {
      bool found = k == 0;
      const int64_t last = end - k;  // the last candidate start
      for (int64_t pos = start; !found && pos <= last; ++pos) {
        found = occurs_at(chars, pattern, pos, end, size, k);
      }
      match[s] = found ? 1 : 0;
    }
  }

  // 2. Warp per long string (k >= 1 here): the long strings of this warp's lanes.
  for (unsigned pending = __nvvm_vote_ballot_sync(0xFFFFFFFFu, long_string); pending != 0;
       pending &= pending - 1) {
    // The lowest pending lane.
    const int64_t ls = s - lane + ((31 - __nvvm_clz_i(pending & (0u - pending))) & 31);
    if (ls < 0 || ls >= n) continue;
    const int64_t start = offsets[ls];
    const int64_t end = offsets[ls + 1];
    const int64_t last = end - k;  // the last candidate start
    bool found = false;            // the same in every lane
    for (int64_t base = start; !found && base <= last; base += 32) {
      const int64_t p = base + lane;
      const bool hit = p <= last && occurs_at(chars, pattern, p, end, size, k);
      found = __nvvm_vote_ballot_sync(0xFFFFFFFFu, hit) != 0;
    }
    if (lane == 0) match[ls] = found ? 1 : 0;
  }
}
