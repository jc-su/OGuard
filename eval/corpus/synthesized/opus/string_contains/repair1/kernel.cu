// string_contains: match[i] = 1 when the byte string i contains the pattern.
//
// Each block handles kBlock consecutive strings in two phases, so that both
// short and long strings, and any mix of them, keep the lanes busy:
//
// 1. Thread per string: a thread tests the candidate start positions of a
//    short string (at most kShortBytes bytes) one after another, and a long
//    string is left for step 2.
// 2. Warp per string: a warp takes the long strings of its own 32 lanes,
//    found with a ballot, one after another. A step of a warp covers
//    kSteps * 32 consecutive start positions; each lane loads the first bytes
//    of its kSteps candidates together (the lanes read consecutive bytes), the
//    warp votes with a ballot, and it stops at the first match.
//
// A candidate is compared further only when its first byte matches, and the
// comparison stops at the first mismatch. The empty pattern occurs in every
// string.
//
// Bounds: a string index is compared with n before it reads `offsets`; a byte
// of string i is read at a position p only when p < offsets[i + 1] (and p is at
// least offsets[i] >= 0), which the candidate and pattern bounds already
// imply; a pattern byte j is read only when j < k. The long strings stay with
// their own warp instead of going through a shared-memory queue, so the kernel
// uses no shared memory, and a warp's step start is the same in every lane, so
// the loop bound is a plain comparison in each lane. The loads are plain
// loads, and the ballot and bit count use the NVVM builtins that the CUDA
// headers wrap: at low optimization levels __ldg, __ballot_sync and __ffs stay
// function calls, while the builtins compile to single instructions.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

extern __device__ __device_builtin__ unsigned int __nvvm_vote_ballot_sync(unsigned int mask,
                                                                          int pred);
extern __device__ __device_builtin__ int __nvvm_clz_i(int x);

namespace string_contains {
constexpr int kBlock = 256;
constexpr int kShortBytes = 128;
constexpr int kSteps = 8;

// Whether pattern[1:k] occurs at chars[pos + 1] (the caller checked byte 0),
// reading chars only below `end`.
__device__ __forceinline__ bool rest_matches(const uint8_t* __restrict__ chars,
                                             const uint8_t* __restrict__ pattern, int64_t pos,
                                             int64_t end, int64_t k) {
  int64_t j = 1;
  while (j < k) {
    const int64_t c = pos + j;
    if (c >= end || chars[c] != pattern[j]) break;
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
  const int64_t s0 = static_cast<int64_t>(blockIdx.x) * kBlock;
  const uint8_t first = k > 0 ? pattern[0] : 0;

  // 1. Thread per short string; the long ones wait for step 2.
  bool long_string = false;
  const int64_t s = s0 + tid;
  if (s < n) {
    const int64_t start = offsets[s];
    const int64_t end = offsets[s + 1];
    if (k > 0 && end - start > kShortBytes) {
      long_string = true;
    } else {
      bool found = k == 0;
      const int64_t last = end - k;  // the last candidate start
      for (int64_t pos = start; !found && pos <= last; ++pos) {
        found = pos < end && chars[pos] == first && rest_matches(chars, pattern, pos, end, k);
      }
      match[s] = found ? 1 : 0;
    }
  }

  // 2. Warp per long string (k >= 1 here): the long strings of this warp's lanes.
  for (unsigned pending = __nvvm_vote_ballot_sync(0xFFFFFFFFu, long_string); pending != 0;
       pending &= pending - 1) {
    // The lowest pending lane.
    const int64_t ls = s - lane + ((31 - __nvvm_clz_i(pending & (0u - pending))) & 31);
    if (ls >= n) continue;
    const int64_t start = offsets[ls];
    const int64_t end = offsets[ls + 1];
    const int64_t last = end - k;  // the last candidate start
    bool found = false;            // the same in every lane
    for (int64_t base = start; !found && base <= last; base += kSteps * 32) {
      int head[kSteps];
      for (int u = 0; u < kSteps; ++u) {
        const int64_t p = base + u * 32 + lane;
        head[u] = p <= last && p < end ? chars[p] : -1;
      }
      bool hit = false;
      for (int u = 0; u < kSteps; ++u) {
        if (!hit && head[u] == first) {
          hit = rest_matches(chars, pattern, base + u * 32 + lane, end, k);
        }
      }
      found = __nvvm_vote_ballot_sync(0xFFFFFFFFu, hit) != 0;
    }
    if (lane == 0) match[ls] = found ? 1 : 0;
  }
}
