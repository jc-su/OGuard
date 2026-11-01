// string_contains: match[i] = 1 when the byte string i contains the pattern.
//
// Each block handles kBlock consecutive strings in two phases, so that both
// short and long strings, and any mix of them, keep the lanes busy:
//
// 1. Thread per string: a thread tests the candidate start positions of a
//    short string (at most kShortBytes bytes) one after another. A long
//    string goes onto a shared-memory queue instead.
// 2. Warp per string: the block's warps take the queued long strings. A step
//    of a warp covers kSteps * 32 consecutive start positions; each lane loads
//    the first bytes of its kSteps candidates together (the lanes read
//    consecutive bytes), the warp votes with a ballot, and it stops at the
//    first match.
//
// A candidate is compared further only when its first byte matches, and the
// comparison stops at the first mismatch. The empty pattern occurs in every
// string.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

namespace string_contains {
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kShortBytes = 128;
constexpr int kSteps = 8;

// Whether pattern[1:k] occurs at chars[pos + 1] (the caller checked byte 0).
__device__ __forceinline__ bool rest_matches(const uint8_t* __restrict__ chars,
                                             const uint8_t* __restrict__ pattern,
                                             int64_t pos, int64_t k) {
  int64_t j = 1;
  while (j < k && __ldg(chars + pos + j) == __ldg(pattern + j)) ++j;
  return j == k;
}
}  // namespace string_contains

extern "C" __global__ void __launch_bounds__(string_contains::kBlock)
kernel(const int32_t* __restrict__ offsets, const uint8_t* __restrict__ chars,
       const uint8_t* __restrict__ pattern, int8_t* __restrict__ match, int64_t n,
       int64_t k) {
  using namespace string_contains;
  __shared__ int32_t long_queue[kBlock];
  __shared__ int32_t long_count;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t s0 = static_cast<int64_t>(blockIdx.x) * kBlock;
  const uint8_t first = k > 0 ? __ldg(pattern) : 0;
  if (tid == 0) long_count = 0;
  __syncthreads();

  // 1. Thread per short string; queue the long ones.
  const int64_t s = s0 + tid;
  if (s < n) {
    const int64_t start = __ldg(offsets + s);
    const int64_t end = __ldg(offsets + s + 1);
    if (k > 0 && end - start > kShortBytes) {
      long_queue[atomicAdd(&long_count, 1)] = tid;
    } else {
      bool found = k == 0;
      for (int64_t pos = start; !found && pos <= end - k; ++pos) {
        found = __ldg(chars + pos) == first && rest_matches(chars, pattern, pos, k);
      }
      match[s] = found ? 1 : 0;
    }
  }
  __syncthreads();

  // 2. Warp per long string (k >= 1 here).
  const int queued = long_count;
  for (int q = warp; q < queued; q += kWarps) {
    const int64_t ls = s0 + long_queue[q];
    const int64_t last = __ldg(offsets + ls + 1) - k;  // last candidate start position
    int64_t pos = __ldg(offsets + ls) + lane;
    bool found = false;  // the same in every lane
    while (!found && __any_sync(0xFFFFFFFFu, pos <= last)) {
      uint8_t head[kSteps];
#pragma unroll
      for (int u = 0; u < kSteps; ++u) {
        const int64_t p = pos + u * 32;
        head[u] = p <= last ? __ldg(chars + p) : static_cast<uint8_t>(~first);
      }
      bool hit = false;
#pragma unroll
      for (int u = 0; u < kSteps; ++u) {
        if (!hit && head[u] == first) hit = rest_matches(chars, pattern, pos + u * 32, k);
      }
      found = __ballot_sync(0xFFFFFFFFu, hit) != 0;
      pos += kSteps * 32;
    }
    if (lane == 0) match[ls] = found ? 1 : 0;
  }
}
