// rolling_sum: trailing window sums of a float64 column.
//
// The van Herk / Gil-Werman scheme, which needs O(1) additions per output and
// no subtraction (so no cancellation, unlike differences of prefix sums). Cut
// the column into segments of w = window rows. Every window [i - w + 1, i]
// contains exactly one segment start c, so
//
//     out[i] = suffix(i - w + 1) + prefix(i)
//
// where prefix(i) sums the segment of i from c up to i, and suffix(j) sums the
// segment of j from j to c - 1 (zero when j = c).
//
// Each block computes the tile of kTile outputs [t0, t0 + kTile) from the span
// of kTile + w - 1 inputs that its windows cover (rows outside [0, n) read as
// zero), staged in shared memory. Span position x is row t0 - w + 1 + x, and
// the segments start where (x + 1) % w == 0. Output u needs prefix at
// x = u + w - 1 (the "prefix range" [w - 1, w - 1 + kTile)) and suffix at
// x = u, whose segment may end past kTile (the "suffix range" [0, suffix_end)).
//
// Both are segmented scans, computed in three steps:
// 1. Each thread scans a chunk of each range sequentially (kPrefixChunk and
//    kSuffixChunk positions) and records the chunk's total and whether the
//    chunk holds a segment boundary.
// 2. Warp 0 scans the 256 prefix chunk totals and warp 1 the 256 suffix chunk
//    totals (8 per lane sequentially, then warp shuffles), giving the carry
//    into each chunk.
// 3. An output adds the carries to its local prefix and suffix only when its
//    segment crosses the chunk boundary.
//
// The RTX A6000 (GA102) runs float64 additions at 1/64 of the float32 rate,
// so the kernel keeps the additions near 3 per output (a local prefix, a local
// suffix, the carries, the final sum) plus the few of step 2.
//
// Launch: 256 threads per block, ceil(n / 1024) blocks; window in [1, 1024].
#include <cstdint>

namespace rolling_sum {
constexpr int kBlock = 256;
constexpr int kTile = 1024;
constexpr int kMaxWindow = 1024;
constexpr int kSpanCap = kTile + kMaxWindow;  // >= kTile + w - 1
constexpr int kPrefixChunk = kTile / kBlock;  // 4: the prefix range has kTile positions
constexpr int kSuffixChunk = 8;               // 256 * 8 >= kTile + w - 1
constexpr int kLaneChunks = kBlock / 32;      // chunk totals per lane in step 2
}  // namespace rolling_sum

extern "C" __global__ void __launch_bounds__(rolling_sum::kBlock)
kernel(const double* __restrict__ values, double* __restrict__ out, int64_t n,
       int64_t window) {
  using namespace rolling_sum;
  __shared__ double xs[kSpanCap];     // inputs; the prefix range becomes local prefix sums
  __shared__ double suffix[kTile];    // local suffix sums of the first kTile positions
  __shared__ double prefix_carry[kBlock];  // chunk totals, then carries into the chunks
  __shared__ double suffix_carry[kBlock];
  __shared__ bool prefix_flag[kBlock];
  __shared__ bool suffix_flag[kBlock];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = static_cast<int>(window);
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int span = kTile + w - 1;
  const int64_t g0 = t0 - w + 1;

  for (int x = tid; x < span; x += kBlock) {
    const int64_t g = g0 + x;
    xs[x] = g >= 0 && g < n ? __ldg(values + g) : 0.0;
  }
  // The suffix range ends with the segment of position kTile - 1, or at
  // kTile - 1 when that position starts a segment (its suffix is not used).
  const int q = kTile % w;  // (x + 1) % w at x = kTile - 1
  const int suffix_end = q == 0 ? kTile - 1 : kTile + w - 1 - q;
  __syncthreads();

  // 1a. Local suffix sums of chunk [8t, 8t + 8) of the suffix range, right to left.
  {
    double run = 0.0;
    bool boundary = false;
    const int base = tid * kSuffixChunk;
    int phase = (base + kSuffixChunk) % w;  // (x + 1) % w at x = base + 7
#pragma unroll
    for (int j = kSuffixChunk - 1; j >= 0; --j) {
      const int x = base + j;
      if (x < suffix_end) {
        const bool end = phase == w - 1;
        const double sum = run + xs[x];
        run = end ? xs[x] : sum;
        boundary = boundary || end;
        if (x < kTile) suffix[x] = run;
      }
      phase = phase > 0 ? phase - 1 : w - 1;
    }
    suffix_carry[tid] = run;
    suffix_flag[tid] = boundary;
  }
  __syncthreads();  // the suffix pass reads positions that step 1b overwrites

  // 1b. Local prefix sums of chunk [4t, 4t + 4) of the prefix range (u = x - w + 1).
  {
    double run = 0.0;
    bool boundary = false;
    const int base = tid * kPrefixChunk;
    int phase = base % w;  // u % w; u starts a segment when it is 0
#pragma unroll
    for (int j = 0; j < kPrefixChunk; ++j) {
      double* slot = xs + (base + j + w - 1);
      const bool start = phase == 0;
      const double sum = run + *slot;
      run = start ? *slot : sum;
      boundary = boundary || start;
      *slot = run;
      phase = phase < w - 1 ? phase + 1 : 0;
    }
    prefix_carry[tid] = run;
    prefix_flag[tid] = boundary;
  }
  __syncthreads();

  // 2. Carries into the chunks: exclusive segmented scans of the chunk totals.
  //    A pair (flag, value) absorbs the pairs before it unless its flag is set.
  if (tid < 64) {
    const bool forward = tid < 32;
    double* total = forward ? prefix_carry : suffix_carry;
    bool* flag = forward ? prefix_flag : suffix_flag;
    const int first = lane * kLaneChunks;
    // The lane's aggregate over its chunks, in scan order.
    double v = 0.0;
    bool f = false;
#pragma unroll
    for (int j = 0; j < kLaneChunks; ++j) {
      const int c = forward ? first + j : first + kLaneChunks - 1 - j;
      v = flag[c] ? total[c] : v + total[c];
      f = f || flag[c];
    }
    // Inclusive scan over the lanes (forward: from lane 0; backward: from lane 31).
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const double ov = forward ? __shfl_up_sync(0xFFFFFFFFu, v, offset)
                                : __shfl_down_sync(0xFFFFFFFFu, v, offset);
      const bool of = forward ? __shfl_up_sync(0xFFFFFFFFu, f, offset)
                              : __shfl_down_sync(0xFFFFFFFFu, f, offset);
      const bool inside = forward ? lane >= offset : lane + offset < 32;
      if (inside) {
        v = f ? v : ov + v;
        f = f || of;
      }
    }
    double carry = forward ? __shfl_up_sync(0xFFFFFFFFu, v, 1)
                           : __shfl_down_sync(0xFFFFFFFFu, v, 1);
    if (forward ? lane == 0 : lane == 31) carry = 0.0;
#pragma unroll
    for (int j = 0; j < kLaneChunks; ++j) {
      const int c = forward ? first + j : first + kLaneChunks - 1 - j;
      const double chunk_total = total[c];
      total[c] = carry;
      carry = flag[c] ? chunk_total : carry + chunk_total;
    }
  }
  __syncthreads();

  // 3. out[t0 + u] = prefix(u + w - 1) + suffix(u); suffix(u) = 0 when u starts a segment.
  const int step = kBlock % w;
  int u_mod = tid % w;  // u % w
#pragma unroll
  for (int k = 0; k < kPrefixChunk; ++k) {
    const int u = tid + k * kBlock;
    // The prefix at u + w - 1 carries in when its segment started before its chunk.
    double p = xs[u + w - 1];
    if (u_mod > (u & (kPrefixChunk - 1))) p += prefix_carry[u / kPrefixChunk];
    // The suffix at u carries in when its segment ends past its chunk.
    const int phase = u_mod + 1 == w ? 0 : u_mod + 1;  // (u + 1) % w
    if (phase != 0) {
      double s = suffix[u];
      if (u + (w - 1 - phase) > (u | (kSuffixChunk - 1))) s += suffix_carry[u / kSuffixChunk];
      p += s;
    }
    const int64_t i = t0 + u;
    if (i < n) out[i] = p;
    u_mod += step;
    if (u_mod >= w) u_mod -= w;
  }
}
