// string_length: number of UTF-8 code points per string.
//
// A code point starts at every byte that is not a continuation byte
// ((byte & 0xC0) != 0x80), so the length is the count of such bytes.
// The kernel picks one of two paths per launch from the average string length:
//
// - short strings (average below WARP_MIN_AVG bytes): one thread per string
//   with a serial byte loop;
// - long strings: one warp per string, lanes read consecutive bytes and the
//   count is a popc of the warp ballot, which gives coalesced loads and no load
//   imbalance between lanes.

#include <cstdint>

#define WARP_MIN_AVG 64

extern "C" __global__ void kernel(const int* __restrict__ offsets,
                                  const uint8_t* __restrict__ chars,
                                  int* __restrict__ lengths,
                                  long long n) {
    const long long total = offsets[n];
    if (total >= n * WARP_MIN_AVG) {
        const unsigned lane = threadIdx.x & 31u;
        const long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
        for (long long s = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; s < n; s += nwarps) {
            const int begin = offsets[s];
            const int end = offsets[s + 1];
            int count = 0;
            for (int p0 = begin; p0 < end; p0 += 32) {
                const int p = p0 + (int)lane;
                const bool is_start = (p < end) && ((chars[p] & 0xC0u) != 0x80u);
                count += __popc(__ballot_sync(0xffffffffu, is_start));
            }
            if (lane == 0u) lengths[s] = count;
        }
    } else {
        const long long stride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            const int begin = offsets[i];
            const int end = offsets[i + 1];
            int count = 0;
            for (int p = begin; p < end; ++p) count += ((chars[p] & 0xC0u) != 0x80u);
            lengths[i] = count;
        }
    }
}
