"""string_length: number of UTF-8 code points per string.

A code point starts at every byte that is not a continuation byte
((byte & 0xC0) != 0x80), so the length is the count of such bytes.
The kernel picks one of two paths per launch from the average string length:

- short strings (average below WARP_MIN_AVG bytes): one thread per string
  with a serial byte loop;
- long strings: one warp per string, lanes read consecutive bytes and the
  count is a popc of the warp ballot, which gives coalesced loads and no load
  imbalance between lanes.
"""

from numba import cuda, int32, int64, uint8, uint32

FULL_MASK = 0xFFFFFFFF
THREADS = 256
MAX_BLOCKS = 2048
WARP_MIN_AVG = 64


def kernel(offsets, chars, lengths, n):
    total = offsets[n]
    if total >= n * WARP_MIN_AVG:
        lane = uint32(cuda.laneid)
        s = cuda.grid(1) >> 5
        nwarps = cuda.gridsize(1) >> 5
        while s < n:
            # Unsigned byte positions: Numba then skips the negative-index
            # wraparound check on every chars[p] load in this hot loop.
            p0 = uint32(offsets[s])
            end = uint32(offsets[s + 1])
            count = 0
            while p0 < end:
                p = p0 + lane
                is_start = False
                if p < end:
                    is_start = (chars[p] & 0xC0) != 0x80
                count += cuda.popc(cuda.ballot_sync(FULL_MASK, is_start))
                p0 += uint32(32)
            if lane == 0:
                lengths[s] = count
            s += nwarps
    else:
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        while i < n:
            # Unsigned byte positions: Numba then skips the negative-index
            # wraparound check on every chars[p] load in this hot loop.
            p = uint32(offsets[i])
            end = uint32(offsets[i + 1])
            count = 0
            while p < end:
                if (chars[p] & 0xC0) != 0x80:
                    count += 1
                p += uint32(1)
            lengths[i] = count
            i += stride


TYPES = (int32[:], uint8[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, lengths, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](offsets, chars, lengths, n)
