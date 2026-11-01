"""segmented_sum: sums[j] = sum of values[offsets[j]:offsets[j + 1]].

The kernel picks one of two paths per launch from the average segment length
(offsets[g] / g):

- short segments (average below WARP_MIN_AVG values): one thread per
  segment with a serial loop;
- long segments: one warp per segment, lanes read consecutive values
  (coalesced) and combine their partial sums with shuffles, so a few very
  long segments cannot stall single threads.

Value positions are unsigned so Numba skips the negative-index wraparound
check on every values[] load.
"""

from numba import cuda, float64, int32, int64, uint32

FULL_MASK = 0xFFFFFFFF
THREADS = 256
MAX_BLOCKS = 2048
WARP_MIN_AVG = 32


def kernel(offsets, values, sums, g):
    n = offsets[g]
    if n >= g * WARP_MIN_AVG:
        lane = uint32(cuda.laneid)
        s = cuda.grid(1) >> 5
        nwarps = cuda.gridsize(1) >> 5
        while s < g:
            p = uint32(offsets[s]) + lane
            end = uint32(offsets[s + 1])
            acc = 0.0
            while p < end:
                acc += values[p]
                p += uint32(32)
            acc += cuda.shfl_xor_sync(FULL_MASK, acc, 16)
            acc += cuda.shfl_xor_sync(FULL_MASK, acc, 8)
            acc += cuda.shfl_xor_sync(FULL_MASK, acc, 4)
            acc += cuda.shfl_xor_sync(FULL_MASK, acc, 2)
            acc += cuda.shfl_xor_sync(FULL_MASK, acc, 1)
            if lane == 0:
                sums[s] = acc
            s += nwarps
    else:
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        while i < g:
            p = uint32(offsets[i])
            end = uint32(offsets[i + 1])
            acc = 0.0
            while p < end:
                acc += values[p]
                p += uint32(1)
            sums[i] = acc
            i += stride


TYPES = (int32[:], float64[:], float64[:], int64)

_kernel = cuda.jit(kernel)


def run(offsets, values, sums, g):
    blocks = max(1, min((g + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](offsets, values, sums, g)
