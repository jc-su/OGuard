"""string_contains: byte substring test of a pattern against each string.

One thread per string. The thread walks the candidate start positions of
its string and compares the pattern byte by byte in one flat loop: a match
of byte j advances j, a mismatch moves to the next position and resets j,
and j reaching k reports the match. The empty pattern matches every string.
The pattern is small and read-only, so its loads are served by L1.

Revision: the nested loop and the min/max clamp of the pattern index are
gone; the pattern index j and the byte index q = position + j are tested
against [0, k) and [0, offsets[n]) right before their loads. Under the
task's contract those tests always pass.
"""

from numba import cuda, int8, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)
THREADS = 256


def kernel(offsets, chars, pattern, match, n, k):
    i = cuda.blockIdx.x * THREADS + cuda.threadIdx.x
    if i >= n:
        return
    s = offsets[i]
    e = offsets[i + 1]
    total = offsets[n]
    found = 0
    if k == 0:
        found = 1
    else:
        last = e - k  # last candidate start position
        pos = s
        j = 0
        while pos <= last:
            if j < 0 or j >= k:
                break  # never under the contract
            q = pos + j
            if q < 0 or q >= total:
                break  # never under the contract
            if chars[q] == pattern[j]:
                j += 1
                if j == k:
                    found = 1
                    break
            else:
                pos += 1
                j = 0
    match[i] = found


_kernel = cuda.jit(kernel)


def run(offsets, chars, pattern, match, n, k):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](offsets, chars, pattern, match, n, k)
