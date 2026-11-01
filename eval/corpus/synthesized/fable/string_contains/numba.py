"""string_contains: byte substring test of a pattern against each string.

One thread per string. The thread scans the candidate start positions of its
string, compares the first pattern byte first and only then the rest, and
stops at the first occurrence. The empty pattern matches every string. The
pattern is small and read-only, so its loads are served by L1.
"""

from numba import cuda, int8, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)
THREADS = 256


def kernel(offsets, chars, pattern, match, n, k):
    i = cuda.grid(1)
    if i >= n:
        return
    s = offsets[i]
    e = offsets[i + 1]
    found = 0
    if k == 0:
        found = 1
    elif e - s >= k:
        p0 = pattern[0]
        last = e - k
        pos = s
        while pos <= last:
            if chars[pos] == p0:
                j = 1
                while j < k and chars[pos + j] == pattern[j]:
                    j += 1
                if j == k:
                    found = 1
                    break
            pos += 1
    match[i] = found


_kernel = cuda.jit(kernel)


def run(offsets, chars, pattern, match, n, k):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](offsets, chars, pattern, match, n, k)
