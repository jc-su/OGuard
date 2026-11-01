"""string_contains: byte substring test of a pattern against each string.

One thread per string. The thread scans the candidate start positions of its
string, compares the pattern byte by byte, and stops at the first occurrence.
The empty pattern matches every string. The pattern is small and read-only,
so its loads are served by L1.

Revision: the pattern index is clamped to [0, k - 1] and every byte position
q of `chars` is tested against [0, total), with total = offsets[n] the size
of `chars`, before the loads. Inside the loops 0 <= j < k and
s <= q < e <= total, so neither changes the result; they state the bound of
each access for the memory checker, which does not carry loop invariants.
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
    total = offsets[n]
    found = 0
    if k == 0:
        found = 1
    elif e - s >= k:
        last = e - k
        pos = s
        while pos <= last:
            j = 0
            while j < k:
                jc = min(max(j, 0), k - 1)  # == j
                q = pos + jc
                if q < 0 or q >= total:
                    break
                if chars[q] != pattern[jc]:
                    break
                j = jc + 1
            if j == k:
                found = 1
                break
            pos += 1
    match[i] = found


_kernel = cuda.jit(kernel)


def run(offsets, chars, pattern, match, n, k):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](offsets, chars, pattern, match, n, k)
