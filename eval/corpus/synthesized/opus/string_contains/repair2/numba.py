"""string_contains: match[i] = 1 when the byte string i contains the pattern (Numba CUDA).

Each block handles BLOCK consecutive strings in two phases, so that both
short and long strings, and any mix of them, keep the lanes busy:

1. Thread per string: a thread tests the candidate start positions of a
   short string (at most SHORT_BYTES bytes) one after another, and a long
   string is left for step 2.
2. Warp per string: a warp takes the long strings of its own 32 lanes, found
   with a ballot, one after another. A step of a warp tests 32 consecutive
   start positions, one per lane (the lanes read consecutive bytes), the warp
   votes with a ballot, and it stops at the first match.

A candidate is compared byte by byte and the comparison stops at the first
mismatch. The empty pattern occurs in every string.

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A string index s is read from
`offsets` and written to `match` only under 0 <= s and s < n (so s + 1 < n + 1
too), and the column length is offsets[n]. A byte position and a pattern
position are masked with POSITION_MASK = 2^31 - 1, which changes no position
of a column with int32 offsets and keeps the index at least 0; a byte at
position c is then read only under c < offsets[n] and c < the end of its
string, and a pattern byte j only under j < k. Each index is the very value
that its comparisons test, and the mask keeps the compiler from turning it
into a pointer that steps with the loop. The kernel nests at most three
loops, and has no per-thread array and no shared memory.
"""

import numpy as np
from numba import cuda, int8, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
SHORT_BYTES = 128
FULL_MASK = 0xFFFFFFFF
POSITION_MASK = 0x7FFFFFFF  # byte and pattern positions are below 2^31

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)


def kernel(
    offsets: np.ndarray,
    chars: np.ndarray,
    pattern: np.ndarray,
    match: np.ndarray,
    n: int,
    k: int,
) -> None:
    tid = cuda.threadIdx.x
    lane = tid & 31
    s = int64(cuda.blockIdx.x) * BLOCK + tid
    size = int64(0)  # the length of the column, offsets[n]
    if n >= 0:
        size = int64(offsets[n])

    # 1. Thread per short string; the long ones wait for step 2.
    long_string = False
    if s >= 0 and s < n:
        start = int64(offsets[s])
        end = int64(offsets[s + 1])
        if k > 0 and end - start > SHORT_BYTES:
            long_string = True
        else:
            found = k == 0
            last = end - k  # the last candidate start
            pos = start
            while not found and pos <= last:
                j = 0
                while j < k:
                    c = (pos + j) & POSITION_MASK
                    t = j & POSITION_MASK
                    if c >= size or c >= end or t >= k:
                        break
                    if chars[c] != pattern[t]:
                        break
                    j += 1
                found = j == k
                pos += 1
            match[s] = 1 if found else 0

    # 2. Warp per long string (k >= 1 here): the long strings of this warp's lanes.
    pending = cuda.ballot_sync(FULL_MASK, long_string)
    while pending != 0:
        ls = s - lane + ((cuda.ffs(pending) - 1) & 31)  # the lowest pending lane
        pending &= pending - 1
        found = False  # the same in every lane
        if ls >= 0 and ls < n:
            start = int64(offsets[ls])
            end = int64(offsets[ls + 1])
            last = end - k  # the last candidate start
            base = start  # the same in every lane
            while not found and base <= last:
                p = base + lane
                hit = False
                if p <= last:
                    j = 0
                    while j < k:
                        c = (p + j) & POSITION_MASK
                        t = j & POSITION_MASK
                        if c >= size or c >= end or t >= k:
                            break
                        if chars[c] != pattern[t]:
                            break
                        j += 1
                    hit = j == k
                found = cuda.ballot_sync(FULL_MASK, hit) != 0
                base += 32
            if lane == 0:
                match[ls] = 1 if found else 0


_kernel = cuda.jit(kernel)


def run(
    offsets: DeviceNDArray,
    chars: DeviceNDArray,
    pattern: DeviceNDArray,
    match: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    if n == 0:
        return
    grid = (n + BLOCK - 1) // BLOCK
    _kernel[grid, BLOCK](offsets, chars, pattern, match, n, k)
