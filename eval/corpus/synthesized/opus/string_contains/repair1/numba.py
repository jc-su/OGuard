"""string_contains: match[i] = 1 when the byte string i contains the pattern (Numba CUDA).

Each block handles BLOCK consecutive strings in two phases, so that both
short and long strings, and any mix of them, keep the lanes busy:

1. Thread per string: a thread tests the candidate start positions of a
   short string (at most SHORT_BYTES bytes) one after another, and a long
   string is left for step 2.
2. Warp per string: a warp takes the long strings of its own 32 lanes, found
   with a ballot, one after another. A step of a warp covers STEPS * 32
   consecutive start positions; each lane loads the first bytes of its STEPS
   candidates together (the lanes read consecutive bytes), the warp votes with
   a ballot, and it stops at the first match.

A candidate is compared further only when its first byte matches, and the
comparison stops at the first mismatch. The empty pattern occurs in every
string.

Bounds: a string index is compared with n before it reads `offsets`; a byte
of string i is read at a position p only when p < offsets[i + 1] (and p is at
least offsets[i] >= 0), which the candidate and pattern bounds already imply;
a pattern byte j is read only when j < k. The long strings stay with their
own warp instead of going through a shared-memory queue, so the kernel uses
no shared memory, and a warp's step start is the same in every lane, so the
loop bound is a plain comparison in each lane. The loops that index memory are
while loops with the bound in their condition.
"""

import numpy as np
from numba import cuda, int8, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
SHORT_BYTES = 128
STEPS = 8
FULL_MASK = 0xFFFFFFFF

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)


def kernel(
    offsets: np.ndarray,
    chars: np.ndarray,
    pattern: np.ndarray,
    match: np.ndarray,
    n: int,
    k: int,
) -> None:
    head = cuda.local.array(STEPS, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    s0 = int64(cuda.blockIdx.x) * BLOCK
    first = 0
    if k > 0:
        first = pattern[0]

    # 1. Thread per short string; the long ones wait for step 2.
    long_string = False
    s = s0 + tid
    if s < n:
        start = int64(offsets[s])
        end = int64(offsets[s + 1])
        if k > 0 and end - start > SHORT_BYTES:
            long_string = True
        else:
            found = k == 0
            last = end - k  # the last candidate start
            pos = start
            while not found and pos <= last:
                if pos < end and chars[pos] == first:
                    j = 1
                    while j < k:
                        c = pos + j
                        if c >= end or chars[c] != pattern[j]:
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
        if ls < n:
            start = int64(offsets[ls])
            end = int64(offsets[ls + 1])
            last = end - k  # the last candidate start
            base = start  # the same in every lane
            found = False  # the same in every lane
            while not found and base <= last:
                u = 0
                while u < STEPS:
                    p = base + u * 32 + lane
                    head[u] = -1
                    if p <= last and p < end:
                        head[u] = chars[p]
                    u += 1
                hit = False
                u = 0
                while u < STEPS:
                    if not hit and head[u] == first:
                        p = base + u * 32 + lane
                        j = 1
                        while j < k:
                            c = p + j
                            if c >= end or chars[c] != pattern[j]:
                                break
                            j += 1
                        hit = j == k
                    u += 1
                found = cuda.ballot_sync(FULL_MASK, hit) != 0
                base += STEPS * 32
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
