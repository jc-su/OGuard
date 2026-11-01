"""string_contains: match[i] = 1 when the byte string i contains the pattern (Numba CUDA).

Each block handles BLOCK consecutive strings in two phases, so that both
short and long strings, and any mix of them, keep the lanes busy:

1. Thread per string: a thread tests the candidate start positions of a
   short string (at most SHORT_BYTES bytes) one after another. A long string
   goes onto a shared-memory queue instead.
2. Warp per string: the block's warps take the queued long strings. A step of
   a warp covers STEPS * 32 consecutive start positions; each lane loads the
   first bytes of its STEPS candidates together (the lanes read consecutive
   bytes), the warp votes with a ballot, and it stops at the first match.

A candidate is compared further only when its first byte matches, and the
comparison stops at the first mismatch. The empty pattern occurs in every
string.
"""

import numpy as np
from numba import cuda, int8, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
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
    long_queue = cuda.shared.array(BLOCK, int32)
    long_count = cuda.shared.array(1, int32)
    head = cuda.local.array(STEPS, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    s0 = int64(cuda.blockIdx.x) * BLOCK
    first = 0
    if k > 0:
        first = pattern[0]
    if tid == 0:
        long_count[0] = 0
    cuda.syncthreads()

    # 1. Thread per short string; queue the long ones.
    s = s0 + tid
    if s < n:
        start = offsets[s]
        end = offsets[s + 1]
        if k > 0 and end - start > SHORT_BYTES:
            long_queue[cuda.atomic.add(long_count, 0, 1)] = tid
        else:
            found = k == 0
            pos = start
            while not found and pos <= end - k:
                if chars[pos] == first:
                    j = 1
                    while j < k and chars[pos + j] == pattern[j]:
                        j += 1
                    found = j == k
                pos += 1
            match[s] = 1 if found else 0
    cuda.syncthreads()

    # 2. Warp per long string (k >= 1 here).
    queued = long_count[0]
    for q in range(warp, queued, WARPS):
        ls = s0 + long_queue[q]
        last = offsets[ls + 1] - k  # last candidate start position
        pos = offsets[ls] + lane
        found = False  # the same in every lane
        while not found and cuda.any_sync(FULL_MASK, pos <= last):
            for u in range(STEPS):
                p = pos + u * 32
                head[u] = chars[p] if p <= last else -1
            hit = False
            for u in range(STEPS):
                if not hit and head[u] == first:
                    p = pos + u * 32
                    j = 1
                    while j < k and chars[p + j] == pattern[j]:
                        j += 1
                    hit = j == k
            found = cuda.ballot_sync(FULL_MASK, hit) != 0
            pos += STEPS * 32
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
