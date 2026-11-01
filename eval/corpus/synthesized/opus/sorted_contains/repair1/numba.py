"""sorted_contains: match[i] = 1 when keys[i] is one of the sorted, distinct members (Numba CUDA).

One thread per key runs a binary search for the last member <= key and
compares it with the key. The search has no branch on the comparison (the
step is a select), so its step count depends only on s and the lanes of a
warp stay converged. The top levels of the search are the same few members
for every key and stay in L1; the kernel is bound by the random reads of the
bottom levels.

Every member index is clamped to last = s - 1 where it is formed. The clamp
never changes the value (pos + half <= pos + length - 1 <= s - 1), but it
makes the bound of each access explicit, without a loop invariant that
relates pos and length.

The kernel has no grid-stride loop because LLVM unrolls such a loop behind a
64-bit division; run launches ceil(n / 256) blocks, which covers every key.
"""

import numpy as np
from numba import cuda, int8, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256

TYPES = (int64[:], int64[:], int8[:], int64, int64)


def kernel(keys: np.ndarray, members: np.ndarray, match: np.ndarray, n: int, s: int) -> None:
    i = cuda.grid(1)
    if i < n:
        key = keys[i]
        found = 0
        if s > 0:
            last = s - 1
            pos = int64(0)  # members[pos] <= key, or pos = 0
            length = s
            while length > 1:
                half = length >> 1
                probe = min(pos + half, last)
                if members[probe] <= key:
                    pos = probe
                length -= half
            if members[min(pos, last)] == key:
                found = 1
        match[i] = found


_kernel = cuda.jit(kernel)


def run(keys: DeviceNDArray, members: DeviceNDArray, match: DeviceNDArray, n: int, s: int) -> None:
    if n == 0:
        return
    grid = (n + BLOCK - 1) // BLOCK
    _kernel[grid, BLOCK](keys, members, match, n, s)
