"""sorted_contains: match[i] = 1 when keys[i] is one of the sorted, distinct members (Numba CUDA).

One thread per key finds the last member <= key by binary lifting and
compares it with the key: starting from pos = 0, it tries the steps
bit = 2^k, 2^(k-1), ..., 1 (2^k the highest power of two <= s) and moves to
pos + bit when that member exists and is <= key. The step count depends only
on s, so the lanes of a warp stay converged. The top levels of the search are
the same few members for every key and stay in L1; the kernel is bound by the
random reads of the bottom levels.

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A key is read and a match written
only under the two comparisons 0 <= i and i < n, and a member only under the
two comparisons 0 <= index and index < s of the very index used. The loop
runs on the step, which comes from s by bit operations, and not on a
comparison with s itself, so the compiler keeps these comparisons as they
are written.

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
    if i >= 0 and i < n:
        key = keys[i]
        # The highest power of two <= s, or 0 when s = 0.
        high = s
        high |= high >> 1
        high |= high >> 2
        high |= high >> 4
        high |= high >> 8
        high |= high >> 16
        high |= high >> 32
        bit = high - (high >> 1)
        pos = int64(0)  # members[pos] <= key, or pos = 0
        while bit > 0:
            probe = pos + bit
            if probe >= 0 and probe < s:
                if members[probe] <= key:
                    pos = probe
            bit >>= 1
        found = 0
        if pos >= 0 and pos < s:
            if members[pos] == key:
                found = 1
        match[i] = found


_kernel = cuda.jit(kernel)


def run(keys: DeviceNDArray, members: DeviceNDArray, match: DeviceNDArray, n: int, s: int) -> None:
    if n == 0:
        return
    grid = (n + BLOCK - 1) // BLOCK
    _kernel[grid, BLOCK](keys, members, match, n, s)
