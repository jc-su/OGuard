"""dictionary_decode: out[i] = dictionary[codes[i]] (Numba CUDA).

One thread per row: a coalesced read of the code, a gather from the
dictionary (a small dictionary stays in L1/L2), and a coalesced store. The
kernel has no grid-stride loop because LLVM unrolls such a loop behind a
64-bit division that costs more than the row itself; run launches
ceil(n / 256) blocks, which covers every row.
"""

import numpy as np
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256

TYPES = (int32[:], float64[:], float64[:], int64, int64)


def kernel(codes: np.ndarray, dictionary: np.ndarray, out: np.ndarray, n: int, d: int) -> None:
    i = cuda.grid(1)
    if i < n:
        out[i] = dictionary[codes[i]]


_kernel = cuda.jit(kernel)


def run(
    codes: DeviceNDArray, dictionary: DeviceNDArray, out: DeviceNDArray, n: int, d: int
) -> None:
    if n == 0:
        return
    grid = (n + BLOCK - 1) // BLOCK
    _kernel[grid, BLOCK](codes, dictionary, out, n, d)
