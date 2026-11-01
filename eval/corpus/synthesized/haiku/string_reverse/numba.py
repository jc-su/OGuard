from numba import cuda, types
import numpy as np

TYPES = (
    types.int32[:],  # offsets
    types.uint8[:],  # chars
    types.uint8[:],  # out
    types.int32,     # n
)


def kernel(offsets, chars, out, n):
    idx = cuda.grid(1)
    if idx < n:
        start = offsets[idx]
        end = offsets[idx + 1]

        # Reverse the string
        length = end - start
        for i in range(length):
            out[start + i] = chars[end - 1 - i]


def run(offsets: np.ndarray, chars: np.ndarray, out: np.ndarray, n: int) -> None:
    if n == 0:
        return

    d_offsets = cuda.to_device(offsets)
    d_chars = cuda.to_device(chars)
    d_out = cuda.to_device(out)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_offsets, d_chars, d_out, n)

    d_out.copy_to_host(out)
