from numba import cuda, types
import numpy as np

TYPES = (
    types.int64[:],  # keys
    types.int64[:],  # sorted
    types.int32[:],  # out
    types.int64,     # n
    types.int64,     # s
)


def kernel(keys, sorted_arr, out, n, s):
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]
        # Count how many elements in sorted < key
        count = 0
        for i in range(s):
            if sorted_arr[i] < key:
                count += 1
            else:
                break  # Array is sorted
        out[idx] = count


def run(keys: np.ndarray, sorted_arr: np.ndarray, out: np.ndarray, n: int, s: int) -> None:
    if n == 0:
        return

    d_keys = cuda.to_device(keys)
    d_sorted = cuda.to_device(sorted_arr)
    d_out = cuda.to_device(out)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_keys, d_sorted, d_out, n, s)

    d_out.copy_to_host(out)
