from numba import cuda, types
import numpy as np

TYPES = (types.int64[:], types.int64, types.int64[:], types.int32[:])

def kernel(keys, capacity, table_keys, table_rows):
    i = cuda.grid(1)
    if i < keys.size:
        key = keys[i]
        hash_val = key & (capacity - 1)

        while table_keys[hash_val] != -1 and table_keys[hash_val] != key:
            hash_val = (hash_val + 1) & (capacity - 1)

        table_keys[hash_val] = key
        table_rows[hash_val] = types.int32(i)

def run(keys: np.ndarray, table_keys: np.ndarray, table_rows: np.ndarray):
    n = keys.size
    capacity = table_keys.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](keys, np.int64(capacity), table_keys, table_rows)
