from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.float64[:], types.int64, types.int32[:], types.float64[:])

def kernel(group, value, capacity, table_keys, table_sums):
    i = cuda.grid(1)
    if i < group.size:
        key = group[i]
        hash_val = key & (capacity - 1)

        while table_keys[hash_val] != -1 and table_keys[hash_val] != key:
            hash_val = (hash_val + 1) & (capacity - 1)

        if table_keys[hash_val] == -1:
            table_keys[hash_val] = key

        cuda.atomic.add(table_sums, hash_val, value[i])

def run(group: np.ndarray, value: np.ndarray, table_keys: np.ndarray, table_sums: np.ndarray):
    capacity = table_keys.size
    threads_per_block = 256
    blocks = (group.size + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](group, value, np.int64(capacity), table_keys, table_sums)
