from numba import cuda, types
import numpy as np

TYPES = (types.int64[:], types.int64, types.int64[:], types.int32[:],
         types.int64, types.int32[:], types.int32[:], types.int32[:])

def kernel(probe_keys, m, table_keys, table_rows, capacity, out_build, out_probe, out_count):
    i = cuda.grid(1)
    if i < m:
        key = probe_keys[i]
        hash_val = key & (capacity - 1)

        while table_keys[hash_val] != -1:
            if table_keys[hash_val] == key:
                idx = cuda.atomic.add(out_count, 0, 1)
                if idx < capacity:
                    out_build[idx] = table_rows[hash_val]
                    out_probe[idx] = i
                break
            hash_val = (hash_val + 1) & (capacity - 1)

def run(probe_keys: np.ndarray, table_keys: np.ndarray, table_rows: np.ndarray,
        out_build: np.ndarray, out_probe: np.ndarray, out_count: np.ndarray):
    m = probe_keys.size
    capacity = table_keys.size
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](probe_keys, np.int64(m), table_keys, table_rows,
                                              np.int64(capacity), out_build, out_probe, out_count)
