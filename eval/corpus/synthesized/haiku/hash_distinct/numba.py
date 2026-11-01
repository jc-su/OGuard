from numba import cuda, types

def kernel(keys, table_keys, table_rows, n, capacity):
    i = cuda.grid(1)
    if i < n:
        key = keys[i]
        h = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        slot = h & (capacity - 1)
        
        while table_keys[slot] != -1 and table_keys[slot] != key:
            slot = (slot + 1) & (capacity - 1)
        
        if table_keys[slot] == -1:
            table_keys[slot] = key
            table_rows[slot] = i
        elif table_rows[slot] > i:
            table_rows[slot] = i

TYPES = (types.int64[:], types.int64[:], types.int32[:], types.int64, types.int64)

def run(keys, table_keys, table_rows, n, capacity):
    cuda.jit(kernel)[(n + 255) // 256, 256](keys, table_keys, table_rows, n, capacity)
