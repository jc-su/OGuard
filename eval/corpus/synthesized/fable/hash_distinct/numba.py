from numba import cuda, int32, int64, uint64


def kernel(keys, table_keys, table_rows, n, capacity):
    # One thread per key: linear probing from the hash slot. A slot that holds
    # -1 is claimed with a compare-and-swap; a slot never changes again once
    # it holds a key, so a plain read that returns a key is final. Whichever
    # thread finds the key's slot lowers table_rows[slot] with an atomic min.
    i = int64(cuda.blockIdx.x) * int64(cuda.blockDim.x) + int64(cuda.threadIdx.x)
    if i < n:
        key = keys[i]
        h = (uint64(key) * uint64(0x9E3779B97F4A7C15)) >> uint64(32)
        mask = capacity - 1
        slot = int64(h) & mask
        while True:
            cur = table_keys[slot]
            if cur == -1:
                cur = cuda.atomic.cas(table_keys, slot, int64(-1), key)
            if cur == -1 or cur == key:
                cuda.atomic.min(table_rows, slot, int32(i))
                break
            slot = (slot + 1) & mask


TYPES = (int64[:], int64[:], int32[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(keys, table_keys, table_rows, n, capacity):
    blocks = max(1, (n + 255) // 256)
    _jit()[blocks, 256](keys, table_keys, table_rows, n, capacity)
