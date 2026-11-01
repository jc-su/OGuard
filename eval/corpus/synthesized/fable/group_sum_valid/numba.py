from numba import cuda, int32, int64, uint8

TILE = 4096  # rows per block
MAX_GROUPS = 1024


def kernel(group, value, valid, counts, sums, n, g):
    # Each block accumulates a tile of rows into shared-memory copies of the
    # g accumulators and flushes them with one atomic per touched group.
    s_counts = cuda.shared.array(MAX_GROUPS, int64)
    s_sums = cuda.shared.array(MAX_GROUPS, int64)
    tid = int64(cuda.threadIdx.x)
    start = int64(cuda.blockIdx.x) * TILE
    end = start + TILE
    if end > n:
        end = n
    for k in range(tid, g, 256):
        s_counts[k] = 0
        s_sums[k] = 0
    cuda.syncthreads()
    for i in range(start + tid, end, 256):
        if (int64(valid[i >> 3]) >> (i & 7)) & 1:
            k = int64(group[i])
            cuda.atomic.add(s_counts, k, int64(1))
            cuda.atomic.add(s_sums, k, value[i])
    cuda.syncthreads()
    for k in range(tid, g, 256):
        c = s_counts[k]
        if c != 0:
            cuda.atomic.add(counts, k, c)
            cuda.atomic.add(sums, k, s_sums[k])


TYPES = (int32[:], int64[:], uint8[:], int64[:], int64[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(group, value, valid, counts, sums, n, g):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](group, value, valid, counts, sums, n, g)
