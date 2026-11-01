from numba import cuda, int32, int64

ITEMS = 8             # rows per thread
TILE = 256 * ITEMS    # rows per block
INT64_MIN = -(1 << 63)


def kernel(values, first, out, n):
    # Each block scans one tile of rows. When the partition of the first tile
    # row starts before the tile, the block first reduces the values from that
    # partition start to the tile start (coalesced, one block reduction) to
    # get the carry. The tile itself is a segmented block scan: each thread
    # scans its own consecutive rows, a shared-memory scan combines the thread
    # aggregates, and each thread rescans its rows with its prefix.
    s_flag = cuda.shared.array(256, int32)
    s_val = cuda.shared.array(256, int64)
    tid = int64(cuda.threadIdx.x)
    t0 = int64(cuda.blockIdx.x) * TILE
    if t0 >= n:
        return
    t1 = t0 + TILE
    if t1 > n:
        t1 = n
    carry = INT64_MIN
    seg = int64(first[t0])
    if seg < t0:
        best = INT64_MIN
        for i in range(seg + tid, t0, 256):
            v = values[i]
            if v > best:
                best = v
        s_val[tid] = best
        cuda.syncthreads()
        s = 128
        while s > 0:
            if tid < s:
                o = s_val[tid + s]
                if o > s_val[tid]:
                    s_val[tid] = o
            cuda.syncthreads()
            s >>= 1
        carry = s_val[0]
        cuda.syncthreads()
    base = t0 + tid * ITEMS
    f = int32(0)
    agg = INT64_MIN
    for k in range(ITEMS):
        i = base + k
        if i < t1:
            v = values[i]
            if int64(first[i]) == i:
                f = int32(1)
                agg = v
            elif v > agg:
                agg = v
    s_flag[tid] = f
    s_val[tid] = agg
    cuda.syncthreads()
    d = 1
    while d < 256:
        pf = int32(0)
        pv = INT64_MIN
        if tid >= d:
            pf = s_flag[tid - d]
            pv = s_val[tid - d]
        cuda.syncthreads()
        if tid >= d and s_flag[tid] == 0:
            s_flag[tid] = pf
            if pv > s_val[tid]:
                s_val[tid] = pv
        cuda.syncthreads()
        d *= 2
    running = carry
    if tid > 0:
        if s_flag[tid - 1] != 0:
            running = s_val[tid - 1]
        elif s_val[tid - 1] > running:
            running = s_val[tid - 1]
    for k in range(ITEMS):
        i = base + k
        if i < t1:
            v = values[i]
            if int64(first[i]) == i or v > running:
                running = v
            out[i] = running


TYPES = (int64[:], int32[:], int64[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, first, out, n):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](values, first, out, n)
