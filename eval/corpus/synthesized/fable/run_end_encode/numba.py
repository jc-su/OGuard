from numba import cuda, int32, int64

ITEMS = 16            # rows per thread
TILE = 256 * ITEMS    # rows per block


def kernel(values, run_ends, run_values, count, n):
    # Each block encodes one tile of rows. A row is a run end when it is the
    # last row or differs from the next one. The index of a run in the output
    # is the number of run ends before it; within the tile that is a block
    # scan, and across tiles it comes from count[0], which holds the number of
    # runs written so far. A tile that holds run ends finds the last run end
    # e before it, waits until the tile owning e has published (count[0] = P
    # with run_ends[P - 1] = e + 1), writes its runs at P, P + 1, ..., and
    # then adds its run count to count[0]. Tiles without run ends write
    # nothing and wait for nothing. Blocks are dispatched in index order, so
    # the tile a block waits for is always resident or finished.
    s_cnt = cuda.shared.array(256, int32)
    s_red = cuda.shared.array(256, int64)
    s_base = cuda.shared.array(1, int64)
    tid = int64(cuda.threadIdx.x)
    t0 = int64(cuda.blockIdx.x) * TILE
    if t0 >= n:
        return
    t1 = t0 + TILE
    if t1 > n:
        t1 = n
    first = t0 + tid * ITEMS
    cnt = int32(0)
    for k in range(ITEMS):
        i = first + k
        if i < t1:
            if i == n - 1 or values[i] != values[i + 1]:
                cnt += 1
    s_cnt[tid] = cnt
    cuda.syncthreads()
    d = 1
    while d < 256:
        v = int32(0)
        if tid >= d:
            v = s_cnt[tid - d]
        cuda.syncthreads()
        s_cnt[tid] += v
        cuda.syncthreads()
        d *= 2
    excl = int64(s_cnt[tid] - cnt)
    total = s_cnt[255]
    if total == 0:
        return
    # Last run end before the tile.
    e = int64(-1)
    b = t0
    while b > 0 and e < 0:
        a = b - TILE
        if a < 0:
            a = 0
        best = int64(-1)
        for i in range(a + tid, b, 256):
            if values[i] != values[i + 1]:
                best = i
        s_red[tid] = best
        cuda.syncthreads()
        s = 128
        while s > 0:
            if tid < s:
                o = s_red[tid + s]
                if o > s_red[tid]:
                    s_red[tid] = o
            cuda.syncthreads()
            s >>= 1
        e = s_red[0]
        cuda.syncthreads()
        b = a
    if tid == 0:
        base = int64(0)
        if e >= 0:
            while True:
                p = int64(cuda.atomic.add(count, 0, int32(0)))
                if p > 0 and int64(cuda.atomic.add(run_ends, p - 1, int32(0))) == e + 1:
                    base = p
                    break
        s_base[0] = base
    cuda.syncthreads()
    j = s_base[0] + excl
    for k in range(ITEMS):
        i = first + k
        if i < t1:
            if i == n - 1 or values[i] != values[i + 1]:
                run_ends[j] = int32(i + 1)
                run_values[j] = values[i]
                j += 1
    cuda.threadfence()
    cuda.syncthreads()
    if tid == 0:
        cuda.atomic.add(count, 0, total)


TYPES = (int64[:], int32[:], int64[:], int32[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, run_ends, run_values, count, n):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](values, run_ends, run_values, count, n)
