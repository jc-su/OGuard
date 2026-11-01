from numba import cuda, int64

ITEMS = 8             # rows per thread
TILE = 256 * ITEMS    # rows per block
PREF = TILE + 1       # prefix entries per chunk
_range = range        # the kernel's last argument is named range


def kernel(ts, values, out, n, range):
    # Each block handles one tile of rows. Every row's frame [lo, hi] comes
    # from two binary searches over ts. The union of the frames of the tile
    # is [first_lo, last_hi]; it is read in chunks of TILE rows (the tile
    # itself, then the chunks before it, then the chunks after it), each
    # chunk's prefix sums are built in shared memory, and every row adds the
    # sum of the chunk rows inside its frame. A frame of any length costs one
    # coalesced pass over it per block, and no global scratch is needed.
    s_pref = cuda.shared.array(PREF, int64)
    s_scan = cuda.shared.array(256, int64)
    s_bound = cuda.shared.array(2, int64)
    tid = int64(cuda.threadIdx.x)
    t0 = int64(cuda.blockIdx.x) * TILE
    if t0 >= n:
        return
    t1 = t0 + TILE
    if t1 > n:
        t1 = n
    base = t0 + tid * ITEMS
    lo = cuda.local.array(ITEMS, int64)
    hi = cuda.local.array(ITEMS, int64)
    acc = cuda.local.array(ITEMS, int64)
    for k in _range(ITEMS):
        i = base + k
        lo[k] = i
        hi[k] = i - 1
        acc[k] = 0
        if i < t1:
            x = ts[i]
            key = x - range
            l = int64(0)
            h = n
            while l < h:
                mid = (l + h) >> 1
                if ts[mid] < key:
                    l = mid + 1
                else:
                    h = mid
            lo[k] = l
            l = int64(0)
            h = n
            while l < h:
                mid = (l + h) >> 1
                if ts[mid] <= x:
                    l = mid + 1
                else:
                    h = mid
            hi[k] = l - 1
    if tid == 0:
        s_bound[0] = lo[0]
    last = t1 - 1 - t0
    if tid == last // ITEMS:
        s_bound[1] = hi[last - tid * ITEMS]
    cuda.syncthreads()
    first_lo = s_bound[0]
    last_hi = s_bound[1]
    before = (t0 - first_lo + TILE - 1) // TILE
    after = (last_hi + 1 - t1 + TILE - 1) // TILE
    for step in _range(1 + before + after):
        if step == 0:
            a = t0
            b = t1
        elif step <= before:
            b = t0 - (step - 1) * TILE
            a = b - TILE
            if a < first_lo:
                a = first_lo
        else:
            a = t1 + (step - before - 1) * TILE
            b = a + TILE
            if b > last_hi + 1:
                b = last_hi + 1
        # s_pref[k] = sum of values[a : a + k] for k in [0, b - a].
        cbase = a + tid * ITEMS
        local = int64(0)
        for k in _range(ITEMS):
            if cbase + k < b:
                local += values[cbase + k]
        s_scan[tid] = local
        cuda.syncthreads()
        d = 1
        while d < 256:
            v = int64(0)
            if tid >= d:
                v = s_scan[tid - d]
            cuda.syncthreads()
            s_scan[tid] += v
            cuda.syncthreads()
            d *= 2
        running = s_scan[tid] - local
        if tid == 0:
            s_pref[0] = 0
        for k in _range(ITEMS):
            if cbase + k < b:
                running += values[cbase + k]
                s_pref[tid * ITEMS + k + 1] = running
        cuda.syncthreads()
        for k in _range(ITEMS):
            if base + k < t1:
                j0 = lo[k] if lo[k] > a else a
                j1 = hi[k] if hi[k] < b - 1 else b - 1
                if j0 <= j1:
                    acc[k] += s_pref[j1 - a + 1] - s_pref[j0 - a]
        cuda.syncthreads()
    for k in _range(ITEMS):
        i = base + k
        if i < t1:
            out[i] = acc[k]


TYPES = (int64[:], int64[:], int64[:], int64, int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(ts, values, out, n, range):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](ts, values, out, n, range)
