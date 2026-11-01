from numba import cuda, int32, int64

TILE = 4096  # rows per block


def kernel(values, row, n):
    # Each block reduces a tile to its (minimum, smallest row) pair, then
    # thread 0 merges the pair into row[0] with a compare-and-swap loop that
    # keeps the pair with the smaller value, or the smaller row on a tie.
    s_val = cuda.shared.array(256, int64)
    s_row = cuda.shared.array(256, int64)
    tid = int64(cuda.threadIdx.x)
    start = int64(cuda.blockIdx.x) * TILE
    end = start + TILE
    if end > n:
        end = n
    bv = int64(0)
    br = int64(-1)
    for i in range(start + tid, end, 256):
        v = values[i]
        if br < 0 or v < bv:
            bv = v
            br = i
    s_val[tid] = bv
    s_row[tid] = br
    cuda.syncthreads()
    s = 128
    while s > 0:
        if tid < s:
            ov = s_val[tid + s]
            orow = s_row[tid + s]
            if orow >= 0 and (br < 0 or ov < bv or (ov == bv and orow < br)):
                bv = ov
                br = orow
                s_val[tid] = bv
                s_row[tid] = br
        cuda.syncthreads()
        s >>= 1
    if tid == 0 and br >= 0:
        cur = int64(cuda.atomic.add(row, 0, int32(0)))
        while True:
            if cur >= 0:
                cv = values[cur]
                if cv < bv or (cv == bv and cur < br):
                    break
            old = int64(cuda.atomic.compare_and_swap(row, int32(cur), int32(br)))
            if old == cur:
                break
            cur = old


TYPES = (int64[:], int32[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, row, n):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](values, row, n)
