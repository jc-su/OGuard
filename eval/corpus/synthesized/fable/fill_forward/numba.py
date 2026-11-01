from numba import cuda, int32, int64, uint8

ITEMS = 8             # rows per thread
TILE = 256 * ITEMS    # rows per block


def kernel(values, valid, out, n):
    # Each block fills one tile of rows. The carry into the tile is the last
    # valid row before it, found by scanning the validity bitmap backward 256
    # bytes at a time with a block reduction; a null run of any length costs
    # one coalesced pass over its bitmap bytes per block. The tile itself is a
    # block scan with the operator "take the right side when it is valid":
    # each thread scans its own consecutive rows, a shared-memory scan
    # combines the thread aggregates, and each thread rescans its rows.
    s_flag = cuda.shared.array(256, int32)
    s_val = cuda.shared.array(256, int64)
    s_red = cuda.shared.array(256, int64)
    tid = int64(cuda.threadIdx.x)
    t0 = int64(cuda.blockIdx.x) * TILE
    if t0 >= n:
        return
    t1 = t0 + TILE
    if t1 > n:
        t1 = n
    carry_row = int64(-1)
    if t0 > 0:
        top = (t0 - 1) >> 3
        q_hi = top
        while q_hi >= 0 and carry_row < 0:
            q = q_hi - tid
            best = int64(-1)
            if q >= 0:
                byte = int64(valid[q])
                if q == top:
                    byte &= (int64(1) << (t0 - q * 8)) - 1
                if byte != 0:
                    bit = 7
                    while (byte >> bit) & 1 == 0:
                        bit -= 1
                    best = q * 8 + bit
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
            carry_row = s_red[0]
            cuda.syncthreads()
            q_hi -= 256
    carry = int64(0)
    if carry_row >= 0:
        carry = values[carry_row]
    base = t0 + tid * ITEMS
    f = int32(0)
    agg = int64(0)
    for k in range(ITEMS):
        i = base + k
        if i < t1:
            if (int64(valid[i >> 3]) >> (i & 7)) & 1:
                f = int32(1)
                agg = values[i]
    s_flag[tid] = f
    s_val[tid] = agg
    cuda.syncthreads()
    d = 1
    while d < 256:
        pf = int32(0)
        pv = int64(0)
        if tid >= d:
            pf = s_flag[tid - d]
            pv = s_val[tid - d]
        cuda.syncthreads()
        if tid >= d and s_flag[tid] == 0:
            s_flag[tid] = pf
            s_val[tid] = pv
        cuda.syncthreads()
        d *= 2
    running = carry
    if tid > 0 and s_flag[tid - 1] != 0:
        running = s_val[tid - 1]
    for k in range(ITEMS):
        i = base + k
        if i < t1:
            if (int64(valid[i >> 3]) >> (i & 7)) & 1:
                running = values[i]
            out[i] = running


TYPES = (int64[:], uint8[:], int64[:], int64)


_dispatcher = None


def _jit():
    # cuda.jit(kernel), compiled once per process.
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = cuda.jit(kernel)
    return _dispatcher


def run(values, valid, out, n):
    blocks = max(1, (n + TILE - 1) // TILE)
    _jit()[blocks, 256](values, valid, out, n)
