from numba import cuda, int32, int64

TYPES = (int32[:], int32[:], int64, int64)


def kernel(source, out, rows, cols):
    # 32 x 32 tiles staged in shared memory (padded to avoid bank conflicts);
    # a block of 256 threads moves 8 rows of the tile at a time.
    tile = cuda.shared.array((32, 33), int32)
    t = cuda.threadIdx.x
    tx = t & 31
    ty = t >> 5
    tc = (cols + 31) >> 5
    ntiles = ((rows + 31) >> 5) * tc
    b = cuda.blockIdx.x
    while b < ntiles:
        r0 = (b // tc) * 32
        c0 = (b % tc) * 32
        for k in range(0, 32, 8):
            r = r0 + ty + k
            c = c0 + tx
            if r < rows and c < cols:
                tile[ty + k, tx] = source[r * cols + c]
        cuda.syncthreads()
        for k in range(0, 32, 8):
            c = c0 + ty + k
            r = r0 + tx
            if r < rows and c < cols:
                out[c * rows + r] = tile[tx, ty + k]
        cuda.syncthreads()
        b += cuda.gridDim.x


def run(source, out, rows, cols):
    ntiles = ((rows + 31) // 32) * ((cols + 31) // 32)
    cuda.jit(kernel)[max(1, ntiles), 256](source, out, rows, cols)
