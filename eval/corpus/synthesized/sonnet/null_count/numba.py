from numba import cuda, int64, uint8

TYPES = (uint8[:], int64[:], int64)


def kernel(valid, count, n):
    # Count zero bits with popcount per byte, reduce in the warp, then the
    # block, and add once per block to the global counter.
    sh = cuda.shared.array(8, int64)
    tid = cuda.threadIdx.x
    nbytes = (n + 7) // 8
    rem = n & 7
    stride = cuda.gridDim.x * cuda.blockDim.x
    i = cuda.grid(1)
    acc = 0
    while i < nbytes:
        v = valid[i]
        if i == nbytes - 1 and rem != 0:
            acc += rem - cuda.popc(v & ((1 << rem) - 1))
        else:
            acc += 8 - cuda.popc(v)
        i += stride
    off = 16
    while off > 0:
        acc += cuda.shfl_down_sync(0xFFFFFFFF, acc, off)
        off >>= 1
    if (tid & 31) == 0:
        sh[tid >> 5] = acc
    cuda.syncthreads()
    if tid == 0:
        total = 0
        for w in range(8):
            total += sh[w]
        if total > 0:
            cuda.atomic.add(count, 0, total)


def run(valid, count, n):
    blocks = min(1024, max(1, (((n + 7) // 8) + 255) // 256))
    cuda.jit(kernel)[blocks, 256](valid, count, n)
