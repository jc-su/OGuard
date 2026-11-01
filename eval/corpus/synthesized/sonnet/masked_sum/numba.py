from numba import cuda, float64, int8, int64

BLOCK = 256


def kernel(values, mask, result, n):
    # Each thread sums its rows, each block reduces its threads with warp shuffles and shared memory,
    # and one thread per block adds the block sum to result[0].
    warp_sums = cuda.shared.array(32, float64)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + tid
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    s = 0.0
    for i in range(start, n, stride):
        if mask[i] != 0:
            s += values[i]
    d = 16
    while d > 0:
        s += cuda.shfl_down_sync(0xFFFFFFFF, s, d)
        d >>= 1
    if lane == 0:
        warp_sums[wid] = s
    cuda.syncthreads()
    if wid == 0:
        s = 0.0
        if lane < (cuda.blockDim.x >> 5):
            s = warp_sums[lane]
        d = 16
        while d > 0:
            s += cuda.shfl_down_sync(0xFFFFFFFF, s, d)
            d >>= 1
        if lane == 0:
            cuda.atomic.add(result, 0, s)


TYPES = (float64[:], int8[:], float64[:], int64)


def run(values, mask, result, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min(-(-n // BLOCK), 4 * sms))
    cuda.jit(kernel)[blocks, BLOCK](values, mask, result, n)
