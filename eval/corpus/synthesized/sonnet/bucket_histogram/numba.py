from numba import cuda, float64, int32, int64

BLOCK = 256
SHARED_BINS = 1024


def kernel(values, bounds, counts, n, b):
    # A small bin count with enough rows per block is counted in a per-block histogram in shared
    # memory, which is added to counts once per block; otherwise rows add to counts directly.
    local = cuda.shared.array(SHARED_BINS, int32)
    tid = cuda.threadIdx.x
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + tid
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    bins = b + 1
    private = bins <= SHARED_BINS and n >= 4 * bins * cuda.gridDim.x
    if private:
        for k in range(tid, bins, cuda.blockDim.x):
            local[k] = 0
        cuda.syncthreads()
    for i in range(start, n, stride):
        v = values[i]
        lo = int64(0)
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if bounds[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        if private:
            cuda.atomic.add(local, lo, 1)
        else:
            cuda.atomic.add(counts, lo, 1)
    if private:
        cuda.syncthreads()
        for k in range(tid, bins, cuda.blockDim.x):
            c = local[k]
            if c != 0:
                cuda.atomic.add(counts, k, c)


TYPES = (float64[:], float64[:], int32[:], int64, int64)


def run(values, bounds, counts, n, b):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min(-(-n // BLOCK), 4 * sms))
    cuda.jit(kernel)[blocks, BLOCK](values, bounds, counts, n, b)
