"""Hash partition: part[i] = h(keys[i]) % p and counts[k] = rows in partition k."""

from numba import cuda, int32, int64, uint64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int64[:], int32[:], int32[:], int64, int64)

THREADS = 256
BLOCKS_PER_SM = 4
MAX_PARTITIONS = 1024
MULTIPLIER = 0x9E3779B97F4A7C15


def kernel(keys, part, counts, n, p):
    # Each block counts its rows in a shared histogram and adds the nonzero
    # bins to counts once at the end.
    hist = cuda.shared.array(MAX_PARTITIONS, int32)
    t = cuda.threadIdx.x
    k = t
    while k < p:
        hist[k] = 0
        k += cuda.blockDim.x
    cuda.syncthreads()

    # All operands are unsigned 64-bit, so the product wraps modulo 2^64.
    parts = uint64(p)
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < n:
        h = (uint64(keys[i]) * uint64(MULTIPLIER)) >> uint64(32)
        k = h % parts
        part[i] = k
        cuda.atomic.add(hist, k, 1)
        i += stride
    cuda.syncthreads()

    k = t
    while k < p:
        c = hist[k]
        if c != 0:
            cuda.atomic.add(counts, k, c)
        k += cuda.blockDim.x


_KERNEL = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    part: DeviceNDArray,
    counts: DeviceNDArray,
    n: int,
    p: int,
) -> None:
    if not 1 <= p <= MAX_PARTITIONS:
        raise ValueError(f"p must be in [1, {MAX_PARTITIONS}], got {p}")
    if n == 0:
        return
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM)
    _KERNEL[blocks, THREADS](keys, part, counts, n, p)
