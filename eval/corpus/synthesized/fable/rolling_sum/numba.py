"""rolling_sum: trailing window sum of width `window` (1 to 1024).

Each block produces 256 outputs. It loads its tile plus the window - 1
preceding values (at most 1279 doubles) into shared memory, computes an
inclusive prefix sum over them in up to five block-wide scan passes (warp
shuffle scan plus a cross-warp carry), and forms every output as the
difference of two prefix values, out[i] = P[i] - P[i - window]. That is two
shared memory reads per output instead of `window` reads. The prefix sums
are local to the block (at most 1279 terms), so the rounding error is of the
same order as that of summing the window directly.
"""

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64, int64)
THREADS = 256
MAX_WINDOW = 1024
SMEM = THREADS + MAX_WINDOW  # >= 256 + 1023 values per block


def kernel(values, out, n, window):
    s = cuda.shared.array(SMEM, float64)
    warp_sum = cuda.shared.array(8, float64)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5
    t0 = cuda.blockIdx.x * THREADS
    if t0 >= n:
        return
    tile = min(THREADS, n - t0)
    h = max(0, t0 - window + 1)  # first value the block needs
    e = t0 + tile - h  # number of values to scan

    l = tid
    while l < e:
        s[l] = values[h + l]
        l += THREADS
    cuda.syncthreads()

    # Inclusive prefix sum of s[0:e], 256 values per pass.
    carry = 0.0
    base = 0
    while base < e:
        l = base + tid
        x = s[l] if l < e else 0.0
        off = 1
        while off < 32:
            y = cuda.shfl_up_sync(0xFFFFFFFF, x, off)
            if lane >= off:
                x += y
            off <<= 1
        if lane == 31:
            warp_sum[wid] = x
        cuda.syncthreads()
        prefix = carry
        for w in range(wid):
            prefix += warp_sum[w]
        total = 0.0
        for w in range(8):
            total += warp_sum[w]
        if l < e:
            s[l] = x + prefix
        carry += total
        cuda.syncthreads()
        base += THREADS

    if tid < tile:
        i = t0 + tid
        l = i - h
        v = s[l]
        if l >= window:
            v -= s[l - window]
        out[i] = v


_kernel = cuda.jit(kernel)


def run(values, out, n, window):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](values, out, n, window)
