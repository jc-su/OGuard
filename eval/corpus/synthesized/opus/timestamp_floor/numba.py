"""timestamp_floor: out[i] = largest multiple of unit that is <= ts[i] (rounds toward
negative infinity, also before 1970).

Python integer division (//) is floor division, and Numba keeps that meaning on
int64, so out = (ts // unit) * unit. |ts| <= 2^62 and unit <= 2^40, so the product
stays in range.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32

TYPES = (
    types.int64[:],  # ts
    types.int64[:],  # out
    types.int64,  # n
    types.int64,  # unit
)


def kernel(ts, out, n, unit):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        out[i] = (ts[i] // unit) * unit
        i += stride


_kernel = cuda.jit(kernel)


def run(ts, out, n, unit):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](ts, out, n, unit)
