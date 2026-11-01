"""decimal_rescale: change the scale of decimal64 values by 10^shift.

shift >= 0: out = values * 10^shift, or out = 0 and overflow = 1 when the product
leaves the int64 range. For p = 10^shift > 1, p does not divide 2^63, so |v| * p
fits exactly when |v| <= INT64_MAX // p, for either sign.
shift < 0: out = values / 10^-shift rounded to nearest, ties away from zero. With
floor division q = v // d and rem = v - q * d in [0, d), the result is q + 1 when
2 * rem > d, and on a tie (2 * rem == d) it is q + 1 for v >= 0 and q for v < 0.
rem < d <= 10^18, so 2 * rem does not overflow.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32
INT64_MAX = 9223372036854775807

TYPES = (
    types.int64[:],  # values
    types.int64[:],  # out
    types.int8[:],  # overflow
    types.int64,  # n
    types.int64,  # shift
)


def kernel(values, out, overflow, n, shift):
    k = shift if shift >= 0 else -shift
    p = types.int64(1)
    for _ in range(k):
        p *= 10
    limit = INT64_MAX // p

    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        v = values[i]
        if shift >= 0:
            if shift == 0 or (v >= -limit and v <= limit):
                out[i] = v * p
                overflow[i] = 0
            else:
                out[i] = 0
                overflow[i] = 1
        else:
            q = v // p
            twice = 2 * (v - q * p)
            if twice > p or (twice == p and v >= 0):
                q += 1
            out[i] = q
            overflow[i] = 0
        i += stride


_kernel = cuda.jit(kernel)


def run(values, out, overflow, n, shift):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](values, out, overflow, n, shift)
