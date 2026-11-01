"""Numba CUDA kernel for q6_sum (TPC-H Q6).

Adds price[i] * discount[i] over the rows with
date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi,
and quantity[i] < qty_max, into result.

Precondition: the caller zeroes result before each launch.

Each block reduces its rows in shared memory and adds its partial sum
into result with one atomic add per block, instead of one atomic add
per matching row.
"""

from numba import cuda, float64, int32, int64

TYPES = (
    int32[:],
    float64[:],
    float64[:],
    float64[:],
    float64[:],
    int32,
    int32,
    float64,
    float64,
    float64,
    int64,
)

BLOCK = 256


def kernel(
    shipdate,
    discount,
    quantity,
    price,
    result,
    date_lo,
    date_hi,
    disc_lo,
    disc_hi,
    qty_max,
    n,
):
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    val = 0.0
    if i < n:
        sd = shipdate[i]
        d = discount[i]
        q = quantity[i]
        if (
            sd >= date_lo
            and sd < date_hi
            and d >= disc_lo
            and d <= disc_hi
            and q < qty_max
        ):
            val = price[i] * d

    partial = cuda.shared.array(BLOCK, dtype=float64)
    partial[tid] = val
    cuda.syncthreads()

    stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        cuda.atomic.add(result, 0, partial[0])


def run(
    shipdate,
    discount,
    quantity,
    price,
    result,
    date_lo,
    date_hi,
    disc_lo,
    disc_hi,
    qty_max,
    n,
):
    """Launch kernel on device arrays. result must already be zeroed."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](
        shipdate,
        discount,
        quantity,
        price,
        result,
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_max,
        n,
    )
