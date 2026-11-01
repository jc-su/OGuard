from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.float64[:], types.float64[:], types.float64[:],
         types.int32, types.int32, types.float64, types.float64, types.float64, types.float64[:])

def kernel(shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result):
    i = cuda.grid(1)
    if i < shipdate.size:
        if date_lo <= shipdate[i] < date_hi and disc_lo <= discount[i] <= disc_hi and quantity[i] < qty_max:
            cuda.atomic.add(result, 0, price[i] * discount[i])

def run(shipdate: np.ndarray, discount: np.ndarray, quantity: np.ndarray, price: np.ndarray,
        date_lo: int, date_hi: int, disc_lo: float, disc_hi: float, qty_max: float, result: np.ndarray):
    n = shipdate.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](shipdate, discount, quantity, price,
                                              np.int32(date_lo), np.int32(date_hi),
                                              np.float64(disc_lo), np.float64(disc_hi), np.float64(qty_max), result)
