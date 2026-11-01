"""Hash build: insert each key into an open-addressing table with linear probing.

Slot `s` holds a key in `table_keys[s]` and the key's row id in `table_rows[s]`.
The home slot of a key is `fmix64(key) & (capacity - 1)`. A key goes into the
first empty slot at or after its home slot, wrapping around the table end.
`hash_probe` uses the same hash and probe order.

ty checks Numba CUDA intrinsics against their Python stub classes in
numba.cuda.stubs, so the calls to them carry `ty: ignore` comments.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher

EMPTY = -1
BLOCK = 256
BLOCKS_PER_SM = 8


@cuda.jit(device=True)
def hash_key(key: np.int64) -> np.uint64:
    """Return the MurmurHash3 fmix64 finalizer of the 64-bit key."""
    h = types.uint64(key)
    h ^= h >> types.uint64(33)
    h *= types.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> types.uint64(33)
    h *= types.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> types.uint64(33)
    return h


def kernel(
    keys: npt.NDArray[np.int64],
    table_keys: npt.NDArray[np.int64],
    table_rows: npt.NDArray[np.int32],
    capacity: int,
    n: int,
) -> None:
    mask = types.uint64(capacity - 1)
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < n:
        key = keys[i]
        slot = hash_key(key) & mask
        while True:
            # A slot changes only from EMPTY to a key, so a stale read of a
            # key is never wrong; the CAS settles a stale read of EMPTY.
            if table_keys[slot] == EMPTY:
                previous = cuda.atomic.cas(table_keys, slot, EMPTY, key)  # ty: ignore[too-many-positional-arguments]
                if previous == EMPTY:
                    table_rows[slot] = types.int32(i)
                    break
            slot = (slot + types.uint64(1)) & mask
        i += stride


TYPES: tuple[types.Type, ...] = (
    types.int64[:],
    types.int64[:],
    types.int32[:],
    types.int64,
    types.int64,
)

_kernel: CUDADispatcher = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    table_keys: DeviceNDArray,
    table_rows: DeviceNDArray,
    capacity: int,
    n: int,
) -> None:
    """Insert keys[0:n] into the table; every table_keys slot must hold EMPTY."""
    if capacity < max(1, 2 * n) or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two >= 2 * {n}")
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid = max(1, min((n + BLOCK - 1) // BLOCK, BLOCKS_PER_SM * sms))
    _kernel[grid, BLOCK](keys, table_keys, table_rows, capacity, n)
