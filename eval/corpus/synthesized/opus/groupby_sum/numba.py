"""Group-by sum: hash aggregation of `value` per `group`.

The table uses open addressing with linear probing. The home slot of a group
is `fmix32(group) & (capacity - 1)`. Slot `s` holds a group in `table_keys[s]`
and the sum of its values in `table_sums[s]`.

Each warp first adds up the values of the lanes that hold the same group, so
only one lane per distinct group in the warp touches the table. This keeps the
number of atomic adds low when there are few groups.

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
def hash_group(group: np.int32) -> np.uint64:
    """Return the MurmurHash3 fmix32 finalizer of the 32-bit group."""
    h = types.uint64(types.uint32(group))
    h ^= h >> types.uint64(16)
    h = (h * types.uint64(0x85EBCA6B)) & types.uint64(0xFFFFFFFF)
    h ^= h >> types.uint64(13)
    h = (h * types.uint64(0xC2B2AE35)) & types.uint64(0xFFFFFFFF)
    h ^= h >> types.uint64(16)
    return h


@cuda.jit(device=True)
def warp_lanes_mask() -> np.uint32:
    """Return the mask of the lanes that exist in the calling warp of a 1-D block."""
    lane = cuda.threadIdx.x % 32  # ty: ignore[unsupported-operator]
    lanes = min(32, cuda.blockDim.x - (cuda.threadIdx.x - lane))
    return types.uint32((1 << lanes) - 1)


def kernel(
    group: npt.NDArray[np.int32],
    value: npt.NDArray[np.float64],
    table_keys: npt.NDArray[np.int32],
    table_sums: npt.NDArray[np.float64],
    capacity: int,
    n: int,
) -> None:
    lane = cuda.threadIdx.x % 32  # ty: ignore[unsupported-operator]
    lanes_mask = warp_lanes_mask()
    lanes_below = (1 << lane) - 1
    lanes_above = (0xFFFFFFFE << lane) & 0xFFFFFFFF
    mask = types.uint64(capacity - 1)
    stride = cuda.gridsize(1)
    # All lanes of a warp run the same number of iterations, so every lane
    # takes part in the warp vote.
    warp_start = cuda.grid(1) - lane
    while warp_start < n:
        j = warp_start + lane
        valid = j < n
        active = types.uint32(cuda.ballot_sync(lanes_mask, valid))
        if valid:
            g = group[j]
            total = value[j]
            # Tree reduction over the lanes that hold the same group. In
            # round r, a lane whose rank among its peers has bit r set gives
            # its partial sum to the next lower remaining peer and stops.
            peers = cuda.match_any_sync(active, g) & 0xFFFFFFFF  # ty: ignore[too-many-positional-arguments, unsupported-operator]
            rank: int = cuda.popc(peers & lanes_below)  # ty: ignore[too-many-positional-arguments, invalid-assignment]
            remaining = peers & lanes_above
            while cuda.any_sync(active, remaining != 0):
                source = cuda.ffs(remaining) - 1  # ty: ignore[too-many-positional-arguments, unsupported-operator]
                if source < 0:
                    source = lane
                partial = cuda.shfl_sync(active, total, source)
                if source != lane:
                    total += partial
                done = cuda.ballot_sync(active, (rank & 1) != 0) & 0xFFFFFFFF
                remaining &= ~done
                rank >>= 1
            if (peers & lanes_below) == 0:
                slot = hash_group(g) & mask
                while True:
                    # A slot changes only from EMPTY to a group, so a stale
                    # read of a group is never wrong; the CAS settles a stale
                    # read of EMPTY.
                    slot_key = table_keys[slot]
                    if slot_key == EMPTY:
                        slot_key = cuda.atomic.cas(table_keys, slot, EMPTY, g)  # ty: ignore[too-many-positional-arguments]
                    if slot_key == EMPTY or slot_key == g:
                        cuda.atomic.add(table_sums, slot, total)  # ty: ignore[too-many-positional-arguments]
                        break
                    slot = (slot + types.uint64(1)) & mask
        warp_start += stride


TYPES: tuple[types.Type, ...] = (
    types.int32[:],
    types.float64[:],
    types.int32[:],
    types.float64[:],
    types.int64,
    types.int64,
)

_kernel: CUDADispatcher = cuda.jit(kernel)


def run(
    group: DeviceNDArray,
    value: DeviceNDArray,
    table_keys: DeviceNDArray,
    table_sums: DeviceNDArray,
    capacity: int,
    n: int,
) -> None:
    """Add value[0:n] into the table; table_keys must hold EMPTY, table_sums 0."""
    if capacity < 1 or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two")
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid = max(1, min((n + BLOCK - 1) // BLOCK, BLOCKS_PER_SM * sms))
    _kernel[grid, BLOCK](group, value, table_keys, table_sums, capacity, n)
