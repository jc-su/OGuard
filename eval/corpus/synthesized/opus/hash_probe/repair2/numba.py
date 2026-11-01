"""Hash probe: look up each probe key in the table that hash_build wrote.

The table uses open addressing with linear probing. The home slot of a key is
`fmix64(key) & (capacity - 1)`, the same as in hash_build. The build keys are
distinct, so each probe row matches at most one build row.

Each warp appends its matches with one atomic add on `out_count`: the lanes
with a match take consecutive positions in their lane order. A lane writes its
pair only when its position is in [0, out_cap). If `out_count` holds 0 on
entry, no position is negative, so the check skips only the pairs at
positions from out_cap on. The check keeps each store inside `out_build` and
`out_probe` for any value in `out_count` on entry.

The same store also checks the probe row itself: j < m. A lane has a match
only if j < m, so the check skips no pair. The check puts the bound of the
value stored in `out_probe` next to the store, where the value is known to be
a row id of `probe_keys`. The first test of j < m is in the lookup, which
reaches the store only through the warp vote.

The device functions have `forceinline=True`, so the kernel has no calls
and returns no value through a pointer to a local variable of the caller.

ty checks Numba CUDA intrinsics against their Python stub classes in
numba.cuda.stubs, so the calls to them carry `ty: ignore` comments. The
signatures of `cuda.jit` that ty sees do not name `forceinline`, which
numba-cuda accepts through its keyword arguments.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher

EMPTY = -1
BLOCK = 256
BLOCKS_PER_SM = 8


@cuda.jit(device=True, forceinline=True)  # ty: ignore[unknown-argument]
def hash_key(key: np.int64) -> np.uint64:
    """Return the MurmurHash3 fmix64 finalizer of the 64-bit key."""
    h = types.uint64(key)
    h ^= h >> types.uint64(33)
    h *= types.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> types.uint64(33)
    h *= types.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> types.uint64(33)
    return h


@cuda.jit(device=True, forceinline=True)  # ty: ignore[unknown-argument]
def warp_lanes_mask() -> np.uint32:
    """Return the mask of the lanes that exist in the calling warp of a 1-D block."""
    lane = cuda.threadIdx.x & 31  # ty: ignore[unsupported-operator]
    lanes = min(32, cuda.blockDim.x - (cuda.threadIdx.x - lane))
    return types.uint32((1 << lanes) - 1)


def kernel(
    probe_keys: npt.NDArray[np.int64],
    table_keys: npt.NDArray[np.int64],
    table_rows: npt.NDArray[np.int32],
    capacity: int,
    out_build: npt.NDArray[np.int32],
    out_probe: npt.NDArray[np.int32],
    out_count: npt.NDArray[np.int32],
    m: int,
    out_cap: int,
) -> None:
    # The bitwise and gives the lane in [0, 31] without the sign handling
    # of the % operator.
    lane = cuda.threadIdx.x & 31  # ty: ignore[unsupported-operator]
    lanes_mask = warp_lanes_mask()
    lanes_below = (1 << lane) - 1
    mask = types.uint64(capacity - 1)
    stride = cuda.gridsize(1)
    # warp_start is the row of lane 0 of the warp. threadIdx.x & ~31 is the
    # thread index of lane 0, so warp_start is never negative. All lanes of a
    # warp run the same number of iterations, so every lane takes part in the
    # warp vote.
    warp_start = cuda.blockIdx.x * cuda.blockDim.x + (cuda.threadIdx.x & ~31)  # ty: ignore[unsupported-operator]
    while warp_start < m:
        j = warp_start + lane
        found = False
        build_row = types.int32(0)
        if j < m:
            key = probe_keys[j]
            # slot is at most mask = capacity - 1, so every table access
            # below stays inside the table.
            slot = hash_key(key) & mask
            while True:
                slot_key = table_keys[slot]
                if slot_key == EMPTY:
                    break
                if slot_key == key:
                    found = True
                    build_row = table_rows[slot]
                    break
                slot = (slot + types.uint64(1)) & mask
        hits = cuda.ballot_sync(lanes_mask, found) & 0xFFFFFFFF
        if found:
            rank = cuda.popc(hits & lanes_below)  # ty: ignore[too-many-positional-arguments]
            leader = cuda.ffs(hits) - 1  # ty: ignore[too-many-positional-arguments, unsupported-operator]
            first = types.int32(0)
            if rank == 0:
                first = cuda.atomic.add(out_count, 0, types.int32(cuda.popc(hits)))
            first = cuda.shfl_sync(types.uint32(hits), first, leader)
            position = first + rank
            if j < m and 0 <= position and position < out_cap:
                out_build[position] = build_row
                out_probe[position] = types.int32(j)
        warp_start += stride


TYPES: tuple[types.Type, ...] = (
    types.int64[:],
    types.int64[:],
    types.int32[:],
    types.int64,
    types.int32[:],
    types.int32[:],
    types.int32[:],
    types.int64,
    types.int64,
)

_kernel: CUDADispatcher = cuda.jit(kernel)


def run(
    probe_keys: DeviceNDArray,
    table_keys: DeviceNDArray,
    table_rows: DeviceNDArray,
    capacity: int,
    out_build: DeviceNDArray,
    out_probe: DeviceNDArray,
    out_count: DeviceNDArray,
    m: int,
    out_cap: int,
) -> None:
    """Probe the table with probe_keys[0:m]; out_count must hold 0 on entry."""
    if capacity < 1 or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two")
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid = max(1, min((m + BLOCK - 1) // BLOCK, BLOCKS_PER_SM * sms))
    _kernel[grid, BLOCK](
        probe_keys,
        table_keys,
        table_rows,
        capacity,
        out_build,
        out_probe,
        out_count,
        m,
        out_cap,
    )
