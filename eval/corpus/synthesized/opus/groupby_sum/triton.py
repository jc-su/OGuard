"""Group-by sum: hash aggregation of `value` per `group`.

The table uses open addressing with linear probing. The home slot of a group
is `fmix32(group) & (capacity - 1)`, as in the Numba and CUDA versions. Slot
`s` holds a group in `table_keys[s]` and the sum of its values in
`table_sums[s]`. A group value of -1 is not supported, because -1 marks an
empty slot.

Each program first adds up the values of each group within its tile of BLOCK
rows, so only one row per distinct group of the tile touches the table. This
keeps the number of atomic adds on one slot low when there are few groups.
The program sorts the rows of the tile by the high bits of the group hash,
with the row's lane in the low bits of the sort key, so the rows of a group
become one run. A hash collision of two groups can interleave their rows;
that only splits a group into more runs. A segmented scan sums each run, and
the last row of each run adds the run sum to the table.

To add a run sum, a row reads its current slot. If the slot is EMPTY, the row
tries to claim it for its group with a compare-and-swap. If the slot holds the
group, the row adds its sum with an atomic add; if it holds another group, the
row moves to the next slot. tl.atomic_cas has no mask, so claim_slot issues
the CAS as predicated inline PTX, only in the lanes that try to claim a slot
(see hash_build). The asm runs once for each element that a thread holds, so
BLOCK must be a multiple of 32 * num_warps.

The order of the floating-point additions depends on thread timing, so the
sums can differ from a sequential sum by rounding.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, and reports each call of a
@triton.jit function (fmix32, claim_slot, tl.sort, tl.max) as an argument
error. The lines where this occurs carry `ty: ignore` comments. A launch takes
host values (tensors, Python scalars, launch options), which the device-side
annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 512  # Rows per program.
NUM_WARPS = 4
EMPTY: tl.constexpr = tl.constexpr(-1)
if BLOCK % (32 * NUM_WARPS) != 0:
    raise ValueError("claim_slot requires BLOCK to be a multiple of 32 * NUM_WARPS")

TYPES = {
    "group": "*i32",
    "value": "*fp64",
    "table_keys": "*i32",
    "table_sums": "*fp64",
    "capacity": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def fmix32(group: tl.tensor) -> tl.tensor:
    """Return the MurmurHash3 fmix32 finalizer of the 32-bit groups as uint32."""
    h = group.to(tl.uint32, bitcast=True)
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


@triton.jit
def claim_slot(slot_key: tl.tensor, group: tl.tensor, attempt: tl.tensor) -> tl.tensor:
    """CAS each int32 slot from EMPTY (-1) to group in the lanes where attempt is set.

    Return the value that the slot held before the CAS. In the other lanes,
    do nothing and return 0.
    """
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .pred attempt;
        setp.ne.b32 attempt, $3, 0;
        mov.b32 $0, 0;
        @attempt atom.global.relaxed.gpu.cas.b32 $0, [$1], -1, $2;
        }""",
        constraints="=r,l,r,r",
        args=[slot_key, group, attempt.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def add_segments(
    a_start: tl.tensor, a_sum: tl.tensor, b_start: tl.tensor, b_sum: tl.tensor
) -> tuple[tl.tensor, tl.tensor]:
    """Combine two parts of a segmented sum; a start flag cuts the sum before its row."""
    return a_start | b_start, tl.where(b_start != 0, b_sum, a_sum + b_sum)


@triton.jit
def kernel(
    group: tl.tensor,
    value: tl.tensor,
    table_keys: tl.tensor,
    table_sums: tl.tensor,
    capacity: tl.int64,  # ty: ignore[invalid-type-form]
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    lane = tl.arange(0, BLOCK)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + lane
    valid = rows < n
    g = tl.load(group + rows, mask=valid, other=EMPTY)
    v = tl.load(value + rows, mask=valid, other=0.0)
    h = fmix32(g)  # ty: ignore[invalid-argument-type]

    # Sort the lanes by hash; order[i] is the lane of the i-th row in the sort.
    sort_key = (h & (0xFFFFFFFF - (BLOCK - 1))) | lane.to(tl.uint32)
    order = (tl.sort(sort_key) & (BLOCK - 1)).to(tl.int32)  # ty: ignore[invalid-argument-type]
    run_group = tl.gather(g, order, 0)
    run_hash = tl.gather(h, order, 0)
    before = tl.gather(run_group, tl.maximum(lane - 1, 0), 0)
    after = tl.gather(run_group, tl.minimum(lane + 1, BLOCK - 1), 0)
    start = ((lane == 0) | (run_group != before)).to(tl.int32)
    _, run_sum = tl.associative_scan((start, tl.gather(v, order, 0)), 0, add_segments)

    # The last row of each run adds the run sum to the table. The rows past n
    # hold the group EMPTY and add nothing.
    pending = ((lane == BLOCK - 1) | (run_group != after)) & (run_group != EMPTY)
    mask = capacity - 1
    slot = run_hash.to(tl.int64) & mask
    while tl.max(pending.to(tl.int32), 0) > 0:  # ty: ignore[invalid-argument-type]
        # A slot changes only from EMPTY to a group, so a stale read of a
        # group is never wrong; the CAS settles a stale read of EMPTY.
        slot_key = tl.load(table_keys + slot, mask=pending, other=EMPTY)
        attempt = pending & (slot_key == EMPTY)
        previous = claim_slot(table_keys + slot, run_group, attempt)  # ty: ignore[invalid-argument-type]
        slot_key = tl.where(attempt, previous, slot_key)
        # EMPTY here means that the CAS of this lane claimed the slot.
        mine = pending & ((slot_key == EMPTY) | (slot_key == run_group))
        tl.atomic_add(table_sums + slot, run_sum, mask=mine, sem="relaxed")
        pending = pending & ~mine
        slot = tl.where(pending, (slot + 1) & mask, slot)


def check_array(name: str, array: torch.Tensor, dtype: torch.dtype, length: int) -> None:
    """Raise ValueError unless array is a contiguous CUDA tensor of dtype and shape (length,)."""
    if (
        array.dtype != dtype
        or tuple(array.shape) != (length,)
        or not array.is_cuda
        or not array.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a contiguous CUDA tensor of {dtype} with shape ({length},), "
            f"got {array.dtype} with shape {tuple(array.shape)} on {array.device}"
        )


def run(
    group: torch.Tensor,
    value: torch.Tensor,
    table_keys: torch.Tensor,
    table_sums: torch.Tensor,
    capacity: int,
    n: int,
) -> None:
    """Add value[0:n] into the table; table_keys must hold EMPTY, table_sums 0."""
    if capacity < 1 or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two")
    check_array("group", group, torch.int32, n)
    check_array("value", value, torch.float64, n)
    check_array("table_keys", table_keys, torch.int32, capacity)
    check_array("table_sums", table_sums, torch.float64, capacity)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + BLOCK - 1) // BLOCK,)])
    launch(group, value, table_keys, table_sums, capacity, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
