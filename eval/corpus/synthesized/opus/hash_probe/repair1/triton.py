"""Hash probe: look up each probe key in the table that hash_build wrote.

The table uses open addressing with linear probing. The home slot of a key is
`fmix64(key) & (capacity - 1)`, the same as in hash_build. The build keys are
distinct, so each probe row matches at most one build row.

Each program probes a tile of BLOCK keys. In each step, every key of the tile
that is still searching reads its current slot: an EMPTY slot ends the search
without a match, a slot with the key ends it with a match, and any other key
moves the search to the next slot. The kernel checks for EMPTY before it
compares keys, so a probe key of -1 never matches. Then the program ranks its
matches with a prefix sum and reserves their output positions with one atomic
add on `out_count`. The pairs of a tile are in probe row order. `out_count`
receives the number of matches, also when it is larger than `out_cap`; the
kernel writes only the pairs at positions below `out_cap`.

A program writes a pair only when its position is in [0, out_cap). The
position is computed in int64, so the sum of the atomic result and the rank
does not wrap. If `out_count` holds 0 on entry, no position is negative, so
the check skips only the pairs at positions from `out_cap` on. The check
keeps each store inside `out_build` and `out_probe` for any value in
`out_count` on entry.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, and reports each call of a
@triton.jit function (fmix64, tl.max, tl.sum, tl.cumsum) as an argument error.
The lines where this occurs carry `ty: ignore` comments. A launch takes host
values (tensors, Python scalars, launch options), which the device-side
annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 32  # Probe keys per program: one warp, one key per lane.
NUM_WARPS = 1
EMPTY: tl.constexpr = tl.constexpr(-1)

TYPES = {
    "probe_keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "capacity": "i64",
    "out_build": "*i32",
    "out_probe": "*i32",
    "out_count": "*i32",
    "m": "i64",
    "out_cap": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def fmix64(key: tl.tensor) -> tl.tensor:
    """Return the MurmurHash3 fmix64 finalizer of the 64-bit keys as uint64."""
    h = key.to(tl.uint64, bitcast=True)
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


@triton.jit
def kernel(
    probe_keys: tl.tensor,
    table_keys: tl.tensor,
    table_rows: tl.tensor,
    capacity: tl.int64,  # ty: ignore[invalid-type-form]
    out_build: tl.tensor,
    out_probe: tl.tensor,
    out_count: tl.tensor,
    m: tl.int64,  # ty: ignore[invalid-type-form]
    out_cap: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    searching = rows < m
    key = tl.load(probe_keys + rows, mask=searching, other=EMPTY)
    mask = capacity - 1
    slot = (fmix64(key) & mask.to(tl.uint64)).to(tl.int64)  # ty: ignore[invalid-argument-type]
    found = tl.full([BLOCK], 0, tl.int1)
    while tl.max(searching.to(tl.int32), 0) > 0:  # ty: ignore[invalid-argument-type]
        slot_key = tl.load(table_keys + slot, mask=searching, other=EMPTY)
        occupied = slot_key != EMPTY
        found = found | (searching & occupied & (slot_key == key))
        searching = searching & occupied & (slot_key != key)
        slot = tl.where(searching, (slot + 1) & mask, slot)
    flag = found.to(tl.int32)
    matches = tl.sum(flag, 0)  # ty: ignore[invalid-argument-type]
    if matches > 0:
        first = tl.atomic_add(out_count, matches, sem="relaxed").to(tl.int64)
        # The exclusive prefix sum of the flags ranks the matches of the tile.
        position = first + (tl.cumsum(flag, 0) - flag)  # ty: ignore[invalid-argument-type]
        write = found & (position >= 0) & (position < out_cap)
        tl.store(out_build + position, tl.load(table_rows + slot, mask=write), mask=write)
        tl.store(out_probe + position, rows.to(tl.int32), mask=write)


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
    probe_keys: torch.Tensor,
    table_keys: torch.Tensor,
    table_rows: torch.Tensor,
    capacity: int,
    out_build: torch.Tensor,
    out_probe: torch.Tensor,
    out_count: torch.Tensor,
    m: int,
    out_cap: int,
) -> None:
    """Probe the table with probe_keys[0:m]; out_count must hold 0 on entry."""
    if capacity < 1 or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two")
    check_array("probe_keys", probe_keys, torch.int64, m)
    check_array("table_keys", table_keys, torch.int64, capacity)
    check_array("table_rows", table_rows, torch.int32, capacity)
    check_array("out_build", out_build, torch.int32, out_cap)
    check_array("out_probe", out_probe, torch.int32, out_cap)
    check_array("out_count", out_count, torch.int32, 1)
    if m == 0:
        return
    launch = cast(Callable[..., object], kernel[((m + BLOCK - 1) // BLOCK,)])
    launch(
        probe_keys,
        table_keys,
        table_rows,
        capacity,
        out_build,
        out_probe,
        out_count,
        m,
        out_cap,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
